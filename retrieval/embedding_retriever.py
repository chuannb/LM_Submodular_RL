"""
Dense Retriever  —  Qwen3-Embedding-0.6B

Model: Qwen/Qwen3-Embedding-0.6B
  - Last-token pooling (EOS token position)
  - L2-normalised embeddings
  - Instruction-aware: prefix query with task description

Workflow:
  1. Offline: index product corpus → FAISS flat index
  2. Online:  embed query → FAISS search → Top-K candidates
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Qwen3 embedding helpers
# ---------------------------------------------------------------------------

EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

# Default task instruction (can be overridden)
DEFAULT_QUERY_INSTRUCTION = (
    "Given a product search query, retrieve relevant product titles and descriptions."
)


def _last_token_pool(
    last_hidden_states: torch.Tensor,   # (B, L, D)
    attention_mask: torch.Tensor,       # (B, L)
) -> torch.Tensor:                      # (B, D)
    """Pool the last non-padding token (right-padding assumed)."""
    seq_len = attention_mask.sum(dim=1) - 1          # index of last real token
    batch_size = last_hidden_states.shape[0]
    hidden = last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        seq_len,
    ]
    return hidden


def _format_query(query: str, instruction: str) -> str:
    """Wrap query with Qwen3 instruction prefix."""
    return f"Instruct: {instruction}\nQuery: {query}"


# ---------------------------------------------------------------------------
# Encoder (shared between retriever and reranker)
# ---------------------------------------------------------------------------

class Qwen3Encoder:
    """
    Wraps Qwen3-Embedding-0.6B for encoding queries and documents.

    Parameters
    ----------
    model_id   : HuggingFace model ID or local path
    device     : "cpu" | "cuda" | "mps"
    batch_size : max tokens per forward pass (batching)
    max_length : max token length per sequence
    """

    def __init__(
        self,
        model_id: str = EMBED_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"Loading {model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(device).eval()

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        is_query: bool = False,
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts.

        Parameters
        ----------
        texts       : list of strings (queries or documents)
        is_query    : if True, prepend instruction prefix (Qwen3 style)
        instruction : task instruction for query encoding
        normalize   : L2-normalise output vectors

        Returns
        -------
        embeddings : (N, D) float32 numpy array
        """
        if is_query:
            texts = [_format_query(t, instruction) for t in texts]

        all_embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            embs = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            if normalize:
                embs = F.normalize(embs, p=2, dim=-1)
            all_embeddings.append(embs.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# FAISS-backed dense index
# ---------------------------------------------------------------------------

@dataclass
class DenseIndex:
    embeddings: np.ndarray           # (N, D)
    item_ids: List[str]
    titles: List[str]
    texts: List[str]
    embed_dim: int = field(init=False)

    def __post_init__(self):
        self.embed_dim = self.embeddings.shape[1]

    def search(
        self,
        query_emb: np.ndarray,   # (D,) or (1, D)
        top_k: int = 50,
    ) -> List[Tuple[str, float, str, str]]:
        """
        Exact cosine search (using numpy dot product on L2-normed vectors).
        For large corpora use FAISS – see build_faiss_index().

        Returns list of (item_id, score, title, text).
        """
        if query_emb.ndim == 1:
            query_emb = query_emb[np.newaxis, :]   # (1, D)
        scores = (query_emb @ self.embeddings.T).squeeze(0)   # (N,)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            (self.item_ids[i], float(scores[i]), self.titles[i], self.texts[i])
            for i in top_idx
        ]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "DenseIndex":
        with open(path, "rb") as f:
            return pickle.load(f)

    def build_faiss_index(self):
        """
        Build FAISS IndexFlatIP (inner product = cosine on L2-normed vecs).
        Returns the faiss index object (not saved here – caller stores it).
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError("Install faiss: pip install faiss-cpu") from e

        d = self.embed_dim
        index = faiss.IndexFlatIP(d)
        index.add(self.embeddings.astype(np.float32))
        return index


# ---------------------------------------------------------------------------
# Public API: EmbeddingRetriever
# ---------------------------------------------------------------------------

class EmbeddingRetriever:
    """
    Dense retriever using Qwen3-Embedding-0.6B.

    Typical usage:
      # Build index (offline)
      retriever = EmbeddingRetriever(device="cuda")
      retriever.build_index(products)
      retriever.save_index("product_index.pkl")

      # Search (online)
      retriever = EmbeddingRetriever(device="cuda")
      retriever.load_index("product_index.pkl")
      results = retriever.search("wireless headphone", top_k=50)
    """

    def __init__(
        self,
        model_id: str = EMBED_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.encoder = Qwen3Encoder(model_id, device, batch_size, max_length)
        self.index: Optional[DenseIndex] = None

    # ------------------------------------------------------------------
    def build_index(
        self,
        products: List[Dict],
        doc_batch_size: int = 256,
    ) -> None:
        """
        Encode all products and build in-memory dense index.

        Each product dict must have: item_id, title, (optionally) description, brand, etc.
        """
        ids, titles, texts = [], [], []
        for p in products:
            ids.append(str(p["item_id"]))
            titles.append(str(p.get("title", "")))
            doc_text = " ".join([
                str(p.get("title", "")),
                str(p.get("brand", "")),
                str(p.get("description", "")),
            ])
            texts.append(doc_text)

        print(f"Encoding {len(products)} products...")
        embs = self.encoder.encode(texts, is_query=False, batch_size=doc_batch_size
                                   if hasattr(self.encoder, 'batch_size') else 32)

        self.index = DenseIndex(
            embeddings=embs,
            item_ids=ids,
            titles=titles,
            texts=texts,
        )
        print(f"Index built: {embs.shape}")

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 50,
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> List[Tuple[str, float, str, str]]:
        """
        Search for top-k products matching the query.
        Returns list of (item_id, score, title, text).
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")
        query_emb = self.encoder.encode([query], is_query=True, instruction=instruction)
        return self.index.search(query_emb[0], top_k=top_k)

    def save_index(self, path: str) -> None:
        if self.index is None:
            raise RuntimeError("No index to save.")
        self.index.save(path)

    def load_index(self, path: str) -> None:
        self.index = DenseIndex.load(path)
