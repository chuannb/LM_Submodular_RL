"""
Reranker  —  Qwen3-Reranker-0.6B

Model: Qwen/Qwen3-Reranker-0.6B
  - Cross-encoder: takes (query, document) pairs -> relevance score
  - Special prompt format with system instruction
  - Outputs logit for "yes" token as relevance score

Usage:
  reranker = Qwen3Reranker(device="cuda")
  scored = reranker.rerank(query, candidates, top_k=10)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"

# Qwen3-Reranker uses this special system + user prompt format
_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements of the Query. "
    "The answer can only be 'yes' or 'no', and you should output 'yes' or 'no' "
    "at the end on a new line."
)

_USER_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "Query: {query}\n"
    "Document: {document}\n"
    "Does the document meet the requirements of the query?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass
class RankedResult:
    item_id: str
    score: float    # higher = more relevant
    title: str = ""
    text: str = ""


class Qwen3Reranker:
    """
    Qwen3-Reranker-0.6B cross-encoder reranker.

    Scores (query, document) pairs by extracting the logit for token "yes"
    from the final assistant token position.

    Parameters
    ----------
    model_id   : HuggingFace model ID or local path
    device     : "cpu" | "cuda" | "mps"
    batch_size : number of (query, doc) pairs per forward pass
    max_length : max token length
    """

    def __init__(
        self,
        model_id: str = RERANKER_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 1024,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"Loading {model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(device).eval()

        # Token ids for "yes" and "no"
        self._yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self._no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]

    # ------------------------------------------------------------------
    def _build_prompts(self, query: str, documents: List[str]) -> List[str]:
        return [
            _USER_TEMPLATE.format(
                system=_SYSTEM_PROMPT,
                query=query,
                document=doc,
            )
            for doc in documents
        ]

    @torch.no_grad()
    def _score_batch(self, prompts: List[str]) -> List[float]:
        """
        Forward pass on a batch of prompts.
        Returns list of relevance scores (log-prob of "yes").
        """
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        # logits at last token position: (B, vocab_size)
        last_logits = outputs.logits[:, -1, :]
        # Extract yes/no logits and compute p(yes)
        yes_no = last_logits[:, [self._yes_id, self._no_id]]   # (B, 2)
        probs = F.softmax(yes_no, dim=-1)                       # (B, 2)
        scores = probs[:, 0].cpu().float().tolist()             # p(yes)
        return scores

    # ------------------------------------------------------------------
    def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Score a list of documents against a query.
        Returns list of float scores in the same order as documents.
        """
        prompts = self._build_prompts(query, documents)
        all_scores: List[float] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start: start + self.batch_size]
            all_scores.extend(self._score_batch(batch))
        return all_scores

    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, str, str]],  # (item_id, bm25_score, title, text)
        top_k: Optional[int] = None,
    ) -> List[RankedResult]:
        """
        Rerank a list of candidates for a given query.

        Parameters
        ----------
        query      : search query
        candidates : list of (item_id, retrieval_score, title, text)
        top_k      : return only top_k results; None = return all

        Returns
        -------
        List of RankedResult sorted by score descending.
        """
        if not candidates:
            return []

        item_ids = [c[0] for c in candidates]
        titles = [c[2] for c in candidates]
        texts = [c[3] for c in candidates]

        scores = self.score(query, texts)

        results = [
            RankedResult(item_id=item_ids[i], score=scores[i], title=titles[i], text=texts[i])
            for i in range(len(candidates))
        ]
        results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]
        return results


# ---------------------------------------------------------------------------
# Fine-tunable wrapper (used by reranker_trainer.py)
# ---------------------------------------------------------------------------

class Qwen3RerankerForTraining(torch.nn.Module):
    """
    Thin wrapper that exposes the reranker's language model for fine-tuning.
    Used in reranker_trainer.py for fine-tuning on pairwise preference data.
    """

    def __init__(self, model_id: str = RERANKER_MODEL_ID, device: str = "cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        )
        self.device = device
        self._yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self._no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]

    def forward_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning p(yes) for each (query, doc) pair.
        Shape: (B,)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_logits = outputs.logits[:, -1, :]
        yes_no = last_logits[:, [self._yes_id, self._no_id]]
        probs = F.softmax(yes_no.float(), dim=-1)
        return probs[:, 0]   # p(yes)

    def encode_pairs(
        self,
        queries: List[str],
        documents: List[str],
        max_length: int = 1024,
    ) -> dict:
        prompts = [
            _USER_TEMPLATE.format(system=_SYSTEM_PROMPT, query=q, document=d)
            for q, d in zip(queries, documents)
        ]
        return self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
