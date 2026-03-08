"""
First-Stage BM25 Retriever.

Hỗ trợ hai backend:
  1. Pyserini  — production, cần Java + Anserini index trên disk
  2. rank_bm25 — đơn giản, full in-memory, dùng để dev / test nhanh

API thống nhất: BM25Retriever.search(query, top_k) -> List[SearchResult]

Xây index:
  retriever = BM25Retriever.build(products, backend="rank_bm25")
  retriever.search("wireless headphone", top_k=100)
"""

from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    item_id: str
    score: float
    title: str = ""
    text: str = ""   # concatenated searchable text


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _build_doc_text(product: Dict[str, Any]) -> str:
    """Concatenate product fields into a single searchable string."""
    parts = [
        str(product.get("title", "")),
        str(product.get("brand", "")),
        str(product.get("description", "")),
        " ".join(product.get("categories", []) if isinstance(product.get("categories"), list) else []),
        " ".join(product.get("feature", []) if isinstance(product.get("feature"), list) else []),
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# rank_bm25 backend (always available, no Java required)
# ---------------------------------------------------------------------------

class _RankBM25Backend:
    def __init__(self, products: List[Dict[str, Any]]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("Install rank_bm25: pip install rank-bm25") from e

        self.products = products
        self.ids = [str(p["item_id"]) for p in products]
        self.texts = [_build_doc_text(p) for p in products]
        corpus_tokens = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int = 100) -> List[SearchResult]:
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_idx:
            p = self.products[idx]
            results.append(SearchResult(
                item_id=self.ids[idx],
                score=float(scores[idx]),
                title=str(p.get("title", "")),
                text=self.texts[idx],
            ))
        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "_RankBM25Backend":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Pyserini backend (production – requires Java 11+ and pyserini)
# ---------------------------------------------------------------------------

class _PyseriniBackend:
    """
    Wraps Pyserini SimpleSearcher for BM25.

    Index must be pre-built. To build:
      python -m pyserini.index.lucene \
          --collection JsonCollection \
          --input  <jsonl_dir> \
          --index  <index_dir> \
          --generator DefaultLuceneDocumentGenerator \
          --storeContents
    """

    def __init__(self, index_dir: str):
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as e:
            raise ImportError("Install pyserini: pip install pyserini") from e

        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(k1=0.9, b=0.4)

    def search(self, query: str, top_k: int = 100) -> List[SearchResult]:
        hits = self.searcher.search(query, k=top_k)
        results = []
        for hit in hits:
            raw = hit.raw
            if raw:
                try:
                    doc = json.loads(raw)
                except Exception:
                    doc = {}
            else:
                doc = {}
            results.append(SearchResult(
                item_id=hit.docid,
                score=float(hit.score),
                title=doc.get("title", ""),
                text=doc.get("contents", ""),
            ))
        return results

    @staticmethod
    def build_jsonl(products: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Dump products to JSONL format ready for Pyserini indexing.
        Each line: {"id": item_id, "contents": full_text}
        """
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "products.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for p in products:
                doc = {
                    "id": str(p["item_id"]),
                    "contents": _build_doc_text(p),
                    "title": str(p.get("title", "")),
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"Wrote {len(products)} products to {out_path}")
        print("Run pyserini indexing:")
        print(f"  python -m pyserini.index.lucene --collection JsonCollection "
              f"--input {output_dir} --index <index_dir> "
              f"--generator DefaultLuceneDocumentGenerator --storeContents")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Unified BM25 retriever.

    Usage:
      # Build in-memory (dev)
      retriever = BM25Retriever.build(products, backend="rank_bm25")

      # Use Pyserini (production)
      retriever = BM25Retriever.from_pyserini_index("/path/to/index")

      # Search
      results = retriever.search("wireless headphone", top_k=100)
    """

    def __init__(self, backend: Any):
        self._backend = backend

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        products: List[Dict[str, Any]],
        backend: str = "rank_bm25",
        pyserini_index_dir: Optional[str] = None,
    ) -> "BM25Retriever":
        if backend == "pyserini":
            if pyserini_index_dir is None:
                raise ValueError("pyserini_index_dir is required for pyserini backend")
            return cls(_PyseriniBackend(pyserini_index_dir))
        else:
            return cls(_RankBM25Backend(products))

    @classmethod
    def from_pyserini_index(cls, index_dir: str) -> "BM25Retriever":
        return cls(_PyseriniBackend(index_dir))

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        """Load a previously saved rank_bm25 index."""
        return cls(_RankBM25Backend.load(path))

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 100) -> List[SearchResult]:
        return self._backend.search(query, top_k)

    def save(self, path: str) -> None:
        """Persist rank_bm25 index to disk (not applicable for Pyserini)."""
        if isinstance(self._backend, _RankBM25Backend):
            self._backend.save(path)
        else:
            raise NotImplementedError("Pyserini index is managed externally.")
