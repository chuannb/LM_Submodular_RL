"""
Full Retrieval Pipeline

Query  →  BM25 (first-stage, fast)
       →  Qwen3-Embedding (second-stage, dense)  [optional, can fuse with BM25]
       →  Qwen3-Reranker (final reranking)
       →  Top-K results + log impression

Usage:
  pipeline = RetrievalPipeline.build(products, device="cuda")
  results, impression_id = pipeline.search(
      query="wireless headphone noise cancelling",
      session_id=session_id,
      top_k=10,
  )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from interaction.logger import EventType, InteractionLogger
from retrieval.bm25_retriever import BM25Retriever, SearchResult
from retrieval.embedding_retriever import EmbeddingRetriever
from retrieval.reranker import Qwen3Reranker, RankedResult


class RetrievalPipeline:
    """
    Full two-stage retrieval + reranking pipeline.

    Stage 1 (fast recall):
      BM25  → top N_bm25 candidates   (default: 200)
      Dense → top N_dense candidates  (default: 100)
      Fuse by Reciprocal Rank Fusion  → top N_fuse candidates

    Stage 2 (accurate reranking):
      Qwen3-Reranker → top K results

    Parameters
    ----------
    bm25_retriever       : BM25Retriever instance
    embedding_retriever  : EmbeddingRetriever instance (optional)
    reranker             : Qwen3Reranker instance
    logger               : InteractionLogger for logging impressions
    n_bm25               : BM25 recall size
    n_dense              : dense recall size
    n_fuse               : after fusion, how many to rerank
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        reranker: Qwen3Reranker,
        logger: Optional[InteractionLogger] = None,
        embedding_retriever: Optional[EmbeddingRetriever] = None,
        n_bm25: int = 200,
        n_dense: int = 100,
        n_fuse: int = 50,
    ):
        self.bm25 = bm25_retriever
        self.dense = embedding_retriever
        self.reranker = reranker
        self.logger = logger
        self.n_bm25 = n_bm25
        self.n_dense = n_dense
        self.n_fuse = n_fuse

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        products: List[Dict],
        device: str = "cpu",
        db_path: str = "interactions.db",
        bm25_backend: str = "rank_bm25",
        bm25_index_path: Optional[str] = None,
        dense_index_path: Optional[str] = None,
        embed_model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        reranker_model_id: str = "Qwen/Qwen3-Reranker-0.6B",
        use_dense: bool = True,
    ) -> "RetrievalPipeline":
        """
        Build the full pipeline from a product list.

        If index paths are provided, load existing indexes instead of rebuilding.
        """
        # BM25
        if bm25_index_path and bm25_backend == "rank_bm25":
            print("Loading BM25 index...")
            bm25 = BM25Retriever.load(bm25_index_path)
        else:
            print("Building BM25 index...")
            bm25 = BM25Retriever.build(products, backend=bm25_backend)
            if bm25_index_path and bm25_backend == "rank_bm25":
                bm25.save(bm25_index_path)

        # Dense
        dense = None
        if use_dense:
            dense = EmbeddingRetriever(model_id=embed_model_id, device=device)
            if dense_index_path:
                print("Loading dense index...")
                dense.load_index(dense_index_path)
            else:
                print("Building dense index...")
                dense.build_index(products)
                if dense_index_path:
                    dense.save_index(dense_index_path)

        # Reranker
        reranker = Qwen3Reranker(model_id=reranker_model_id, device=device)

        # Logger
        logger = InteractionLogger(db_path)

        return cls(
            bm25_retriever=bm25,
            reranker=reranker,
            logger=logger,
            embedding_retriever=dense,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: List[Tuple[str, float, str, str]],
        k: int = 60,
    ) -> List[Tuple[str, float, str, str]]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.
        RRF score: Σ 1/(k + rank_i)
        """
        scores: Dict[str, float] = {}
        meta: Dict[str, Tuple[str, str]] = {}   # item_id -> (title, text)

        for result_list in result_lists:
            for rank, (item_id, _, title, text) in enumerate(result_list, start=1):
                scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
                meta[item_id] = (title, text)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(iid, sc, *meta[iid]) for iid, sc in ranked]

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 10,
        page: int = 1,
    ) -> Tuple[List[RankedResult], Optional[str]]:
        """
        Execute full pipeline and return final results.

        Returns
        -------
        results       : List[RankedResult] sorted by reranker score
        impression_id : logged impression id (None if no logger)
        """
        # ---- Stage 1a: BM25 recall ----
        bm25_results = self.bm25.search(query, top_k=self.n_bm25)
        bm25_tuples = [(r.item_id, r.score, r.title, r.text) for r in bm25_results]

        # ---- Stage 1b: Dense recall (optional) ----
        if self.dense is not None:
            dense_results = self.dense.search(query, top_k=self.n_dense)
            fused = self._reciprocal_rank_fusion(bm25_tuples, dense_results)[: self.n_fuse]
        else:
            fused = bm25_tuples[: self.n_fuse]

        # ---- Stage 2: Rerank ----
        results = self.reranker.rerank(query, fused, top_k=top_k)

        # ---- Log impression ----
        impression_id = None
        if self.logger is not None and session_id is not None:
            impression_id = self.logger.log_impression(
                session_id=session_id,
                query=query,
                shown_items=[r.item_id for r in results],
                scores=[r.score for r in results],
                page=page,
            )

        return results, impression_id
