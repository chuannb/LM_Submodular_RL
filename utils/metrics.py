"""
Evaluation metrics for slate-based recommendation.

Metrics:
  hit_at_k         : 1 if ground-truth item is in the recommended slate
  ndcg_at_k        : normalised discounted cumulative gain
  precision_at_k   : fraction of slate items that are relevant
  diversity_score  : mean pairwise distance between slate embeddings
  coverage         : fraction of items seen at least once across all slates
"""

from typing import List, Set

import numpy as np
import torch


def hit_at_k(slate: List[int], target: int) -> float:
    """1.0 if target appears in the slate, else 0.0."""
    return 1.0 if target in slate else 0.0


def ndcg_at_k(slate: List[int], target: int) -> float:
    """NDCG@k for a single relevant item (binary relevance)."""
    if target not in slate:
        return 0.0
    rank = slate.index(target) + 1  # 1-indexed
    return 1.0 / np.log2(rank + 1)


def precision_at_k(slate: List[int], targets: Set[int]) -> float:
    """Fraction of slate items that are in the target set."""
    if not slate:
        return 0.0
    hits = sum(1 for i in slate if i in targets)
    return hits / len(slate)


def diversity_score(
    slate: List[int],
    embeddings: torch.Tensor,  # (num_items, embed_dim)
) -> float:
    """
    Mean pairwise cosine distance between item embeddings in the slate.
    Returns 0 if slate has < 2 items.
    """
    if len(slate) < 2:
        return 0.0
    slate_embs = embeddings[slate]           # (k, d)
    slate_embs = torch.nn.functional.normalize(slate_embs, dim=-1)
    sim = slate_embs @ slate_embs.T          # (k, k)
    k = len(slate)
    mask = ~torch.eye(k, dtype=torch.bool, device=sim.device)
    mean_sim = sim[mask].mean().item()
    return 1.0 - mean_sim                    # distance = 1 - cosine_similarity


def coverage(
    all_slates: List[List[int]],
    num_items: int,
) -> float:
    """Fraction of items recommended at least once."""
    seen: Set[int] = set()
    for slate in all_slates:
        seen.update(slate)
    return len(seen) / num_items


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------

class SlateMetrics:
    """Accumulate metrics over many prediction steps and compute averages."""

    def __init__(self):
        self.hits: List[float] = []
        self.ndcgs: List[float] = []
        self.all_slates: List[List[int]] = []

    def update(self, slate: List[int], target: int) -> None:
        self.hits.append(hit_at_k(slate, target))
        self.ndcgs.append(ndcg_at_k(slate, target))
        self.all_slates.append(slate)

    def compute(self, num_items: int) -> dict:
        return {
            "hit@k": float(np.mean(self.hits)) if self.hits else 0.0,
            "ndcg@k": float(np.mean(self.ndcgs)) if self.ndcgs else 0.0,
            "coverage": coverage(self.all_slates, num_items),
            "n_samples": len(self.hits),
        }

    def reset(self) -> None:
        self.hits.clear()
        self.ndcgs.clear()
        self.all_slates.clear()
