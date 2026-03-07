"""
Submodular Utility Function  f_θ

f_θ(S | x) models the quality of a slate S given context x.

Formulation (coverage / facility location style):
  f_θ(S | x) = Σ_{u in U} max_{i in S} w_u * sim(i, u)
             = relevance term + diversity term (weighted by α_t)

Concretely:
  f_θ(S | x) = α * relevance(S, x) + (1-α) * diversity(S)

  relevance(S, x) = (1/|S|) Σ_{i in S} score(i | x)
  diversity(S)    = (1/|S|*(|S|-1)) Σ_{i≠j in S} (1 - cos_sim(e_i, e_j))

The trade-off α is controlled by the RL knob α_t via:
  α_t = sigmoid(raw_α) * (α_max - α_min) + α_min

Parameters θ:
  - Learnable relevance scoring network (MLP over item + context)
  - Item embeddings shared with generator
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubmodularUtility(nn.Module):
    """
    Differentiable approximation of the submodular utility.

    Parameters
    ----------
    num_items      : catalogue size
    embed_dim      : item embedding dimension
    hidden_dim     : hidden dimension for relevance MLP
    alpha_init     : initial relevance weight (0-1)
    kernel         : "rbf" or "dot" for diversity kernel
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        alpha_init: float = 0.7,
        kernel: str = "rbf",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel = kernel

        # Shared item embeddings (can be tied with generator)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        # Relevance scoring: MLP(context || item_emb) -> scalar
        self.relevance_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Learnable log-alpha (unconstrained; mapped to [0,1] via sigmoid)
        self.log_alpha = nn.Parameter(
            torch.tensor([float(torch.log(torch.tensor(alpha_init / (1 - alpha_init))))],
                         dtype=torch.float32)
        )

        # RBF bandwidth
        self.log_bandwidth = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha)

    # ------------------------------------------------------------------
    def relevance_scores(
        self,
        item_ids: torch.Tensor,   # (M,) or (B, M)
        context: torch.Tensor,    # (embed_dim,) or (B, embed_dim)
    ) -> torch.Tensor:            # (M,) or (B, M)
        """Score each candidate item given context."""
        item_embs = self.item_emb(item_ids)   # (..., embed_dim)
        if context.dim() == 1:
            ctx = context.unsqueeze(0).expand_as(item_embs)
        else:
            # (B, embed_dim) -> (B, M, embed_dim)
            ctx = context.unsqueeze(1).expand_as(item_embs)
        pairs = torch.cat([item_embs, ctx], dim=-1)
        scores = self.relevance_mlp(pairs).squeeze(-1)   # (..., M)
        return scores

    # ------------------------------------------------------------------
    def diversity_matrix(
        self,
        item_ids: torch.Tensor,   # (M,)
    ) -> torch.Tensor:             # (M, M) pairwise distance
        embs = self.item_emb(item_ids)       # (M, embed_dim)
        embs = F.normalize(embs, dim=-1)
        sim = embs @ embs.T                  # (M, M)
        if self.kernel == "rbf":
            bandwidth = torch.exp(self.log_bandwidth).clamp(min=1e-3)
            dist = (1.0 - sim) / bandwidth
            kernel_val = torch.exp(-dist)
        else:
            kernel_val = (sim + 1.0) / 2.0  # dot -> [0,1]
        return 1.0 - kernel_val             # diversity (higher = more diverse)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        slate_ids: List[int],
        context: torch.Tensor,     # (embed_dim,)
        alpha_override: float = None,
    ) -> float:
        """
        Evaluate f_θ(S | x) for a given slate (list of item ids).
        Returns a Python float (used inside greedy search).
        """
        if not slate_ids:
            return 0.0
        with torch.no_grad():
            ids = torch.tensor(slate_ids, dtype=torch.long, device=context.device)
            rel_scores = self.relevance_scores(ids, context)   # (|S|,)
            rel = rel_scores.mean().item()

            if len(slate_ids) >= 2:
                div_mat = self.diversity_matrix(ids)           # (|S|, |S|)
                mask = ~torch.eye(len(slate_ids), dtype=torch.bool, device=div_mat.device)
                div = div_mat[mask].mean().item()
            else:
                div = 0.0

        alpha = alpha_override if alpha_override is not None else self.alpha.item()
        return alpha * rel + (1.0 - alpha) * div

    # ------------------------------------------------------------------
    def marginal_gain(
        self,
        slate_ids: List[int],
        new_item: int,
        context: torch.Tensor,
        alpha_override: float = None,
    ) -> float:
        """
        Δ f_θ(S ∪ {i} | x) - f_θ(S | x)   (marginal gain of adding item i).
        """
        score_with = self.evaluate(slate_ids + [new_item], context, alpha_override)
        score_without = self.evaluate(slate_ids, context, alpha_override)
        return score_with - score_without

    # ------------------------------------------------------------------
    def ranking_loss(
        self,
        context: torch.Tensor,          # (B, embed_dim)
        pos_ids: torch.Tensor,           # (B,) positive item id per sample
        neg_ids: torch.Tensor,           # (B, num_neg) negative item ids
        alpha_override: float = None,
    ) -> torch.Tensor:
        """
        Pairwise ranking loss: score(pos) should be > score(neg).
        Used to update θ from training trajectories.
        """
        B, num_neg = neg_ids.shape

        pos_embs = self.item_emb(pos_ids)                  # (B, embed_dim)
        neg_embs = self.item_emb(neg_ids)                  # (B, num_neg, embed_dim)

        ctx_pos = context                                   # (B, embed_dim)
        ctx_neg = context.unsqueeze(1).expand(B, num_neg, -1)

        pos_pairs = torch.cat([pos_embs, ctx_pos], dim=-1)       # (B, 2*d)
        neg_pairs = torch.cat([neg_embs, ctx_neg], dim=-1)       # (B, num_neg, 2*d)

        pos_scores = self.relevance_mlp(pos_pairs).squeeze(-1)   # (B,)
        neg_scores = self.relevance_mlp(neg_pairs).squeeze(-1)   # (B, num_neg)

        # BPR-style: -log σ(pos - neg)
        diff = pos_scores.unsqueeze(1) - neg_scores              # (B, num_neg)
        loss = -F.logsigmoid(diff).mean()

        # Diversity regularisation: encourage spread in positive items
        alpha = alpha_override if alpha_override is not None else self.alpha.item()
        div_reg = 0.0
        if alpha < 1.0:
            pos_norm = F.normalize(pos_embs, dim=-1)
            sim = pos_norm @ pos_norm.T
            div_reg = sim.mean()

        return loss + (1.0 - alpha) * div_reg
