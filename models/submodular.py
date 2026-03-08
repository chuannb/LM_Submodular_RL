"""
Submodular Utility Function  f_θ

Hai class:

1. SubmodularUtility  (pipeline RL thuần túy)
   f_θ(S|x) = α * MLP_relevance(S, x) + (1-α) * diversity(S)

2. RerankerBackedSubmodular  (unified pipeline – MỚI)
   Relevance đến từ Qwen3-Reranker (external scores), không cần MLP:

   f_θ(S | q, x) = α_t * reranker_rel(S, q)
                 + (1-α_t) * div_θ(S)

   div_θ(S) = (1/|S|(|S|-1)) Σ_{i≠j} (1 - k_θ(e_i, e_j))
   k_θ      = RBF kernel với bandwidth học được và item embeddings học được

   Parameters θ:
     - item_emb      : diversity embeddings (d chiều, tách khỏi relevance)
     - log_bandwidth : RBF bandwidth (scalar)
   RL controls:
     - α_t : relevance–diversity trade-off  (từ Actor)
     - κ_t : exploration temperature trong greedy (từ Actor)

   Gradients cho θ:
     L_sub   = L_reinforce + λ_rank * L_div_rank
     L_reinforce = -Σ_t r_t * log f_θ(S_t | q_t, x_t)   (REINFORCE)
     L_div_rank  = margin loss trên diversity: items khác category nên xa nhau
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


# ===========================================================================
# RerankerBackedSubmodular  (unified pipeline)
# ===========================================================================

class RerankerBackedSubmodular(nn.Module):
    """
    Submodular utility where relevance = Qwen3-Reranker scores (external).
    Chỉ học diversity parameters θ = (item_emb, log_bandwidth).

    f_θ(S | q, x) = α * mean_{i∈S}[rel(i)]   # reranker p(yes)
                  + (1-α) * mean_{i≠j∈S}[1 - k_θ(e_i, e_j)]

    Parameters
    ----------
    num_items      : catalogue size
    embed_dim      : diversity embedding dimension
    alpha_init     : initial relevance weight
    kernel         : "rbf" | "dot"
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        alpha_init: float = 0.7,
        kernel: str = "rbf",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel = kernel

        # Diversity-only embeddings (không dùng cho relevance)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)

        # Learnable RBF bandwidth
        self.log_bandwidth = nn.Parameter(torch.zeros(1))

        # Default α (override at runtime by RL knob)
        self.log_alpha = nn.Parameter(
            torch.tensor(
                [float(torch.log(torch.tensor(alpha_init / (1.0 - alpha_init))))],
                dtype=torch.float32,
            )
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha)

    # ------------------------------------------------------------------
    def diversity_scores_for_item(
        self,
        item_id: int,
        slate_ids: List[int],
        device: torch.device,
    ) -> float:
        """
        Mean diversity of item i w.r.t. existing slate S.
        Returns 0 if slate is empty.
        """
        if not slate_ids:
            return 0.0
        with torch.no_grad():
            e_i = self.item_emb(torch.tensor([item_id], device=device))  # (1, d)
            e_s = self.item_emb(torch.tensor(slate_ids, device=device))  # (|S|, d)
            e_i = F.normalize(e_i, dim=-1)
            e_s = F.normalize(e_s, dim=-1)
            sim = (e_i @ e_s.T).squeeze(0)   # (|S|,)
            if self.kernel == "rbf":
                bw = torch.exp(self.log_bandwidth).clamp(min=1e-3)
                k = torch.exp(-(1.0 - sim) / bw)
            else:
                k = (sim + 1.0) / 2.0
            return (1.0 - k).mean().item()

    # ------------------------------------------------------------------
    def evaluate_with_scores(
        self,
        slate_ids: List[int],
        reranker_scores: List[float],   # precomputed p(yes) per item
        alpha_override: float = None,
    ) -> float:
        """
        Evaluate f_θ(S) given external reranker relevance scores.
        Used inside greedy to score full slates.
        """
        if not slate_ids:
            return 0.0

        rel = float(sum(reranker_scores) / len(reranker_scores))

        if len(slate_ids) >= 2:
            device = self.item_emb.weight.device
            with torch.no_grad():
                ids_t = torch.tensor(slate_ids, device=device)
                embs = F.normalize(self.item_emb(ids_t), dim=-1)   # (k, d)
                sim = embs @ embs.T                                  # (k, k)
                if self.kernel == "rbf":
                    bw = torch.exp(self.log_bandwidth).clamp(min=1e-3)
                    k_mat = torch.exp(-(1.0 - sim) / bw)
                else:
                    k_mat = (sim + 1.0) / 2.0
                mask = ~torch.eye(len(slate_ids), dtype=torch.bool, device=device)
                div = (1.0 - k_mat)[mask].mean().item()
        else:
            div = 0.0

        alpha = alpha_override if alpha_override is not None else self.alpha.item()
        return alpha * rel + (1.0 - alpha) * div

    # ------------------------------------------------------------------
    def marginal_gain_with_scores(
        self,
        slate_ids: List[int],
        slate_rel_scores: List[float],
        new_item: int,
        new_item_rel: float,
        alpha_override: float = None,
    ) -> float:
        """
        Marginal gain of adding item i to slate S, using precomputed reranker scores.
        Δ f_θ(S∪{i}) - f_θ(S)
        """
        score_with = self.evaluate_with_scores(
            slate_ids + [new_item], slate_rel_scores + [new_item_rel], alpha_override
        )
        score_without = self.evaluate_with_scores(
            slate_ids, slate_rel_scores, alpha_override
        )
        return score_with - score_without

    # ------------------------------------------------------------------
    def soft_slate_score(
        self,
        item_ids: torch.Tensor,          # (B, K) — slate items
        reranker_scores: torch.Tensor,   # (B, K) — precomputed rel scores
        alpha: torch.Tensor,             # (B,) or scalar
    ) -> torch.Tensor:                   # (B,) — differentiable f_θ value
        """
        Differentiable version used for computing gradients through θ.

        relevance = mean of reranker_scores (no gradient through reranker)
        diversity = mean pairwise distance in diversity embedding space (has gradient)
        """
        B, K = item_ids.shape
        embs = self.item_emb(item_ids)          # (B, K, d)
        embs_norm = F.normalize(embs, dim=-1)   # (B, K, d)

        # Pairwise similarities
        sim = torch.bmm(embs_norm, embs_norm.transpose(1, 2))  # (B, K, K)
        if self.kernel == "rbf":
            bw = torch.exp(self.log_bandwidth).clamp(min=1e-3)
            k_mat = torch.exp(-(1.0 - sim) / bw)
        else:
            k_mat = (sim + 1.0) / 2.0

        # Mask diagonal
        eye = torch.eye(K, dtype=torch.bool, device=item_ids.device)
        k_mat = k_mat.masked_fill(eye.unsqueeze(0), 0.0)
        div = (1.0 - k_mat).sum(dim=(1, 2)) / (K * (K - 1) + 1e-8)   # (B,)

        rel = reranker_scores.mean(dim=-1).detach()   # (B,) — reranker detached

        if isinstance(alpha, float):
            return alpha * rel + (1.0 - alpha) * div
        alpha_b = alpha.view(B)
        return alpha_b * rel + (1.0 - alpha_b) * div

    # ------------------------------------------------------------------
    def diversity_ranking_loss(
        self,
        pos_ids: torch.Tensor,    # (B,) positive item ids
        neg_ids: torch.Tensor,    # (B, num_neg) negative item ids
        margin: float = 0.5,
    ) -> torch.Tensor:
        """
        Pushes apart diversity embeddings of items from different groups.
        Used as L_div_rank to prevent embedding collapse.

        Simple contrastive: diverse pairs (random) should be far,
        similar pairs (same batch positive) should be relatively close.
        No category labels needed: uses in-batch structure.
        """
        B = pos_ids.shape[0]
        num_neg = neg_ids.shape[1]

        pos_embs = F.normalize(self.item_emb(pos_ids), dim=-1)           # (B, d)
        neg_embs = F.normalize(self.item_emb(neg_ids.view(-1)), dim=-1)  # (B*num_neg, d)
        neg_embs = neg_embs.view(B, num_neg, self.embed_dim)             # (B, num_neg, d)

        # Pos-pos similarity (should stay moderate, not collapse to 1)
        pp_sim = pos_embs @ pos_embs.T  # (B, B)
        pp_mask = ~torch.eye(B, dtype=torch.bool, device=pos_ids.device)
        pp_loss = F.relu(pp_sim[pp_mask] - (1.0 - margin)).mean()

        # Pos-neg should be lower than pos-pos → pushes negatives away
        pn_sim = (pos_embs.unsqueeze(1) * neg_embs).sum(dim=-1)  # (B, num_neg)
        pn_loss = F.relu(pn_sim + margin).mean()

        return pp_loss + pn_loss
