"""
Algorithm 2: BudgetedSubmodularGreedy  (Knapsack Greedy)

Selects a slate S ⊆ C of size ≤ k subject to:
  Σ_{i ∈ S} c(i) ≤ B

At each step greedily picks:
  i* = argmax_{i ∈ F} q(i)    with probability 1 - ε(κ)
     OR
  i* ~ Softmax(q(i) / τ(κ))  with probability   ε(κ)

where:
  q(i) = Δ(i) / c(i)         (benefit-to-cost ratio)
  Δ(i) = f(S ∪ {i} | x) - f(S | x)  (marginal gain)

  ε(κ): exploration probability controlled by RL knob κ
  τ(κ): softmax temperature controlled by RL knob κ
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.submodular import SubmodularUtility, RerankerBackedSubmodular


def eps_from_kappa(kappa: float) -> float:
    """Map κ ∈ [0,1] to exploration probability ε ∈ [0, 0.5]."""
    return 0.5 * float(kappa)


def tau_from_kappa(kappa: float, tau_min: float = 0.1, tau_max: float = 5.0) -> float:
    """Map κ ∈ [0,1] to softmax temperature τ."""
    return tau_min + (tau_max - tau_min) * float(kappa)


def budgeted_submodular_greedy(
    candidates: List[int],           # C: candidate item ids
    utility: SubmodularUtility,      # f_θ
    context: torch.Tensor,           # x_t: (embed_dim,) state context
    slate_size: int,                 # k: max slate size
    budget: float,                   # B_t: budget constraint
    costs: Optional[Dict[int, float]] = None,  # c(i) per item; None = uniform 1
    alpha_override: Optional[float] = None,    # α from RL knob
    kappa: float = 0.0,              # exploration knob from RL
) -> Tuple[List[int], float]:
    """
    Run Knapsack Greedy on the submodular utility.

    Returns
    -------
    slate        : selected item ids
    final_score  : f_θ(slate | x)
    """
    if costs is None:
        costs = {i: 1.0 for i in candidates}

    slate: List[int] = []
    spent: float = 0.0
    eps = eps_from_kappa(kappa)
    tau = tau_from_kappa(kappa)

    for _ in range(slate_size):
        # Feasible set F
        feasible = [i for i in candidates if i not in slate and spent + costs.get(i, 1.0) <= budget]
        if not feasible:
            break

        # Marginal gains
        gains = {
            i: utility.marginal_gain(slate, i, context, alpha_override)
            for i in feasible
        }
        # Benefit-to-cost ratio
        ratios = {i: (gains[i] / max(costs.get(i, 1.0), 1e-8)) for i in feasible}

        # Epsilon-greedy with softmax exploration
        if random.random() < eps:
            # Stochastic: sample from softmax(q / τ)
            items = list(ratios.keys())
            q_vals = np.array([ratios[i] for i in items], dtype=np.float32)
            q_vals = q_vals / tau
            q_vals -= q_vals.max()   # numerical stability
            probs = np.exp(q_vals)
            probs /= probs.sum()
            i_star = int(np.random.choice(items, p=probs))
        else:
            # Greedy: pick max ratio
            i_star = max(ratios, key=ratios.__getitem__)

        slate.append(i_star)
        spent += costs.get(i_star, 1.0)

    final_score = utility.evaluate(slate, context, alpha_override)
    return slate, final_score


# ---------------------------------------------------------------------------
# Greedy variant for RerankerBackedSubmodular (unified pipeline)
# ---------------------------------------------------------------------------

def budgeted_submodular_greedy_reranker(
    candidates: List[int],                   # C: candidate item ids (catalogue indices)
    reranker_score_map: Dict[int, float],     # precomputed p(yes) per item
    utility: RerankerBackedSubmodular,        # f_θ (diversity params)
    slate_size: int,
    budget: float,
    costs: Optional[Dict[int, float]] = None,
    alpha_override: Optional[float] = None,
    kappa: float = 0.0,
) -> Tuple[List[int], float]:
    """
    Vectorized BudgetedSubmodularGreedy for the unified pipeline.

    Precomputes all pairwise kernel values once (one GPU op), then computes
    marginal gains for the entire feasible set in batch each greedy step —
    replacing O(slate_size × n_candidates) tensor allocations with O(slate_size)
    numpy array ops.

    Returns
    -------
    slate       : list of selected item ids (catalogue indices)
    final_score : f_θ(slate | q, x)
    """
    if not candidates:
        return [], 0.0

    n = len(candidates)
    device = utility.item_emb.weight.device
    alpha = float(alpha_override) if alpha_override is not None else utility.alpha.item()

    # ── One-time precomputation: embeddings + full pairwise kernel (n×n) ──
    with torch.no_grad():
        cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
        embs = F.normalize(utility.item_emb(cand_t), dim=-1)  # (n, d)
        sim = embs @ embs.T                                    # (n, n)
        if utility.kernel == "rbf":
            bw = torch.exp(utility.log_bandwidth).clamp(min=1e-3)
            k_mat = torch.exp(-(1.0 - sim) / bw)
        else:
            k_mat = (sim + 1.0) / 2.0
    k_np = k_mat.cpu().numpy()  # (n, n) — kernel values for all candidate pairs

    rel_arr  = np.array([reranker_score_map.get(c, 0.0) for c in candidates], dtype=np.float32)
    cost_arr = np.array([(costs or {}).get(c, 1.0) for c in candidates], dtype=np.float32)

    slate_local: List[int] = []   # local indices into candidates[]
    in_slate = np.zeros(n, dtype=bool)
    budget_rem = float(budget)
    kernel_sum = 0.0   # Σ_{i≠j ∈ slate} k(e_i, e_j) (ordered pairs)
    sum_rel    = 0.0   # Σ_{i ∈ slate} rel(i)
    k_cur      = 0     # current slate size

    eps = eps_from_kappa(kappa)
    tau = tau_from_kappa(kappa)

    for _ in range(slate_size):
        feasible = np.where(~in_slate & (cost_arr <= budget_rem))[0]
        if feasible.size == 0:
            break

        # ── Vectorized marginal gain for all feasible items ──────────────
        if k_cur == 0:
            # Single-item slate: diversity = 0, only relevance counts
            benefit = alpha * rel_arr[feasible]
        else:
            # f(S) — same for all candidates, compute once
            div_S = (1.0 - kernel_sum / (k_cur * (k_cur - 1))) if k_cur >= 2 else 0.0
            f_S   = alpha * (sum_rel / k_cur) + (1.0 - alpha) * div_S

            # For each feasible item i: Σ_{j∈slate} k(e_i, e_j)  → shape (|feasible|,)
            sum_k = k_np[np.ix_(feasible, slate_local)].sum(axis=1)

            new_k   = k_cur + 1
            rel_new = (sum_rel + rel_arr[feasible]) / new_k
            div_new = 1.0 - (kernel_sum + 2.0 * sum_k) / (new_k * k_cur)
            benefit = alpha * rel_new + (1.0 - alpha) * div_new - f_S

        ratios = benefit / np.maximum(cost_arr[feasible], 1e-8)

        # ── Selection (greedy or ε-exploration) ──────────────────────────
        if random.random() < eps:
            q = (ratios / tau).astype(np.float64)
            q -= q.max()
            probs = np.exp(q); probs /= probs.sum()
            local_idx = int(np.random.choice(len(feasible), p=probs))
        else:
            local_idx = int(np.argmax(ratios))

        chosen = int(feasible[local_idx])

        # Update incremental state
        if k_cur > 0:
            kernel_sum += 2.0 * k_np[chosen, slate_local].sum()
        sum_rel += rel_arr[chosen]
        slate_local.append(chosen)
        in_slate[chosen] = True
        budget_rem -= cost_arr[chosen]
        k_cur += 1

    slate_ids    = [candidates[i] for i in slate_local]
    slate_scores = [reranker_score_map.get(c, 0.0) for c in slate_ids]
    final_score  = utility.evaluate_with_scores(slate_ids, slate_scores, alpha_override)
    return slate_ids, final_score


# ---------------------------------------------------------------------------
# Batched wrapper for use inside joint training
# ---------------------------------------------------------------------------

def select_slates_batch(
    candidate_ids: torch.Tensor,           # (B, M) item indices
    utility: SubmodularUtility,
    contexts: torch.Tensor,                # (B, embed_dim)
    slate_size: int,
    budgets: List[float],                  # length B
    costs_map: Optional[Dict[int, float]] = None,  # global price map
    alphas: Optional[torch.Tensor] = None,          # (B,) or (B, 1)
    kappas: Optional[torch.Tensor] = None,          # (B,) or (B, 1)
) -> List[List[int]]:
    """
    Run BudgetedSubmodularGreedy for an entire batch (sequentially).

    Returns list of slates (list of item ids) for each batch element.
    """
    B = contexts.shape[0]
    slates = []
    for b in range(B):
        cands = candidate_ids[b].tolist()
        ctx = contexts[b]
        budget = float(budgets[b])
        alpha_b = float(alphas[b]) if alphas is not None else None
        kappa_b = float(kappas[b]) if kappas is not None else 0.0

        slate, _ = budgeted_submodular_greedy(
            candidates=cands,
            utility=utility,
            context=ctx,
            slate_size=slate_size,
            budget=budget,
            costs=costs_map,
            alpha_override=alpha_b,
            kappa=kappa_b,
        )
        slates.append(slate)
    return slates
