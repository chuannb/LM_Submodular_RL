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

from models.submodular import SubmodularUtility


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
