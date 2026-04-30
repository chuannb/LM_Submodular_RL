"""
UnifiedPipeline
===============

Kết hợp toàn bộ các thành phần thành một pipeline nhất quán:

  Query + User History
       │
       ├─ BM25 recall ─────────┐
       │                        ├─ RRF Fusion → Candidates C (M items)
       └─ Dense recall ─────────┘
                │
        Qwen3-Reranker → {rel_score(q, i)}  (precomputed cho tất cả i ∈ C)
                │
     RL Policy π_ϕ(s_t) → (α_t, κ_t)
         s_t = StateEncoder(history)
                │
     RerankerBackedSubmodular f_θ:
       f_θ(S|q,x) = α_t * mean_rel(S)  +  (1-α_t) * div_θ(S)
                │
     BudgetedSubmodularGreedy → Slate S_t (k items)
                │
       Log impression → user signals
                │ (offline job)
     pairwise preferences → fine-tune Reranker (DPO/ORPO / pairwise)
                +
     online RL + submodular update

Online gradient flow:
  - π_ϕ: actor-critic từ replay buffer
  - θ (diversity params): REINFORCE + diversity ranking loss
  Reranker: updated offline (detached từ online loop)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from algorithms.greedy_selector import budgeted_submodular_greedy_reranker
from interaction.logger import InteractionLogger
from models.rl_policy import RLPolicy, ALPHA_DIM, KAPPA_DIM
from models.submodular import RerankerBackedSubmodular
from retrieval.bm25_retriever import BM25Retriever
from retrieval.embedding_retriever import EmbeddingRetriever
from retrieval.reranker import Qwen3Reranker, RankedResult
from utils.encoders import StateEncoder, pad_history


# ---------------------------------------------------------------------------
# Candidate container (internal)
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    item_id: str          # original string id (from product DB)
    item_idx: int         # integer index used by models
    rel_score: float      # Qwen3-Reranker p(yes)
    title: str
    text: str


@dataclass
class UnifiedSearchResult:
    item_id: str
    item_idx: int
    rel_score: float       # reranker score
    submodular_score: float  # f_θ contribution
    title: str
    slate_position: int


# ---------------------------------------------------------------------------
# RL action space (no z_t in unified pipeline; retrieval handles candidate gen)
# ---------------------------------------------------------------------------
UNIFIED_ACTION_DIM = ALPHA_DIM + KAPPA_DIM   # 2


class UnifiedRLPolicy(torch.nn.Module):
    """
    Simplified RL policy for unified pipeline.
    Action: a_t = (α_t, κ_t)  — no generator knob needed.

    Wraps RLPolicy but projects to UNIFIED_ACTION_DIM.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, lr: float = 3e-4,
                 gamma: float = 0.99):
        super().__init__()
        # Reuse RLPolicy infrastructure but with smaller action space
        from models.rl_policy import Actor, Critic
        self.actor = Actor(state_dim, hidden_dim, num_layers=2, z_dim=0)
        # Override actor output to 2 dims only
        self.actor.mean_head = torch.nn.Linear(hidden_dim, UNIFIED_ACTION_DIM)
        import math
        self.actor.log_std = torch.nn.Parameter(
            torch.full((UNIFIED_ACTION_DIM,), math.log(0.5))
        )
        self.critic = Critic(state_dim, hidden_dim)
        self.target_critic = Critic(state_dim, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.gamma = gamma
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        if deterministic:
            mean, _ = self.actor.forward(state)
            action = mean
        else:
            action, _ = self.actor.sample(state)
        alpha = torch.sigmoid(action[:, 0:1])    # (B, 1)
        kappa = torch.sigmoid(action[:, 1:2])    # (B, 1)
        return {"alpha": alpha, "kappa": kappa, "raw": action}

    def soft_update_target(self, tau: float = 0.005) -> None:
        for p, pt in zip(self.critic.parameters(), self.target_critic.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

    def update(self, states, actions, rewards, next_states, dones, bc_coeff=0.1):
        import torch.nn.functional as F
        with torch.no_grad():
            v_next = self.target_critic(next_states)
            targets = rewards + self.gamma * v_next * (~dones).float()

        v_pred = self.critic(states)
        critic_loss = F.mse_loss(v_pred, targets)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        new_actions, log_probs = self.actor.sample(states)
        advantages = (targets - v_pred.detach())
        pg_loss = -(log_probs * advantages).mean()
        bc_loss = F.mse_loss(new_actions, actions)
        actor_loss = pg_loss + bc_coeff * bc_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return {
            "rl/critic_loss": critic_loss.item(),
            "rl/actor_loss": actor_loss.item(),
            "rl/pg_loss": pg_loss.item(),
        }


# ---------------------------------------------------------------------------
# Main UnifiedPipeline
# ---------------------------------------------------------------------------

class UnifiedPipeline:
    """
    Full unified pipeline:
      BM25 + Dense → Reranker → Submodular(RL) → Slate

    Parameters
    ----------
    bm25            : BM25Retriever
    reranker        : Qwen3Reranker
    submodular      : RerankerBackedSubmodular (learnable θ)
    rl_policy       : UnifiedRLPolicy
    state_encoder   : StateEncoder (encodes user history → s_t)
    id_map          : {item_id_str: item_idx_int} — maps string ids to model indices
    logger          : InteractionLogger (optional)
    dense           : EmbeddingRetriever (optional)
    device          : torch device
    n_bm25          : BM25 recall size
    n_dense         : dense recall size
    n_fuse          : candidates after RRF, scored by reranker
    slate_size      : final slate size k
    history_length  : L for StateEncoder
    costs_map       : {item_idx: price} for budget constraint
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        reranker: Qwen3Reranker,
        submodular: RerankerBackedSubmodular,
        rl_policy: UnifiedRLPolicy,
        state_encoder: StateEncoder,
        id_map: Dict[str, int],          # str_id → int_idx
        logger: Optional[InteractionLogger] = None,
        dense: Optional[EmbeddingRetriever] = None,
        device: torch.device = torch.device("cpu"),
        n_bm25: int = 200,
        n_dense: int = 100,
        n_fuse: int = 200,
        slate_size: int = 10,
        history_length: int = 10,
        costs_map: Optional[Dict[int, float]] = None,
    ):
        self.bm25 = bm25
        self.reranker = reranker
        self.submodular = submodular
        self.rl_policy = rl_policy
        self.state_encoder = state_encoder
        self.id_map = id_map
        self.id_map_inv = {v: k for k, v in id_map.items()}
        self.logger = logger
        self.dense = dense
        self.device = device
        self.n_bm25 = n_bm25
        self.n_dense = n_dense
        self.n_fuse = n_fuse
        self.slate_size = slate_size
        self.history_length = history_length
        self.costs_map = costs_map or {}

    # ------------------------------------------------------------------
    @staticmethod
    def _rrf(
        *lists: List[Tuple[str, float, str, str]],
        k: int = 60,
    ) -> List[Tuple[str, float, str, str]]:
        scores: Dict[str, float] = {}
        meta: Dict[str, Tuple[str, str]] = {}
        for lst in lists:
            for rank, (iid, _, title, text) in enumerate(lst, start=1):
                scores[iid] = scores.get(iid, 0.0) + 1.0 / (k + rank)
                meta[iid] = (title, text)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(iid, sc, *meta[iid]) for iid, sc in ranked]

    # ------------------------------------------------------------------
    def _recall(self, query: str) -> List[Tuple[str, float, str, str]]:
        """Stage 1: BM25 + Dense → RRF → top n_fuse candidates."""
        bm25_res = self.bm25.search(query, top_k=self.n_bm25)
        bm25_tuples = [(r.item_id, r.score, r.title, r.text) for r in bm25_res]

        if self.dense is not None:
            dense_res = self.dense.search(query, top_k=self.n_dense)
            fused = self._rrf(bm25_tuples, dense_res)[: self.n_fuse]
        else:
            fused = bm25_tuples[: self.n_fuse]
        return fused

    # ------------------------------------------------------------------
    def _rerank_all(
        self, query: str, candidates: List[Tuple[str, float, str, str]]
    ) -> List[ScoredCandidate]:
        """
        Stage 2: Score ALL candidates with Qwen3-Reranker.
        Returns ScoredCandidate list (includes items not in final slate).
        These scores feed directly into the submodular utility as relevance.
        """
        if not candidates:
            return []

        texts = [c[3] for c in candidates]
        rel_scores = self.reranker.score(query, texts)

        scored = []
        for (item_id, _, title, text), rel in zip(candidates, rel_scores):
            item_idx = self.id_map.get(item_id, -1)
            if item_idx < 0:
                continue
            scored.append(ScoredCandidate(
                item_id=item_id,
                item_idx=item_idx,
                rel_score=rel,
                title=title,
                text=text,
            ))

        # Sort by relevance descending
        scored.sort(key=lambda x: x.rel_score, reverse=True)
        return scored

    # ------------------------------------------------------------------
    def _encode_state(
        self,
        history_ids: List[int],
        history_extras: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Encode user history → state vector (1, embed_dim)."""
        ids_t, ext_t = pad_history(
            [history_ids],
            [history_extras] if history_extras else None,
            self.history_length,
            self.device,
        )
        with torch.no_grad():
            state = self.state_encoder(ids_t, ext_t)   # (1, embed_dim)
        return state

    def encode_state(
        self,
        history_ids: List[int],
        history_extras: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Public wrapper for state encoding — returns numpy array (d,)."""
        state = self._encode_state(history_ids, history_extras)
        return state.detach().cpu().numpy()[0]

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        history_ids: Optional[List[int]] = None,
        history_extras: Optional[List[float]] = None,
        session_id: Optional[str] = None,
        budget: Optional[float] = None,
        page: int = 1,
        deterministic: bool = True,
    ) -> Tuple[List[UnifiedSearchResult], Optional[str]]:
        """
        Full pipeline: recall → rerank all → RL → submodular select → log.

        Parameters
        ----------
        query          : search query string
        history_ids    : list of item indices (int) in user history
        history_extras : list of signal weights (e.g. event weights / stars)
        session_id     : for interaction logging
        budget         : cost budget B_t (None = use slate_size as budget)
        page           : page number (for logging)
        deterministic  : if True, RL policy uses mean action

        Returns
        -------
        results        : List[UnifiedSearchResult] — final slate
        impression_id  : logged impression id
        """
        history_ids = history_ids or []
        budget = budget or float(self.slate_size)

        # ---- Stage 1: Recall ----
        candidates_raw = self._recall(query)

        # ---- Stage 2: Rerank all candidates ----
        scored_candidates = self._rerank_all(query, candidates_raw)
        if not scored_candidates:
            return [], None

        # ---- Stage 3: Encode user state ----
        state = self._encode_state(history_ids, history_extras)   # (1, d)

        # ---- Stage 4: RL policy → (α_t, κ_t) ----
        knobs = self.rl_policy.act(state, deterministic=deterministic)
        alpha_t = float(knobs["alpha"].item())
        kappa_t = float(knobs["kappa"].item())

        # ---- Stage 5: Build candidate maps for greedy ----
        cand_idx_list = [c.item_idx for c in scored_candidates]
        rel_score_map: Dict[int, float] = {
            c.item_idx: c.rel_score for c in scored_candidates
        }
        cost_map: Dict[int, float] = {
            c.item_idx: self.costs_map.get(c.item_idx, 1.0)
            for c in scored_candidates
        }

        # ---- Stage 6: BudgetedSubmodularGreedy ----
        slate_idx, final_sub_score = budgeted_submodular_greedy_reranker(
            candidates=cand_idx_list,
            reranker_score_map=rel_score_map,
            utility=self.submodular,
            slate_size=self.slate_size,
            budget=budget,
            costs=cost_map,
            alpha_override=alpha_t,
            kappa=kappa_t,
        )

        # Build result objects
        idx_to_cand = {c.item_idx: c for c in scored_candidates}
        results = [
            UnifiedSearchResult(
                item_id=idx_to_cand[idx].item_id,
                item_idx=idx,
                rel_score=rel_score_map[idx],
                submodular_score=final_sub_score,
                title=idx_to_cand[idx].title,
                slate_position=pos,
            )
            for pos, idx in enumerate(slate_idx)
            if idx in idx_to_cand
        ]

        # ---- Stage 7: Log impression ----
        impression_id = None
        if self.logger is not None and session_id is not None:
            impression_id = self.logger.log_impression(
                session_id=session_id,
                query=query,
                shown_items=[r.item_id for r in results],
                scores=[r.rel_score for r in results],
                page=page,
            )

        return results, impression_id

    # ------------------------------------------------------------------
    def _recall_with_bm25_scores(self, query: str) -> List["ScoredCandidate"]:
        """
        Fast alternative to _rerank_all: uses normalised BM25 scores as relevance
        proxy instead of calling the Qwen3-Reranker.  Used during training to
        avoid expensive LLM inference.
        """
        bm25_res = self.bm25.search(query, top_k=self.n_fuse)
        if not bm25_res:
            return []
        max_score = max(r.score for r in bm25_res) or 1.0
        scored = []
        for r in bm25_res:
            item_idx = self.id_map.get(r.item_id, -1)
            if item_idx < 0:
                continue
            scored.append(ScoredCandidate(
                item_id=r.item_id,
                item_idx=item_idx,
                rel_score=float(r.score) / max_score,   # normalised to [0,1]
                title=r.title,
                text=r.text,
            ))
        scored.sort(key=lambda x: x.rel_score, reverse=True)
        return scored

    def collect_transition(
        self,
        query: str,
        history_ids: List[int],
        history_extras: Optional[List[float]],
        budget: float,
        target_item_idx: int,
        reward: float,
        dataset_type: str = "amazon",
        stars: float = None,
        fast_mode: bool = False,
        inject_target: bool = True,
    ) -> Optional[dict]:
        """
        Run one step of the pipeline and return a transition dict
        for the unified trainer replay buffer.

        fast_mode=True: skip Qwen3-Reranker, use BM25 scores as relevance proxy.

        inject_target=True: if the ground-truth item is absent from retrieved
        candidates, insert it with median rel_score so the greedy can select it
        and reward > 0 is possible.  Standard practice in offline RL for rec —
        without this, reward is always 0 with large catalogs + small top-K recall.
        Does NOT affect evaluation (inject_target is False there).

        Returns None if no candidates found.
        """
        history_ids = history_ids or []
        if fast_mode:
            scored_candidates = self._recall_with_bm25_scores(query)
        else:
            candidates_raw = self._recall(query)
            scored_candidates = self._rerank_all(query, candidates_raw)
        if not scored_candidates:
            return None

        # ── Inject target if missing (training only) ──────────────────────
        if inject_target and target_item_idx >= 0:
            if not any(c.item_idx == target_item_idx for c in scored_candidates):
                target_str_id = self.id_map_inv.get(target_item_idx)
                if target_str_id is not None:
                    # Inject above the current max so greedy always selects target,
                    # guaranteeing a non-zero reward signal during early training.
                    max_score = max((c.rel_score for c in scored_candidates), default=1.0)
                    scored_candidates.append(ScoredCandidate(
                        item_id=target_str_id,
                        item_idx=target_item_idx,
                        rel_score=max_score + 0.01,
                        title="",
                        text="",
                    ))

        state = self._encode_state(history_ids, history_extras)

        knobs = self.rl_policy.act(state, deterministic=False)
        alpha_t = float(knobs["alpha"].item())
        kappa_t = float(knobs["kappa"].item())
        raw_action = knobs["raw"].detach().cpu().numpy()[0]

        cand_idx_list = [c.item_idx for c in scored_candidates]
        rel_score_map = {c.item_idx: c.rel_score for c in scored_candidates}
        cost_map = {c.item_idx: self.costs_map.get(c.item_idx, 1.0) for c in scored_candidates}

        slate_idx, _ = budgeted_submodular_greedy_reranker(
            candidates=cand_idx_list,
            reranker_score_map=rel_score_map,
            utility=self.submodular,
            slate_size=self.slate_size,
            budget=budget,
            costs=cost_map,
            alpha_override=alpha_t,
            kappa=kappa_t,
        )

        return {
            "state": state.detach().cpu().numpy()[0],          # (d,)
            "action": raw_action,                               # (2,)
            "slate": slate_idx,
            "rel_scores": [rel_score_map[i] for i in slate_idx],
            "slate_cands_idx": cand_idx_list,
            "slate_cands_scores": list(rel_score_map.values()),
            "target": target_item_idx,
            "reward": reward,
            "alpha": alpha_t,
            "kappa": kappa_t,
        }
