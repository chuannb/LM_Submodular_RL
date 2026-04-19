"""
UnifiedJointTrainer
===================

Loss function thống nhất tối ưu đồng thời 3 component:

  L_total = L_rl  +  λ_sub * L_sub  +  λ_reranker * L_reranker

  L_rl:
    L_critic  = MSE(V(s_t), r_t + γ V'(s_{t+1}))
    L_actor   = -E[log π(a|s) * A(s,a)] + β * BC_loss     (offline AC)

  L_sub  (gradients qua diversity params θ):
    L_reinforce = -Σ_t r_t * log f_θ(S_t | q_t)           (REINFORCE)
    L_div_rank  = diversity_ranking_loss(θ)                 (anti-collapse)
    L_sub = L_reinforce + λ_rank * L_div_rank

  L_reranker  (offline, từ interaction logs):
    L_pairwise  = -Σ log σ(score(q,pos) - score(q,neg))    (margin loss)
    → hoặc thay bằng DPO/ORPO từ dpo_trainer.py

Gradient flow summary:
  ┌───────────────────────────────────────────────────────────┐
  │ Component          │ Gradient source          │ Freq      │
  ├───────────────────────────────────────────────────────────┤
  │ π_ϕ (RL policy)    │ actor-critic + BC        │ online    │
  │ θ (diversity emb)  │ REINFORCE + div ranking  │ online    │
  │ Reranker weights   │ pairwise / DPO / ORPO    │ offline   │
  │ StateEncoder       │ shared with RL            │ online    │
  └───────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.submodular import RerankerBackedSubmodular
from retrieval.unified_pipeline import UnifiedPipeline, UnifiedRLPolicy
from utils.encoders import encode_history
from utils.metrics import SlateMetrics


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class UnifiedTransition:
    state: np.ndarray           # (d,)
    action: np.ndarray          # (2,) = raw (alpha_raw, kappa_raw)
    slate: List[int]            # selected item indices
    slate_rel_scores: List[float]
    all_cand_idx: List[int]     # all candidate indices (for soft submodular)
    all_cand_scores: List[float]
    target_idx: int             # ground-truth next item
    reward: float
    next_state: np.ndarray      # (d,)
    done: bool = False


class UnifiedReplayBuffer:
    def __init__(self, max_size: int = 20_000):
        self.buf: Deque[UnifiedTransition] = deque(maxlen=max_size)

    def push(self, t: UnifiedTransition) -> None:
        self.buf.append(t)

    def sample(self, n: int) -> List[UnifiedTransition]:
        return random.sample(self.buf, min(n, len(self.buf)))

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# Unified trainer
# ---------------------------------------------------------------------------

class UnifiedJointTrainer:
    """
    Orchestrates joint training của tất cả online components.

    Offline reranker fine-tuning được thực hiện riêng qua:
      RerankerTrainer  (training/reranker_trainer.py)
      DPOFinetuner     (training/dpo_trainer.py)

    Parameters
    ----------
    pipeline        : UnifiedPipeline instance
    rl_policy       : UnifiedRLPolicy
    submodular      : RerankerBackedSubmodular (θ)
    state_encoder   : StateEncoder
    lambda_sub      : weight for L_sub in total loss
    lambda_rank     : weight for L_div_rank inside L_sub
    lr_sub          : learning rate for submodular parameters
    lr_encoder      : learning rate for state encoder
    batch_size      : replay buffer batch size
    buffer_size     : max replay buffer capacity
    min_buffer      : minimum samples before training begins
    gamma           : discount factor
    device          : torch device
    """

    def __init__(
        self,
        pipeline: UnifiedPipeline,
        rl_policy: UnifiedRLPolicy,
        submodular: RerankerBackedSubmodular,
        state_encoder,
        lambda_sub: float = 0.5,
        lambda_rank: float = 0.1,
        lr_sub: float = 1e-3,
        lr_encoder: float = 1e-3,
        batch_size: int = 32,
        buffer_size: int = 20_000,
        min_buffer: int = 128,
        gamma: float = 0.99,
        history_length: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        self.pipeline = pipeline
        self.rl = rl_policy
        self.sub = submodular
        self.encoder = state_encoder
        self.lambda_sub = lambda_sub
        self.lambda_rank = lambda_rank
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.gamma = gamma
        self.history_length = history_length
        self.device = device

        self.replay = UnifiedReplayBuffer(max_size=buffer_size)
        self.metrics = SlateMetrics()

        self.sub_opt = torch.optim.Adam(
            list(submodular.parameters()) + list(state_encoder.parameters()),
            lr=lr_sub,
        )

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def step(self, transitions: List[UnifiedTransition]) -> Dict[str, float]:
        """
        Process a list of fresh transitions:
          1. Push to replay buffer
          2. If buffer large enough: update RL + submodular
        Returns loss dict.
        """
        for t in transitions:
            self.replay.push(t)
            self.metrics.update(t.slate, t.target_idx)

        if len(self.replay) < self.min_buffer:
            return {}

        batch = self.replay.sample(self.batch_size)
        losses = {}
        losses.update(self._update_rl(batch))
        losses.update(self._update_submodular(batch))
        self.rl.soft_update_target()
        return losses

    # ------------------------------------------------------------------
    # RL update  (L_rl)
    # ------------------------------------------------------------------

    def _update_rl(self, batch: List[UnifiedTransition]) -> Dict[str, float]:
        states = torch.FloatTensor(np.stack([t.state for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.stack([t.action for t in batch])).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.stack([t.next_state for t in batch])).to(self.device)
        dones = torch.BoolTensor([t.done for t in batch]).to(self.device)

        return self.rl.update(states, actions, rewards, next_states, dones)

    # ------------------------------------------------------------------
    # Submodular update  (L_sub = L_reinforce + λ_rank * L_div_rank)
    # ------------------------------------------------------------------

    def _update_submodular(self, batch: List[UnifiedTransition]) -> Dict[str, float]:
        """
        L_reinforce:
          For each transition t:
            log_score = log f_θ(S_t) = log(soft_slate_score)
            L_rf = -Σ r_t * log_score_t

          f_θ(S_t) computed differentiably via RerankerBackedSubmodular.soft_slate_score()
          Reranker scores are detached (no grad through reranker).

        L_div_rank:
          Uses positive (target items) and random negatives.
        """
        B = len(batch)
        K = self.pipeline.slate_size

        # ---- Prepare slate tensors ----
        slate_ids_list = []
        rel_scores_list = []
        alphas_list = []
        rewards = []

        for t in batch:
            slate = t.slate[:K]
            rel_s = t.slate_rel_scores[:K]
            # Pad if shorter than K
            while len(slate) < K:
                slate.append(slate[0] if slate else 0)
                rel_s.append(0.0)
            slate_ids_list.append(slate)
            rel_scores_list.append(rel_s)
            alphas_list.append(t.action[0])   # raw alpha (before sigmoid)
            rewards.append(t.reward)

        slate_ids = torch.LongTensor(slate_ids_list).to(self.device)     # (B, K)
        rel_scores = torch.FloatTensor(rel_scores_list).to(self.device)  # (B, K)
        alphas = torch.FloatTensor(alphas_list).to(self.device)          # (B,)
        alphas_sig = torch.sigmoid(alphas)                               # (B,) in [0,1]
        reward_t = torch.FloatTensor(rewards).to(self.device)            # (B,)

        self.sub_opt.zero_grad()

        # ---- L_reinforce ----
        soft_scores = self.sub.soft_slate_score(
            item_ids=slate_ids,
            reranker_scores=rel_scores,
            alpha=alphas_sig,
        )   # (B,) differentiable through diversity embeddings

        # log(f_θ) — add small eps for numerical stability
        log_scores = torch.log(soft_scores.clamp(min=1e-6))
        l_reinforce = -(reward_t * log_scores).mean()

        # ---- L_div_rank ----
        pos_ids = torch.LongTensor([t.target_idx for t in batch]).to(self.device)  # (B,)
        num_items = self.sub.item_emb.num_embeddings
        num_neg = min(10, num_items - 1)
        neg_ids = torch.randint(0, num_items, (B, num_neg), device=self.device)
        # Avoid same as pos
        for b in range(B):
            mask = neg_ids[b] == pos_ids[b]
            neg_ids[b][mask] = (pos_ids[b].item() + 1) % num_items

        l_div_rank = self.sub.diversity_ranking_loss(pos_ids, neg_ids)

        # ---- Total submodular loss ----
        l_sub = l_reinforce + self.lambda_rank * l_div_rank
        l_sub.backward()
        nn.utils.clip_grad_norm_(self.sub.parameters(), 1.0)
        self.sub_opt.step()

        return {
            "sub/reinforce_loss": l_reinforce.item(),
            "sub/div_rank_loss": l_div_rank.item(),
            "sub/total": l_sub.item(),
            "sub/alpha_mean": alphas_sig.mean().item(),
        }

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        trajectory_steps: list,        # List[TrajectoryStep] từ trajectory_builder
        pipeline_query_fn,             # fn(step) -> query string
        steps_per_epoch: int = 300,
        log_every: int = 50,
        dataset_type: str = "amazon",
        fast_mode: bool = True,        # skip reranker during training (use BM25 scores)
    ) -> Dict[str, float]:
        """
        Run one training epoch:
          For each step in trajectory:
            1. collect_transition từ pipeline
            2. compute reward
            3. push to buffer + update
        """
        from algorithms.trajectory_builder import (
            proxy_reward_amazon,
            proxy_reward_retailrocket,
        )

        epoch_losses: Dict[str, List[float]] = {}
        random.shuffle(trajectory_steps)

        for i, step in enumerate(trajectory_steps[:steps_per_epoch]):
            query = pipeline_query_fn(step)

            trans_dict = self.pipeline.collect_transition(
                query=query,
                history_ids=step.history_ids,
                history_extras=step.history_extras,
                budget=step.budget,
                target_item_idx=step.item_id,
                reward=0.0,   # placeholder
                dataset_type=dataset_type,
                fast_mode=fast_mode,
            )
            if trans_dict is None:
                continue

            # Shaped reward với candidate info (fix sparse reward)
            if dataset_type == "amazon":
                stars = step.history_extras[-1] if step.history_extras else None
                reward = proxy_reward_amazon(
                    slate=trans_dict["slate"],
                    target=step.item_id,
                    stars=stars,
                    cand_idx=trans_dict["slate_cands_idx"],
                    cand_scores=trans_dict["slate_cands_scores"],
                )
            else:
                stars = None
                reward = proxy_reward_retailrocket(
                    trans_dict["slate"], step.item_id, step.event
                )
            trans_dict["reward"] = reward

            # Fix next_state: encode history thực của bước tiếp theo
            # s_{t+1} = encode(history + [target_item]) — phản ánh đúng state sau interaction
            next_hist_ids = (step.history_ids + [step.item_id])[-self.history_length:]
            next_hist_ext = ((step.history_extras or []) + [stars if stars is not None else 5.0])[-self.history_length:]
            next_state_t = encode_history(
                [next_hist_ids], [next_hist_ext],
                self.encoder, self.history_length, self.device,
            )
            next_state = next_state_t.cpu().numpy()[0]

            t = UnifiedTransition(
                state=trans_dict["state"],
                action=trans_dict["action"],
                slate=trans_dict["slate"],
                slate_rel_scores=trans_dict["rel_scores"],
                all_cand_idx=trans_dict["slate_cands_idx"],
                all_cand_scores=trans_dict["slate_cands_scores"],
                target_idx=trans_dict["target"],
                reward=reward,
                next_state=next_state,
                done=False,
            )

            losses = self.step([t])
            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v)

            if log_every and i % log_every == 0 and losses:
                loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
                print(f"  step {i:4d} | {loss_str}")

        return {k: float(np.mean(v)) for k, v in epoch_losses.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        eval_steps: list,
        pipeline_query_fn,
        dataset_type: str = "amazon",
    ) -> Dict[str, float]:
        """Evaluate hit@k, NDCG@k, coverage on eval split."""
        from algorithms.trajectory_builder import (
            proxy_reward_amazon,
            proxy_reward_retailrocket,
        )

        self.metrics.reset()
        for step in eval_steps:
            query = pipeline_query_fn(step)
            results, _ = self.pipeline.search(
                query=query,
                history_ids=step.history_ids,
                history_extras=step.history_extras,
                budget=step.budget,
                deterministic=True,
            )
            slate = [r.item_idx for r in results]
            self.metrics.update(slate, step.item_id)

        num_items = self.sub.item_emb.num_embeddings
        return self.metrics.compute(num_items)
