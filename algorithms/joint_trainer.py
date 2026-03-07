"""
Algorithm 3: Joint Training
  Generative Candidate  +  Submodular (Budget)  +  RL Knob Controller

Training loop:
  For each mini-batch (x_t, B_t, y_t):
    1. Construct state s_t from encoder
    2. Sample control knobs a_t = (z_t, α_t, κ_t) ~ π_ϕ(· | s_t)
    3. Generate candidates C_t = G_ψ(x_t, z_t)
    4. Select slate S_t via BudgetedSubmodularGreedy
    5. Compute proxy reward r_t
    6. Store transition (s_t, a_t, r_t) in replay buffer
    7. Update π_ϕ (actor-critic)
    8. Update G_ψ (contrastive loss)
    9. Update θ (submodular ranking loss)
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from algorithms.greedy_selector import select_slates_batch
from algorithms.trajectory_builder import (
    TrajectoryStep,
    proxy_reward_amazon,
    proxy_reward_retailrocket,
)
from models.generator import GeneratorModel
from models.rl_policy import RLPolicy, Z_DIM, ALPHA_DIM, KAPPA_DIM
from models.submodular import SubmodularUtility
from utils.encoders import StateEncoder, pad_history
from utils.metrics import SlateMetrics


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray         # (embed_dim,)
    action: np.ndarray        # (action_dim,)
    reward: float
    next_state: np.ndarray    # (embed_dim,)
    done: bool


class ReplayBuffer:
    def __init__(self, max_size: int = 10_000):
        self.buffer: Deque[Transition] = deque(maxlen=max_size)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Joint trainer
# ---------------------------------------------------------------------------

class JointTrainer:
    """
    Orchestrates training of all three components as per Algorithm 3.

    Parameters
    ----------
    encoder          : StateEncoder (shared item embeddings)
    generator        : GeneratorModel  G_ψ
    submodular       : SubmodularUtility  f_θ
    policy           : RLPolicy  π_ϕ
    slate_size       : k
    history_length   : L
    candidate_size   : M
    gamma            : discount factor
    batch_size       : mini-batch size
    buffer_size      : replay buffer capacity
    min_buffer_size  : min samples before training starts
    lr_gen           : learning rate for generator
    lr_sub           : learning rate for submodular
    device           : torch device
    dataset_type     : "amazon" | "retailrocket"
    costs_map        : {item_id: price}  (None = uniform costs)
    """

    def __init__(
        self,
        encoder: StateEncoder,
        generator: GeneratorModel,
        submodular: SubmodularUtility,
        policy: RLPolicy,
        slate_size: int = 10,
        history_length: int = 10,
        candidate_size: int = 50,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        min_buffer_size: int = 256,
        lr_gen: float = 1e-3,
        lr_sub: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        dataset_type: str = "amazon",
        costs_map: Optional[Dict[int, float]] = None,
    ):
        self.encoder = encoder.to(device)
        self.generator = generator.to(device)
        self.submodular = submodular.to(device)
        self.policy = policy.to(device)

        self.slate_size = slate_size
        self.history_length = history_length
        self.candidate_size = candidate_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.device = device
        self.dataset_type = dataset_type
        self.costs_map = costs_map

        self.replay = ReplayBuffer(max_size=buffer_size)
        self.metrics = SlateMetrics()

        self.gen_opt = torch.optim.Adam(
            list(generator.parameters()) + list(encoder.parameters()), lr=lr_gen
        )
        self.sub_opt = torch.optim.Adam(submodular.parameters(), lr=lr_sub)

        self._step_count = 0

    # ------------------------------------------------------------------
    def _encode_state(
        self,
        history_ids: List[List[int]],
        history_extras: Optional[List[List[float]]],
    ) -> torch.Tensor:
        """Encode a batch of histories -> (B, embed_dim) state tensor."""
        ids_t, ext_t = pad_history(
            history_ids, history_extras, self.history_length, self.device
        )
        with torch.no_grad():
            state = self.encoder(ids_t, ext_t)
        return state

    # ------------------------------------------------------------------
    def _compute_reward(
        self,
        slate: List[int],
        step: TrajectoryStep,
    ) -> float:
        if self.dataset_type == "amazon":
            stars = (
                float(step.history_extras[-1])
                if step.history_extras
                else None
            )
            return proxy_reward_amazon(slate, step.item_id, stars)
        else:
            return proxy_reward_retailrocket(slate, step.item_id, step.event)

    # ------------------------------------------------------------------
    def _action_to_tensor(self, action_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate knob tensors into a single action vector."""
        return torch.cat([
            action_dict["z"],
            action_dict["alpha"],
            action_dict["kappa"],
        ], dim=-1)

    # ------------------------------------------------------------------
    def run_step(
        self,
        steps: List[TrajectoryStep],
    ) -> Dict[str, float]:
        """
        Process a mini-batch of trajectory steps (one iteration of Algorithm 3).

        Returns dict of scalar losses for logging.
        """
        B = len(steps)

        # ---- 1. Construct states ----
        hist_ids = [s.history_ids for s in steps]
        hist_ext = [s.history_extras for s in steps]
        states = self._encode_state(hist_ids, hist_ext)   # (B, embed_dim)

        # ---- 2. Sample control knobs ----
        self.encoder.eval()
        with torch.no_grad():
            action_dict = self.policy.act(states, deterministic=False)
        self.encoder.train()

        z = action_dict["z"]           # (B, Z_DIM)
        alphas = action_dict["alpha"]  # (B, 1)
        kappas = action_dict["kappa"]  # (B, 1)

        # ---- 3. Generate candidates ----
        cand_ids, _ = self.generator.generate_candidates(
            states, z, candidate_size=self.candidate_size
        )  # (B, M)

        # ---- 4. Select slates ----
        slates = select_slates_batch(
            candidate_ids=cand_ids,
            utility=self.submodular,
            contexts=states,
            slate_size=self.slate_size,
            budgets=[s.budget for s in steps],
            costs_map=self.costs_map,
            alphas=alphas.squeeze(-1),
            kappas=kappas.squeeze(-1),
        )

        # ---- 5. Compute proxy rewards ----
        rewards = [
            self._compute_reward(slates[b], steps[b])
            for b in range(B)
        ]

        # ---- 6. Store in replay buffer ----
        actions_np = self._action_to_tensor(action_dict).detach().cpu().numpy()
        states_np = states.detach().cpu().numpy()

        # Construct dummy "next state" (shift by 1 within batch; last step reuses own state)
        next_states_np = np.roll(states_np, -1, axis=0)
        next_states_np[-1] = states_np[-1]

        for b in range(B):
            self.replay.push(Transition(
                state=states_np[b],
                action=actions_np[b],
                reward=rewards[b],
                next_state=next_states_np[b],
                done=False,
            ))
            self.metrics.update(slates[b], steps[b].item_id)

        self._step_count += 1

        # ---- 7-9. Update models ----
        losses: Dict[str, float] = {}
        if len(self.replay) >= self.min_buffer_size:
            losses.update(self._update_policy())
            losses.update(self._update_generator(states, z, steps))
            losses.update(self._update_submodular(states, steps))

        return losses

    # ------------------------------------------------------------------
    def _update_policy(self) -> Dict[str, float]:
        """Update π_ϕ from replay buffer (actor-critic step)."""
        batch = self.replay.sample(self.batch_size)
        states = torch.FloatTensor(np.stack([t.state for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.stack([t.action for t in batch])).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.stack([t.next_state for t in batch])).to(self.device)
        dones = torch.BoolTensor([t.done for t in batch]).to(self.device)

        return self.policy.update(states, actions, rewards, next_states, dones)

    # ------------------------------------------------------------------
    def _update_generator(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
        steps: List[TrajectoryStep],
    ) -> Dict[str, float]:
        """Update G_ψ via contrastive loss on (state, positive_item)."""
        pos_ids = torch.LongTensor([s.item_id for s in steps]).to(self.device)

        self.gen_opt.zero_grad()
        loss = self.generator.contrastive_loss(states, z, pos_ids)
        loss.backward()
        nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.gen_opt.step()

        return {"gen_loss": loss.item()}

    # ------------------------------------------------------------------
    def _update_submodular(
        self,
        states: torch.Tensor,
        steps: List[TrajectoryStep],
        num_neg: int = 10,
    ) -> Dict[str, float]:
        """Update θ (submodular parameters) via learning-to-rank signals."""
        B = len(steps)
        num_items = self.submodular.item_emb.num_embeddings
        pos_ids = torch.LongTensor([s.item_id for s in steps]).to(self.device)

        # Random in-batch negatives
        neg_ids = torch.randint(0, num_items, (B, num_neg), device=self.device)
        # Ensure negatives != positives
        for b in range(B):
            mask = neg_ids[b] == pos_ids[b]
            neg_ids[b][mask] = (pos_ids[b].item() + 1) % num_items

        self.sub_opt.zero_grad()
        loss = self.submodular.ranking_loss(states, pos_ids, neg_ids)
        loss.backward()
        nn.utils.clip_grad_norm_(self.submodular.parameters(), 1.0)
        self.sub_opt.step()

        return {"sub_loss": loss.item()}

    # ------------------------------------------------------------------
    def train_epoch(
        self,
        train_steps: List[TrajectoryStep],
        steps_per_epoch: int,
        log_every: int = 50,
    ) -> Dict[str, float]:
        """
        Run one epoch of Algorithm 3 over the training trajectories.
        """
        epoch_losses: Dict[str, List[float]] = {}
        indices = list(range(len(train_steps)))
        random.shuffle(indices)

        for i in range(0, min(steps_per_epoch * self.batch_size, len(indices)), self.batch_size):
            batch_idx = indices[i: i + self.batch_size]
            if not batch_idx:
                break
            batch = [train_steps[j] for j in batch_idx]
            losses = self.run_step(batch)

            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v)

            step_num = i // self.batch_size
            if log_every and step_num % log_every == 0 and losses:
                loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
                print(f"  step {step_num:4d} | {loss_str}")

            # Soft-update target critic
            self.policy.soft_update_target(tau=0.005)

        avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        return avg

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        eval_steps: List[TrajectoryStep],
    ) -> Dict[str, float]:
        """
        Evaluate on val/test trajectories (no gradient updates).
        """
        self.metrics.reset()
        for i in range(0, len(eval_steps), self.batch_size):
            batch = eval_steps[i: i + self.batch_size]
            if not batch:
                break

            hist_ids = [s.history_ids for s in batch]
            hist_ext = [s.history_extras for s in batch]
            states = self._encode_state(hist_ids, hist_ext)

            action_dict = self.policy.act(states, deterministic=True)
            z = action_dict["z"]
            alphas = action_dict["alpha"]
            kappas = action_dict["kappa"]

            cand_ids, _ = self.generator.generate_candidates(
                states, z, candidate_size=self.candidate_size
            )

            slates = select_slates_batch(
                candidate_ids=cand_ids,
                utility=self.submodular,
                contexts=states,
                slate_size=self.slate_size,
                budgets=[s.budget for s in batch],
                costs_map=self.costs_map,
                alphas=alphas.squeeze(-1),
                kappas=kappas.squeeze(-1),
            )

            for b, step in enumerate(batch):
                self.metrics.update(slates[b], step.item_id)

        num_items = self.submodular.item_emb.num_embeddings
        return self.metrics.compute(num_items)
