"""
RL Policy  π_ϕ  (Actor-Critic)

At each step t the policy receives state s_t and outputs control knobs:
  a_t = (z_t, α_t, κ_t)  ~  π_ϕ(· | s_t)

  z_t  : latent control for the generator (dim = latent_dim)
  α_t  : relevance–diversity trade-off for submodular (scalar)
  κ_t  : exploration / temperature control for greedy (scalar)

Architecture:
  Actor  : s_t  -> (z_t, α_t, κ_t)  (Gaussian policy)
  Critic : s_t  -> V(s_t)            (value function for advantage estimation)

Training:
  Conservative offline actor-critic (BCQ/CQL style):
  L_actor  = -E[Q(s,a)] + β * KL(π || μ)   (behaviour cloning regularisation)
  L_critic = (r + γ V(s') - V(s))^2

We simplify to vanilla actor-critic with a behaviour-cloning regularisation term.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Action space sizes
# ---------------------------------------------------------------------------
Z_DIM = 32       # latent_dim for generator knob
ALPHA_DIM = 1    # relevance weight knob
KAPPA_DIM = 1    # exploration knob


class Actor(nn.Module):
    """
    Gaussian actor: s_t -> mean and log_std for (z_t, α_t, κ_t).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        z_dim: int = Z_DIM,
        action_std_init: float = 0.5,
    ):
        super().__init__()
        self.z_dim = z_dim
        action_dim = z_dim + ALPHA_DIM + KAPPA_DIM

        layers = []
        in_dim = state_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), math.log(action_std_init))
        )

    def forward(
        self,
        state: torch.Tensor,   # (B, state_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(state)
        mean = self.mean_head(h)
        log_std = self.log_std.expand_as(mean).clamp(-4, 2)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action using reparameterisation trick.
        Returns:
          action    : (B, action_dim)  raw (unbounded)
          log_prob  : (B,)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def decode_action(
        self,
        action: torch.Tensor,  # (B, action_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Split raw action tensor into named knobs with appropriate activations.

        z     : raw (passed to generator MLP)
        alpha : sigmoid -> [0, 1]
        kappa : sigmoid -> [0, 1]  (0=greedy, 1=full exploration)
        """
        z = action[:, : self.z_dim]
        alpha_raw = action[:, self.z_dim: self.z_dim + ALPHA_DIM]
        kappa_raw = action[:, self.z_dim + ALPHA_DIM:]

        return {
            "z": z,
            "alpha": torch.sigmoid(alpha_raw),
            "kappa": torch.sigmoid(kappa_raw),
        }


class Critic(nn.Module):
    """
    State-value function V(s_t).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)  # (B,)


class RLPolicy(nn.Module):
    """
    Full RL policy: actor + critic (+ target critic for stable training).

    Parameters
    ----------
    state_dim    : dimension of s_t vector
    hidden_dim   : hidden layer size
    num_layers   : depth of actor / critic networks
    gamma        : discount factor
    lr           : learning rate
    bc_coeff     : behaviour-cloning regularisation weight
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        gamma: float = 0.99,
        lr: float = 3e-4,
        bc_coeff: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma
        self.bc_coeff = bc_coeff

        self.actor = Actor(state_dim, hidden_dim, num_layers)
        self.critic = Critic(state_dim, hidden_dim, num_layers)
        self.target_critic = Critic(state_dim, hidden_dim, num_layers)
        self._sync_target()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    # ------------------------------------------------------------------
    def _sync_target(self) -> None:
        """Hard-copy critic weights to target critic."""
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_target(self, tau: float = 0.005) -> None:
        for p, p_target in zip(self.critic.parameters(), self.target_critic.parameters()):
            p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,           # (B, state_dim)
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample knobs from the policy.
        Returns decoded dict: {"z": ..., "alpha": ..., "kappa": ...}
        """
        if deterministic:
            mean, _ = self.actor.forward(state)
            action = mean
        else:
            action, _ = self.actor.sample(state)
        return self.actor.decode_action(action)

    # ------------------------------------------------------------------
    def update(
        self,
        states: torch.Tensor,        # (B, state_dim)
        actions: torch.Tensor,       # (B, action_dim)  raw, stored in replay buffer
        rewards: torch.Tensor,       # (B,)
        next_states: torch.Tensor,   # (B, state_dim)
        dones: torch.Tensor,         # (B,) bool
    ) -> Dict[str, float]:
        """
        One gradient step for actor and critic.
        Returns loss dict for logging.
        """
        # ---- Critic update ----
        with torch.no_grad():
            v_next = self.target_critic(next_states)       # (B,)
            targets = rewards + self.gamma * v_next * (~dones).float()

        v_pred = self.critic(states)
        critic_loss = F.mse_loss(v_pred, targets)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ---- Actor update ----
        new_actions, log_probs = self.actor.sample(states)
        advantages = (targets - v_pred.detach())           # (B,)

        # Policy gradient loss
        pg_loss = -(log_probs * advantages).mean()

        # Behaviour-cloning regularisation (conservative offline RL)
        bc_loss = F.mse_loss(new_actions, actions)

        actor_loss = pg_loss + self.bc_coeff * bc_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "pg_loss": pg_loss.item(),
            "bc_loss": bc_loss.item(),
        }
