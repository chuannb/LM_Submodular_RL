"""
Global configuration for the Recommendation System.
Combines Generative Candidate Generation + Submodular Optimization + RL.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    dataset: str = "amazon"          # "amazon" | "retailrocket"
    history_length: int = 10         # L: number of past interactions to encode
    session_gap_seconds: int = 1800  # RetailRocket: max gap (seconds) within a session
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # Amazon paths
    amazon_review_path: Optional[str] = None
    amazon_meta_path: Optional[str] = None
    # RetailRocket paths
    rr_events_path: Optional[str] = None
    rr_category_tree_path: Optional[str] = None
    rr_item_props_path: Optional[str] = None
    rr_item_props_path2: Optional[str] = None


@dataclass
class ModelConfig:
    # Shared embedding dim
    embed_dim: int = 128
    hidden_dim: int = 256

    # Generator (Gψ)
    candidate_size: int = 50          # M: number of candidates generated per step
    generator_layers: int = 2
    generator_dropout: float = 0.1

    # Submodular utility (fθ)
    relevance_weight: float = 0.7     # default trade-off relevance vs diversity
    diversity_weight: float = 0.3
    submodular_kernel: str = "rbf"    # "rbf" | "dot"

    # RL policy (πϕ)
    slate_size: int = 10              # k: final slate size
    rl_hidden_dim: int = 256
    rl_layers: int = 2
    gamma: float = 0.99               # discount factor
    lr_policy: float = 3e-4
    lr_generator: float = 1e-3
    lr_submodular: float = 1e-3

    # Replay buffer
    buffer_size: int = 10_000
    batch_size: int = 64
    min_buffer_size: int = 256

    # Training
    num_epochs: int = 20
    steps_per_epoch: int = 500
    target_update_freq: int = 10      # hard update target network every N steps
    exploration_eps: float = 0.1      # ε for exploration in greedy


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    device: str = "cpu"               # "cpu" | "cuda"
    log_every: int = 50
