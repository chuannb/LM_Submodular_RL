"""
Feature encoders for state representation.

EncodeHistory  : encode user + item history -> context vector x_t
EncodeEvents   : encode RetailRocket event sequence -> context vector x_t

State s_t = (x_t, h_t) where:
  x_t = EncodeHistory / EncodeEvents output
  h_t = summary of last L interactions (mean-pooled embeddings)
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Simple learnable embedding-based encoder
# ---------------------------------------------------------------------------

class ItemEncoder(nn.Module):
    """
    Maps item ids -> dense embeddings, then produces a context vector via GRU.

    Input:
      item_ids  : (B, L)  integer item indices
      extra     : (B, L, extra_dim)  optional extra features (stars, event weight)

    Output:
      context   : (B, embed_dim)
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        extra_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim + extra_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        item_ids: torch.Tensor,          # (B, L)
        extra: Optional[torch.Tensor] = None,  # (B, L, extra_dim)
    ) -> torch.Tensor:                    # (B, embed_dim)
        x = self.item_emb(item_ids)      # (B, L, embed_dim)
        if extra is not None:
            x = torch.cat([x, extra], dim=-1)
        else:
            x = torch.cat([x, torch.zeros(*x.shape[:-1], 1, device=x.device)], dim=-1)
        _, h = self.gru(x)               # h: (1, B, hidden_dim)
        h = h.squeeze(0)                 # (B, hidden_dim)
        return self.proj(self.dropout(h))  # (B, embed_dim)


class StateEncoder(nn.Module):
    """
    Full state encoder: produces s_t = (x_t, h_t) as a single vector.

    x_t = ItemEncoder output (GRU last hidden state)
    h_t = mean of last L item embeddings (lightweight summary)

    Concatenated and projected to embed_dim.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        extra_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_encoder = ItemEncoder(num_items, embed_dim, hidden_dim, extra_dim, dropout)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        item_ids: torch.Tensor,
        extra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_t = self.item_encoder(item_ids, extra)          # (B, embed_dim)
        # h_t: mean-pool raw embeddings
        raw_embs = self.item_encoder.item_emb(item_ids)   # (B, L, embed_dim)
        h_t = raw_embs.mean(dim=1)                        # (B, embed_dim)
        state = self.proj(torch.cat([x_t, h_t], dim=-1))  # (B, embed_dim)
        return state


# ---------------------------------------------------------------------------
# Utility: pad / collate history lists to tensors
# ---------------------------------------------------------------------------

def pad_history(
    history_ids: List[List[int]],
    history_extras: Optional[List[List[float]]],
    max_len: int,
    device: torch.device,
) -> tuple:
    """
    Pad a batch of variable-length histories.

    Returns:
      ids_tensor   : (B, max_len)  LongTensor
      extra_tensor : (B, max_len, 1) FloatTensor  (or None)
    """
    B = len(history_ids)
    ids_arr = np.zeros((B, max_len), dtype=np.int64)
    for i, hist in enumerate(history_ids):
        hist = hist[-max_len:]
        ids_arr[i, :len(hist)] = hist

    ids_tensor = torch.from_numpy(ids_arr).to(device)

    if history_extras is not None:
        ext_arr = np.zeros((B, max_len, 1), dtype=np.float32)
        for i, hist in enumerate(history_extras):
            hist = hist[-max_len:]
            ext_arr[i, :len(hist), 0] = hist
        extra_tensor = torch.from_numpy(ext_arr).to(device)
    else:
        extra_tensor = None

    return ids_tensor, extra_tensor


def encode_history(
    history_ids: List[List[int]],
    history_extras: Optional[List[List[float]]],
    encoder: StateEncoder,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convenience wrapper: pad + encode a batch of histories.
    Returns (B, embed_dim) state tensor.
    """
    ids_t, ext_t = pad_history(history_ids, history_extras, max_len, device)
    with torch.no_grad():
        state = encoder(ids_t, ext_t)
    return state
