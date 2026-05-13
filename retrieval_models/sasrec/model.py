"""
SASRec — Self-Attentive Sequential Recommendation (ICDM 2018)
Adapted for two-tower dense retrieval.

Architecture follows pmixer/SASRec.pytorch with two changes:
  1. Embeddings are L2-normalized → cosine similarity = inner product → FAISS-compatible
  2. user_embedding() and get_all_item_embeddings() exposed for retrieval use

Reference: Wang-Cheng Kang, Julian McAuley.
  "Self-Attentive Sequential Recommendation." ICDM 2018.
  https://arxiv.org/abs/1808.09781
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    """Position-wise FFN using 1D convolutions (same as original SASRec)."""

    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.conv1    = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu     = nn.ReLU()
        self.conv2    = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d) → transpose for Conv1d → (B, d, L) → back
        out = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(x.transpose(-1, -2)))))
        )
        return out.transpose(-1, -2)


class SASRec(nn.Module):
    """
    SASRec encoder adapted for retrieval (two-tower style).

    Items are 1-indexed: valid ids ∈ [1, item_num]. Index 0 is padding.

    Key retrieval interface:
      user_embedding(seqs)         → (B, d) normalized user vectors
      get_all_item_embeddings()    → (item_num, d) for FAISS indexing
    """

    def __init__(
        self,
        item_num: int,
        maxlen: int = 20,
        hidden_dim: int = 64,
        num_heads: int = 1,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.item_num   = item_num
        self.maxlen     = maxlen
        self.hidden_dim = hidden_dim

        self.item_emb     = nn.Embedding(item_num + 1, hidden_dim, padding_idx=0)
        self.pos_emb      = nn.Embedding(maxlen + 1,   hidden_dim, padding_idx=0)
        self.emb_dropout  = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers     = nn.ModuleList()
        self.forward_layernorms   = nn.ModuleList()
        self.forward_layers       = nn.ModuleList()
        self.last_layernorm       = nn.LayerNorm(hidden_dim, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_dim, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout_rate, batch_first=False)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_dim, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_dim, dropout_rate))

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,  std=0.02)

    # ------------------------------------------------------------------
    def encode_sequence(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        seqs: (B, L) item ids (0 = padding, left-padded).
        Returns (B, L, d) contextualized embeddings.
        """
        B, L = seqs.shape
        # Positional index: 1..L for non-padding, 0 for padding
        pos = torch.arange(1, L + 1, device=seqs.device).unsqueeze(0).expand(B, -1)
        pos = pos * (seqs != 0).long()

        x = self.item_emb(seqs) * (self.hidden_dim ** 0.5) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        # Causal (lower-triangular) mask: True = ignore in attention
        causal = ~torch.tril(torch.ones(L, L, dtype=torch.bool, device=seqs.device))

        for ln_q, attn, ln_ff, ff in zip(
            self.attention_layernorms, self.attention_layers,
            self.forward_layernorms,  self.forward_layers,
        ):
            # Pre-LN self-attention + residual
            x_t  = x.transpose(0, 1)
            q    = ln_q(x_t)
            out, _ = attn(q, x_t, x_t, attn_mask=causal)
            x    = x + out.transpose(0, 1)
            # Pre-LN FFN + residual
            x    = x + ff(ln_ff(x))

        return self.last_layernorm(x)   # (B, L, d)

    # ------------------------------------------------------------------
    def user_embedding(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Encode history and return normalized user embedding.
        seqs: (B, L) — last L purchased items, 0 = padding
        Returns: (B, d) unit vectors
        """
        out  = self.encode_sequence(seqs)                              # (B, L, d)
        lens = (seqs != 0).sum(dim=1).clamp(min=1) - 1                # (B,) last valid pos
        idx  = torch.arange(seqs.size(0), device=seqs.device)
        h    = out[idx, lens]                                          # (B, d)
        return F.normalize(h, dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_all_item_embeddings(self) -> torch.Tensor:
        """Normalized embeddings for items 1..item_num. Returns (item_num, d)."""
        ids = torch.arange(1, self.item_num + 1, device=self.item_emb.weight.device)
        return F.normalize(self.item_emb(ids), dim=-1)

    # ------------------------------------------------------------------
    def forward(
        self,
        seqs:    torch.Tensor,   # (B, L)
        pos_ids: torch.Tensor,   # (B,)   positive item ids
        neg_ids: torch.Tensor,   # (B, K) negative item ids
    ):
        """Training forward: returns (pos_logits, neg_logits)."""
        u_emb   = self.user_embedding(seqs)                           # (B, d)
        pos_emb = F.normalize(self.item_emb(pos_ids), dim=-1)         # (B, d)
        B, K    = neg_ids.shape
        neg_emb = F.normalize(
            self.item_emb(neg_ids.view(-1)), dim=-1
        ).view(B, K, self.hidden_dim)                                  # (B, K, d)

        pos_logits = (u_emb * pos_emb).sum(dim=-1)                    # (B,)
        neg_logits = torch.bmm(neg_emb, u_emb.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return pos_logits, neg_logits

    # ------------------------------------------------------------------
    @staticmethod
    def bce_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits)
        )
        return pos_loss + neg_loss
