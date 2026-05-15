"""
BERT4Rec — Sequential Recommendation with Bidirectional Encoder Representations from Transformer
Adapted for two-tower dense retrieval (ANN / FAISS).

Paper: Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, Peng Jiang
  "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"
  CIKM 2019.  https://arxiv.org/abs/1904.06690

Reference implementation ported from:
  https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch

Key differences from SASRec:
  1. Bidirectional attention — no causal mask, sees full history context
  2. Cloze training objective — mask random items, predict originals
  3. Inference via [MASK] token appended at end of history

Token scheme:
  0              : padding (ignored in attention)
  1 .. item_num  : real items
  item_num + 1   : [MASK] token used in training and inference
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention from the reference repo."""

    def forward(self, query, key, value, mask=None, dropout=None):
        # query/key/value: (B, h, L, d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-head attention — ported from reference repo."""

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h   = h
        self.linears    = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out_linear = nn.Linear(d_model, d_model)
        self.attn       = ScaledDotProductAttention()
        self.dropout    = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        # Project and split heads: (B, h, L, d_k)
        query, key, value = [
            l(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, _ = self.attn(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.out_linear(x)


# ---------------------------------------------------------------------------
# Feed-forward & sublayer connection
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1  = nn.Linear(d_model, d_ff)
        self.w2  = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.drop(F.gelu(self.w1(x))))


class SublayerConnection(nn.Module):
    """Pre-LN residual connection (norm → sublayer → dropout → add)."""

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.drop(sublayer(self.norm(x)))


# ---------------------------------------------------------------------------
# Transformer block (bidirectional)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.attn_sublayer = SublayerConnection(hidden, dropout)
        self.ffn_sublayer  = SublayerConnection(hidden, dropout)
        self.attention     = MultiHeadedAttention(heads, hidden, dropout)
        self.ffn           = PositionwiseFeedForward(hidden, hidden * 4, dropout)
        self.drop          = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.attn_sublayer(x, lambda q: self.attention(q, q, q, mask=mask))
        x = self.ffn_sublayer(x, self.ffn)
        return self.drop(x)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class BERTEmbedding(nn.Module):
    """
    Token embedding + learned positional embedding.
    Positional embedding is indexed by absolute position (0..maxlen-1),
    independent of token values.
    """

    def __init__(self, vocab_size: int, embed_size: int, max_len: int, dropout: float):
        super().__init__()
        self.token    = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = nn.Embedding(max_len, embed_size)
        self.dropout  = nn.Dropout(dropout)
        self.embed_size = embed_size

        nn.init.normal_(self.token.weight,    std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # (B, L)
        return self.dropout(self.token(x) + self.position(pos))


# ---------------------------------------------------------------------------
# BERT4Rec — main model
# ---------------------------------------------------------------------------

class BERT4Rec(nn.Module):
    """
    BERT4Rec adapted for two-tower dense retrieval.

    Retrieval interface (same as SASRec version in this repo):
      user_embedding(seqs)         → (B, d) normalized user vectors
      get_all_item_embeddings()    → (item_num, d) for FAISS indexing
    """

    def __init__(
        self,
        item_num:    int,
        maxlen:      int   = 50,
        hidden_dim:  int   = 128,
        num_heads:   int   = 2,
        num_blocks:  int   = 2,
        dropout_rate: float = 0.2,
        mask_prob:   float = 0.2,
    ):
        super().__init__()
        self.item_num   = item_num
        self.maxlen     = maxlen
        self.hidden_dim = hidden_dim
        self.mask_prob  = mask_prob
        # Token indices:  0=pad, 1..item_num=items, item_num+1=MASK
        self.mask_token = item_num + 1
        vocab_size      = item_num + 2

        self.embedding = BERTEmbedding(vocab_size, hidden_dim, maxlen, dropout_rate)
        self.blocks    = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.out_norm  = nn.LayerNorm(hidden_dim, eps=1e-6)

    # ------------------------------------------------------------------
    def _attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional padding mask.
        Returns (B, 1, L, L) where 0 means "ignore this key position".
        Padding positions (x==0) are masked out; all other tokens attend freely.
        """
        # non_pad: (B, L) bool, True = valid token
        non_pad = (x > 0)                                          # (B, L)
        # Expand to (B, 1, 1, L) so every query can see non-padding keys
        return non_pad.unsqueeze(1).unsqueeze(2)                   # (B, 1, 1, L)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L)  — token ids (0=pad, real items, or mask_token)
        Returns (B, L, d) contextualized embeddings.
        """
        mask = self._attention_mask(x)     # (B, 1, 1, L)
        h    = self.embedding(x)           # (B, L, d)
        for block in self.blocks:
            h = block(h, mask=mask)
        return self.out_norm(h)            # (B, L, d)

    # ------------------------------------------------------------------
    def user_embedding(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Produce normalized user vector for FAISS retrieval.
        Appends [MASK] after the history, takes embedding at that position.

        seqs: (B, L)  — purchase history (0=padding on left side)
        Returns (B, d) unit vectors.
        """
        B = seqs.size(0)
        mask_col = seqs.new_full((B, 1), self.mask_token)         # (B, 1)
        inp      = torch.cat([seqs, mask_col], dim=1)             # (B, L+1)
        # Keep only last maxlen tokens (left-truncate if needed)
        if inp.size(1) > self.maxlen:
            inp = inp[:, -self.maxlen:]
        out = self.encode(inp)                                     # (B, L', d)
        h   = out[:, -1]                                           # [MASK] position
        return F.normalize(h, dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_all_item_embeddings(self) -> torch.Tensor:
        """
        Normalized item embeddings for items 1..item_num.
        Returns (item_num, d) — row i corresponds to item_id i+1.
        """
        ids = torch.arange(1, self.item_num + 1, device=self.embedding.token.weight.device)
        return F.normalize(self.embedding.token(ids), dim=-1)

    # ------------------------------------------------------------------
    def forward(
        self,
        masked_seqs: torch.Tensor,   # (B, L) — history with MASK tokens inserted
        pos_ids:     torch.Tensor,   # (B, L) — original items at masked positions, 0 elsewhere
        neg_ids:     torch.Tensor,   # (B, L) — random negatives at masked positions, 0 elsewhere
    ):
        """
        Cloze forward pass: predict original items at [MASK] positions.
        Returns pos_logits, neg_logits for BCE loss (only at masked positions).
        """
        all_h   = self.encode(masked_seqs)                          # (B, L, d)
        all_h   = F.normalize(all_h, dim=-1)

        pos_emb = F.normalize(self.embedding.token(pos_ids.clamp(min=1)), dim=-1)  # (B, L, d)
        neg_emb = F.normalize(self.embedding.token(neg_ids.clamp(min=1)), dim=-1)  # (B, L, d)

        pos_logits = (all_h * pos_emb).sum(-1)                      # (B, L)
        neg_logits = (all_h * neg_emb).sum(-1)                      # (B, L)
        return pos_logits, neg_logits

    # ------------------------------------------------------------------
    @staticmethod
    def bce_loss(
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        mask:       torch.Tensor,   # (B, L) bool — True at [MASK] positions
    ) -> torch.Tensor:
        """BCE loss computed only at masked positions."""
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits[mask], torch.ones(mask.sum(), device=pos_logits.device)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits[mask], torch.zeros(mask.sum(), device=neg_logits.device)
        )
        return pos_loss + neg_loss
