"""
Generative Candidate Model  G_ψ

Role in the pipeline:
  C_t = G_ψ(x_t, z_t)

  x_t  : state context vector (B, embed_dim), output of StateEncoder
  z_t  : control knob from RL policy that steers candidate generation
         (e.g. latent noise / temperature shift)

Architecture:
  - Item embedding matrix shared with the RL policy
  - MLP that maps (x_t + z_t) -> candidate scores over all items
  - TopM sampling to produce candidate set C of size M

Training signal (Algorithm 3):
  - Contrastive / pairwise loss using (x_t, slate S_t, ground-truth y_t)
  - Positive: y_t  (next item)
  - Negatives: items in C_t \ {y_t}  (in-batch negatives)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorModel(nn.Module):
    """
    G_ψ: context + control knob -> candidate scores.

    Parameters
    ----------
    num_items    : total number of items in the catalogue
    embed_dim    : dimension of item / context embeddings
    hidden_dim   : hidden layer size
    latent_dim   : dimension of control knob z_t
    num_layers   : depth of the score MLP
    dropout      : dropout probability
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim

        # Shared item embedding (also used by RL policy)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        # MLP: (context [embed_dim] + knob [latent_dim]) -> hidden -> logits [num_items]
        layers: List[nn.Module] = []
        in_dim = embed_dim + latent_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, embed_dim))   # project to item space
        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        context: torch.Tensor,   # (B, embed_dim)
        knob: torch.Tensor,      # (B, latent_dim)
    ) -> torch.Tensor:           # (B, num_items) – raw logit scores
        h = self.mlp(torch.cat([context, knob], dim=-1))  # (B, embed_dim)
        # Dot product with all item embeddings
        item_embs = self.item_emb.weight                  # (N, embed_dim)
        item_embs = F.normalize(item_embs, dim=-1)
        h = F.normalize(h, dim=-1)
        scores = h @ item_embs.T                          # (B, N)
        return scores

    # ------------------------------------------------------------------
    def generate_candidates(
        self,
        context: torch.Tensor,   # (B, embed_dim)
        knob: torch.Tensor,      # (B, latent_dim)
        candidate_size: int,
        exclude_ids: List[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample top-M candidates per batch element.

        Returns
        -------
        cand_ids    : (B, M)  LongTensor – item indices
        cand_scores : (B, M)  FloatTensor – relevance scores
        """
        with torch.no_grad():
            scores = self.forward(context, knob)   # (B, N)

        if exclude_ids is not None:
            for b, excl in enumerate(exclude_ids):
                if excl:
                    scores[b, excl] = float("-inf")

        topk = min(candidate_size, self.num_items)
        cand_scores, cand_ids = torch.topk(scores, k=topk, dim=-1)  # (B, M)
        return cand_ids, cand_scores

    # ------------------------------------------------------------------
    def contrastive_loss(
        self,
        context: torch.Tensor,      # (B, embed_dim)
        knob: torch.Tensor,          # (B, latent_dim)
        pos_ids: torch.Tensor,       # (B,) positive item ids
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.
        Positive pair: (context, pos item embedding)
        Negatives: all other items in the batch (in-batch negatives)
        """
        scores = self.forward(context, knob)   # (B, N)
        # Scale by temperature
        scores = scores / temperature
        loss = F.cross_entropy(scores, pos_ids)
        return loss
