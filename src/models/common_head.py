"""Shared Re-ID head (spec sections 11/12).

Input tokens -> LayerNorm -> temporal attention pooling -> projection MLP
(embedding dim 512) -> L2 normalization -> identity classifier (training only).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .temporal_pooling import TemporalAttentionPool


class SharedReIDHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(token_dim)
        self.pool = TemporalAttentionPool(
            token_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout
        )
        self.proj = nn.Linear(token_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward(self, tokens: Tensor, token_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Return embedding (L2-normalized), logits (if num_classes), attn_weights."""
        x = self.norm(tokens)
        x = self.pool(x, token_mask)
        embedding = F.normalize(self.proj(x), dim=-1)
        out: dict[str, Tensor] = {"embedding": embedding}
        if self.classifier is not None:
            out["logits"] = self.classifier(embedding)
        return out
