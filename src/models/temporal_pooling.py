"""Temporal pooling for tracklet token sequences (spec section 11/12)."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalMeanPool(nn.Module):
    """Mean-pool over valid time tokens (mask: True = valid)."""

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        weights = mask.unsqueeze(-1).float()
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class TemporalAttentionPool(nn.Module):
    """2-layer temporal Transformer over time tokens, then mean-pool (spec §12)."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = TemporalMeanPool()

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        src_key_padding_mask = None if mask is None else ~mask
        out = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        return self.pool(out, mask)


def build_temporal_pool(name: str, token_dim: int, **kwargs) -> nn.Module:
    """Factory: 'mean' or 'attention' (configurable temporal head)."""
    if name == "mean":
        return TemporalMeanPool()
    if name == "attention":
        return TemporalAttentionPool(token_dim, **kwargs)
    raise ValueError(f"unknown temporal head {name!r} (mean|attention)")
