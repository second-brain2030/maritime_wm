"""Temporal pooling for tracklet token sequences (spec section 11/12).

Two pooling modules share one interface::

    forward(tokens: Tensor, mask: Tensor | None = None) -> Tensor

with input ``[B, T, D]`` and output ``[B, D]``. The mask convention is
``True = valid`` everywhere (padding entries are ``False``).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MeanPool(nn.Module):
    """Masked mean-pool over the time dimension.

    With ``mask=None`` this is a plain mean over ``T``; with a mask
    (``True = valid``) it averages only the valid entries and never
    divides by zero even when a row is entirely padding.
    """

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        weights = mask.unsqueeze(-1).float()  # [B, T, 1]
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class AttentionPool(nn.Module):
    """Single learned query vector attending over the token sequence.

    The query is a ``nn.Parameter`` of shape ``[1, 1, dim]`` that
    cross-attends to the tokens via ``nn.MultiheadAttention``
    (``batch_first=True``). The ``T`` axis of the MHA output is length 1
    and is squeezed away, yielding ``[B, dim]``.
    """

    def __init__(self, dim: int, num_heads: int = 1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"AttentionPool requires dim % num_heads == 0, got dim={dim}, num_heads={num_heads}"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(1, 1, dim) * (dim**-0.5))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        # nn.MultiheadAttention key_padding_mask semantics: True = ignore,
        # so flip the True=valid mask.
        key_padding_mask = None if mask is None else ~mask
        query = self.query.expand(tokens.shape[0], -1, -1)  # [B, 1, dim]
        out, _ = self.attn(
            query,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
        )
        out = out.squeeze(1)
        if mask is not None:
            # All-padding rows (softmax over zero valid keys -> NaN) get a
            # zero vector, mirroring MeanPool's all-False guard.
            valid = mask.any(dim=-1)  # [B]
            out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
        return out


def get_pooler(name: str, dim: int) -> nn.Module:
    """Factory returning ``MeanPool`` or ``AttentionPool`` by name."""
    if name == "mean":
        return MeanPool()
    if name == "attention":
        return AttentionPool(dim)
    raise ValueError(f"unknown pooler {name!r} (mean|attention)")
