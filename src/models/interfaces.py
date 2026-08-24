"""Representation adapter interfaces (spec section 11).

Every representation adapter implements ``TrackletEncoder``; the shared Re-ID
head is ``models.common_head.SharedReIDHead``.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class TrackletEncoder(Protocol):
    """Frozen backbone + optional predictor, as seen by the probe pipeline."""

    name: str
    embedding_dim: int

    def preprocess(self, frames: Tensor) -> Tensor: ...

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor) -> Tensor:
        """Return ``[batch, time_or_tokens, dim]`` features for observed frames."""

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor) -> Tensor | None:
        """Return predictor-derived latent features, or None if unsupported."""

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        """Return latent tokens predicted for the disappearance interval (Arm C
        gap-conditioned usage, spec section 6.C), or None if unsupported."""
