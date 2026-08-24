"""Arm D: VLA-derived vision representation (spec section 6.D, 14).

Feature source must be recorded exactly:
  openvla_direct_fused | prismatic_dinosiglip | fallback_dinov2_plus_siglip
No language prompts in the primary fair comparison; no generated actions as
Re-ID features.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import TrackletEncoder

FEATURE_SOURCES = ("openvla_direct_fused", "prismatic_dinosiglip", "fallback_dinov2_plus_siglip")


class OpenVLAVisionAdapter(TrackletEncoder):
    def __init__(
        self,
        checkpoint: str = "openvla-7b",
        feature_source: str = "prismatic_dinosiglip",
        frozen: bool = True,
        embedding_dim: int = 1792,
        **kwargs,
    ) -> None:
        if feature_source not in FEATURE_SOURCES:
            raise ValueError(f"feature_source {feature_source!r} not in {FEATURE_SOURCES}")
        self.name = f"openvla_vision_{feature_source}"
        self.checkpoint = checkpoint  # PIN exact checkpoint and commit hash (spec §14)
        self.feature_source = feature_source
        self.frozen = frozen
        self.embedding_dim = embedding_dim

    def preprocess(self, frames: Tensor) -> Tensor:
        raise NotImplementedError("OpenVLA vision preprocessing lands with the adapter (later commit)")

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor) -> Tensor:
        raise NotImplementedError("OpenVLA fused vision features land with the adapter (later commit)")

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor) -> Tensor | None:
        return None

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        return None
