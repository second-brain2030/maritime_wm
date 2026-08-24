"""Arm A: conventional Re-ID baseline (spec section 6.A).

Scaffold stub: instantiable so the registry and pipeline wire up; the
backbone (resnet50_ibn_a / osnet_x1_0) implementation lands in a later commit.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import TrackletEncoder


class CnnReidEncoder(TrackletEncoder):
    def __init__(
        self,
        backbone: str = "resnet50_ibn_a",
        pretrained: bool = True,
        embedding_dim: int = 2048,
        **kwargs,
    ) -> None:
        self.name = f"cnn_reid_{backbone}"
        self.backbone = backbone
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim

    def preprocess(self, frames: Tensor) -> Tensor:
        raise NotImplementedError("Arm A preprocessing lands with the backbone implementation")

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor) -> Tensor:
        raise NotImplementedError("Arm A encoder lands with the backbone implementation")

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor) -> Tensor | None:
        return None

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        return None
