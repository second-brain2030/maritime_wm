"""Arm A: conventional Re-ID baseline encoder (spec section 6.A; brief P3).

Frozen ImageNet-pretrained CNN trunk; per-frame features mean-pooled over the
spatial grid -> [B, T, D]. ``resnet50`` is implemented with torchvision;
``resnet50_ibn_a`` / ``osnet_x1_0`` remain configurable options requiring
torchreid (documented; NotImplemented until torchreid is installed).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .interfaces import TrackletEncoder

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class CnnReidEncoder(TrackletEncoder):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embedding_dim: int = 2048,
        input_size: int = 224,
        **kwargs,
    ) -> None:
        if backbone not in ("resnet50",):
            raise NotImplementedError(
                f"backbone {backbone!r} requires torchreid (resnet50_ibn_a/osnet_x1_0); "
                "use resnet50 for the current pilot (spec §6.A configurable)"
            )
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.name = f"cnn_reid_{backbone}"
        self._model: nn.Module | None = None

    def _load(self) -> nn.Module:
        if self._model is not None:
            return self._model
        import torchvision

        weights = (
            torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        )
        trunk = torchvision.models.resnet50(weights=weights)
        self._model = nn.Sequential(*list(trunk.children())[:-2])  # [B, 2048, 7, 7]
        if self.pretrained:
            for p in self._model.parameters():
                p.requires_grad_(False)
        self._model.eval()
        return self._model

    def preprocess(self, frames: Tensor) -> Tensor:
        """frames: [B, T, 3, H, W] floats in [0, 1] -> normalized, resized."""
        if frames.ndim != 5:
            raise ValueError(f"expected [B, T, 3, H, W], got {tuple(frames.shape)}")
        b, t = frames.shape[:2]
        x = frames.flatten(0, 1)
        size = self.input_size + 32
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        pad = (size - self.input_size) // 2
        x = x[..., pad : pad + self.input_size, pad : pad + self.input_size]
        mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.view(b, t, 3, self.input_size, self.input_size)

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        model = self._load()
        x = self.preprocess(frames)
        b, t = x.shape[:2]
        feat = model(x.flatten(0, 1))  # [B*T, 2048, 7, 7]
        feat = feat.mean(dim=(2, 3))  # [B*T, 2048]
        return feat.view(b, t, self.embedding_dim)

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor | None = None) -> Tensor | None:
        return None

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        return None
