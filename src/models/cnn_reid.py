"""Arm A: conventional Re-ID baseline (spec section 6.A / 11).

Frozen CNN backbone (torchreid build_model when available, else torchvision
resnet50 fallback) that encodes each frame into a per-frame descriptor.
Implements the ``TrackletEncoder`` protocol; there is no predictor, so
``encode_predicted`` / ``predict_future`` return ``None``.
"""
from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .interfaces import TrackletEncoder

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class CNNReIDEncoder(TrackletEncoder):
    name: str = "cnn_reid"

    def __init__(
        self,
        backbone: str = "osnet_x1_0",
        pretrained: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.device = device
        model, feat_dim = self._load_backbone(backbone, pretrained)
        self.backbone = model.to(device).eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.embedding_dim = feat_dim

    def _load_backbone(self, backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
        """Return (frozen backbone, feature_dim). Prefers torchreid; falls
        back to torchvision resnet50 (fc replaced with Identity)."""
        try:  # lazy: keep module import cheap and dependency-light
            import torchreid
        except ImportError:  # pragma: no cover - exercised when torchreid absent
            torchreid = None

        if torchreid is not None:
            model = torchreid.models.build_model(
                name=backbone, num_classes=1, pretrained=pretrained
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                features = (
                    model.features(dummy)
                    if hasattr(model, "features")
                    else model(dummy)
                )
                feat_dim = int(features.shape[1])
            return model, feat_dim

        warnings.warn(
            "torchreid not available; falling back to torchvision resnet50 "
            f"for backbone={backbone!r}",
            stacklevel=2,
        )
        import torchvision.models as tv_models

        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet50(weights=weights)
        model.fc = nn.Identity()  # expose [B, 2048] features
        return model, 2048

    def preprocess(self, frames: Tensor) -> Tensor:
        """Resize to 256 (short side) -> center-crop 224, ImageNet-normalize
        when the input is in [0, 1]. Accepts [B, T, C, H, W] or [B, C, H, W];
        returns [B, T, C, 224, 224]."""
        if frames.dim() == 4:  # [B, C, H, W] -> [B, 1, C, H, W]
            frames = frames.unsqueeze(1)
        if frames.dim() != 5:
            raise ValueError(f"expected [B,T,C,H,W] or [B,C,H,W], got {tuple(frames.shape)}")

        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        from torchvision.transforms import functional as TF  # lazy: keep import cheap

        resized = TF.resize(flat, 256)  # short side -> 256, keep aspect
        cropped = TF.center_crop(resized, 224)  # [B*T, C, 224, 224]

        if cropped.max() <= 1.0 + 1e-5:  # unnormalized [0, 1] input
            mean = torch.tensor(_IMAGENET_MEAN, device=cropped.device).view(1, C, 1, 1)
            std = torch.tensor(_IMAGENET_STD, device=cropped.device).view(1, C, 1, 1)
            cropped = (cropped - mean) / std

        return cropped.reshape(B, T, C, 224, 224)

    @torch.no_grad()
    def encode_observed(
        self, frames: Tensor, frame_mask: Tensor | None = None
    ) -> Tensor:
        """Encode each frame: [B, T, C, H, W] -> [B, T, D], L2-normalized.
        ``frame_mask`` is accepted for protocol compatibility but the CNN
        encodes every frame; masking is applied downstream (e.g. by the
        SharedReIDHead's ``token_mask``)."""
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W).to(
            next(self.backbone.parameters()).device
        )
        feats = self.backbone(flat)  # [B*T, D]
        feats = feats.reshape(B, T, -1)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_predicted(
        self, frames: Tensor, frame_mask: Tensor | None = None
    ) -> None:
        return None  # CNN has no predictor

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> None:
        return None  # CNN has no predictor
