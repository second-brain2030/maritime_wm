"""Arm B: VLA-derived vision encoders via Hugging Face transformers (brief P3).

Implements the static appearance towers inside the OpenVLA visual pathway:
DINOv2 and SigLIP (spec section 6.E: mandatory static baselines; spec section
6.D fallback). Frozen; per-frame tokens -> [B, T, D]. Models load lazily on
first encode so constructors are network-free (tests do not download).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .interfaces import TrackletEncoder

_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)


class HFVisionEncoder(TrackletEncoder):
    """Generic frozen HF vision model with per-frame pooling."""

    def __init__(
        self,
        name: str,
        checkpoint: str,
        embedding_dim: int,
        pool: str = "mean",
        input_size: int = 224,
        **kwargs,
    ) -> None:
        self.name = name
        self.checkpoint = checkpoint
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.input_size = input_size
        self._model = None
        self._mean = list(_DINO_MEAN)
        self._std = list(_DINO_STD)

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError("install transformers (requirements/vjepa.txt)") from e
        model = AutoModel.from_pretrained(self.checkpoint)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        self._model = model
        return model

    def preprocess(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5:
            raise ValueError(f"expected [B, T, 3, H, W], got {tuple(frames.shape)}")
        b, t = frames.shape[:2]
        x = frames.flatten(0, 1)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        mean = torch.tensor(self._mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self._std, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.view(b, t, 3, self.input_size, self.input_size)

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        model = self._load()
        x = self.preprocess(frames)
        b, t = x.shape[:2]
        out = model(x.flatten(0, 1))
        hidden = getattr(out, "last_hidden_state", None)  # [BT, L, D]
        if hidden is not None:
            feat = hidden.mean(dim=1) if self.pool == "mean" else hidden[:, 0]
        else:
            feat = out.pooler_output if out.pooler_output is not None else out.logits
        return feat.view(b, t, self.embedding_dim)

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor | None = None) -> Tensor | None:
        return None

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        return None


class DinoV2Encoder(HFVisionEncoder):
    def __init__(self, checkpoint: str = "facebook/dinov2-base", **kwargs) -> None:
        super().__init__(
            name="dinov2",
            checkpoint=checkpoint,
            embedding_dim=kwargs.pop("embedding_dim", 768),
            **kwargs,
        )


class SigLIPEncoder(HFVisionEncoder):
    def __init__(self, checkpoint: str = "google/siglip-base-patch16-224", **kwargs) -> None:
        super().__init__(
            name="siglip",
            checkpoint=checkpoint,
            embedding_dim=kwargs.pop("embedding_dim", 768),
            **kwargs,
        )
