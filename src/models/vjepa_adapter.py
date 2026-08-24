"""Arm C: frozen V-JEPA 2/2.1 encoder (spec section 6.B/C; brief P3).

Integration path (verified available): ``torch.hub.load("facebookresearch/vjepa2",
"vjepa2_1_vit_base_384", trust_repo=True)`` — the official repo ships a
``hubconf.py`` with 2.1 entries (base/large/giant/gigantic at 384px).

First use downloads the repo + pretrained weights; output-shape verification
must happen in the run environment after the download (not in CI). If the
public predictor interface cannot be reached, predictor methods report
``blocked_by_api`` and Arms A/B/D run without silent substitution
(spec section 6.C).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .interfaces import TrackletEncoder

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

HUB_MODELS = (
    "vjepa2_ac_vit_giant",
    "vjepa2_vit_giant",
    "vjepa2_vit_giant_384",
    "vjepa2_vit_huge",
    "vjepa2_vit_large",
    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant_384",
    "vjepa2_1_vit_gigantic_384",
)


class VJEPAEncoderAdapter(TrackletEncoder):
    def __init__(
        self,
        checkpoint: str = "facebookresearch/vjepa2:vjepa2_1_vit_base_384",
        frozen: bool = True,
        embedding_dim: int = 768,
        input_size: int = 384,
        **kwargs,
    ) -> None:
        if ":" not in checkpoint:
            raise ValueError(f"checkpoint must be '<repo>:<hub-model>', got {checkpoint!r}")
        repo, model = checkpoint.rsplit(":", 1)
        if model not in HUB_MODELS:
            raise ValueError(f"unknown V-JEPA hub model {model!r}; known: {HUB_MODELS}")
        self.name = f"vjepa_encoder_{model}"
        self.checkpoint = checkpoint
        self.frozen = frozen
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self._model: nn.Module | None = None

    def _load(self) -> nn.Module:
        if self._model is not None:
            return self._model
        try:
            import einops  # noqa: F401
            import timm  # noqa: F401
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError(
                "V-JEPA 2 requires timm + einops: pip install -r requirements/vjepa.txt"
            ) from e
        repo, model = self.checkpoint.rsplit(":", 1)
        self._model = torch.hub.load(repo, model, trust_repo=True)  # downloads weights
        if self.frozen:
            for p in self._model.parameters():
                p.requires_grad_(False)
        self._model.eval()
        return self._model

    def preprocess(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5:
            raise ValueError(f"expected [B, T, 3, H, W], got {tuple(frames.shape)}")
        import torch.nn.functional as F

        b, t = frames.shape[:2]
        x = frames.flatten(0, 1)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.view(b, t, 3, self.input_size, self.input_size)

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        model = self._load()
        x = self.preprocess(frames)
        b, t = x.shape[:2]
        out = model(x.flatten(0, 1))
        tokens = out[0] if isinstance(out, (tuple, list)) else out  # [BT, L, D]
        feat = tokens.mean(dim=1) if tokens.ndim == 3 else tokens
        return feat.view(b, t, self.embedding_dim)

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor | None = None) -> Tensor | None:
        raise NotImplementedError(
            "V-JEPA predictor future-latents: not exposed by the hub interface; "
            "mark blocked_by_api (spec section 6.C)"
        )

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        raise NotImplementedError(
            "V-JEPA gap-conditioned prediction blocked_by_api until the official "
            "predictor interface is wired (spec section 6.C [HARD TEST])"
        )
