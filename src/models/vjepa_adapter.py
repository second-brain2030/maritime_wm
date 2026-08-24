"""Arm C: frozen V-JEPA 2/2.1 encoder + predictor (spec section 6.B/C; brief P3).

Integration path: official ``facebookresearch/vjepa2`` weights (HuggingFace /
pytorch-hub) + the official inference code, pinned by checkpoint id. The
public interface may not expose predictor future-latents reliably; in that
case this arm reports ``blocked_by_api`` and Arms A/B/D run without silent
substitution (spec section 6.C).
"""
from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import TrackletEncoder


class VJEPAEncoderAdapter(TrackletEncoder):
    def __init__(
        self,
        checkpoint: str = "facebookresearch/vjepa2:vitl16",
        frozen: bool = True,
        embedding_dim: int = 1024,
        **kwargs,
    ) -> None:
        self.name = f"vjepa_encoder_{checkpoint.split('/')[-1]}"
        self.checkpoint = checkpoint
        self.frozen = frozen
        self.embedding_dim = embedding_dim

    def _load(self):
        """Load the official V-JEPA 2 model; raises with integration guidance."""
        try:
            import vjepa2  # noqa: F401  (official package, if installable)
        except ImportError as e:
            raise NotImplementedError(
                "V-JEPA 2 integration is blocked_by_api until the official "
                "facebookresearch/vjepa2 package/checkpoints are pinned in this "
                "environment (spec §6.C); install per requirements/vjepa.txt "
                "and pin the checkpoint id in configs/models/vjepa_encoder.yaml"
            ) from e
        raise NotImplementedError(
            "V-JEPA 2 forward pass: implement with the official inference code "
            "once the pinned package is available (spec §13)"
        )

    def preprocess(self, frames: Tensor) -> Tensor:
        raise NotImplementedError("V-JEPA preprocessing requires the official preprocessing")

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        self._load()
        raise NotImplementedError("V-JEPA feature extraction blocked_by_api")

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor | None = None) -> Tensor | None:
        self._load()
        raise NotImplementedError("V-JEPA predictor features blocked_by_api")

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        self._load()
        raise NotImplementedError("V-JEPA gap-conditioned prediction blocked_by_api")
