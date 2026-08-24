"""Arm B: frozen V-JEPA 2.1 encoder-only probe (spec section 6.B, 13).

Scaffold stub: wiring exists; HF AutoModel loading and token extraction land
in a later commit. PIN the exact checkpoint id and package versions.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import TrackletEncoder


class VJEPAEncoderAdapter(TrackletEncoder):
    def __init__(
        self,
        checkpoint: str = "facebook/vjepa2-<pin-exact-id>",
        frozen: bool = True,
        embedding_dim: int = 768,
        **kwargs,
    ) -> None:
        self.name = f"vjepa_encoder_{checkpoint.split('/')[-1]}"
        self.checkpoint = checkpoint
        self.frozen = frozen
        self.embedding_dim = embedding_dim

    def preprocess(self, frames: Tensor) -> Tensor:
        raise NotImplementedError(
            "V-JEPA preprocessing lands with the HF AutoModel integration (later commit)"
        )

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor) -> Tensor:
        raise NotImplementedError(
            "V-JEPA feature extraction lands with the HF AutoModel integration (later commit)"
        )

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor) -> Tensor | None:
        return None

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        return None
