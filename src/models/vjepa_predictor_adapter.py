"""Arm C: frozen V-JEPA encoder + predictor probe (spec section 6.C, 13).

The predictor arm is the world-model hypothesis; gap-conditioned usage
(predict latent tokens into the disappearance interval) is spec section 6.C
[HARD TEST]. If the public V-JEPA interface cannot expose predictor features,
mark this arm ``blocked_by_api`` and run A/B/D without silent substitution.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import TrackletEncoder


class VJEPAPredictorAdapter(TrackletEncoder):
    def __init__(
        self,
        checkpoint: str = "facebook/vjepa2-<pin-exact-id>",
        frozen: bool = True,
        embedding_dim: int = 768,
        predictor_horizon_delta: float = 60.0,
        **kwargs,
    ) -> None:
        self.name = f"vjepa_predictor_{checkpoint.split('/')[-1]}"
        self.checkpoint = checkpoint
        self.frozen = frozen
        self.embedding_dim = embedding_dim
        self.predictor_horizon_delta = predictor_horizon_delta

    def preprocess(self, frames: Tensor) -> Tensor:
        raise NotImplementedError("V-JEPA preprocessing lands with the predictor integration")

    @torch.no_grad()
    def encode_observed(self, frames: Tensor, frame_mask: Tensor) -> Tensor:
        raise NotImplementedError("V-JEPA encoder extraction lands with the predictor integration")

    @torch.no_grad()
    def encode_predicted(self, frames: Tensor, frame_mask: Tensor) -> Tensor | None:
        raise NotImplementedError(
            "predictor future-latent extraction requires the official V-JEPA "
            "predictor API; mark blocked_by_api if unavailable (spec section 6.C)"
        )

    @torch.no_grad()
    def predict_future(self, observed_tokens: Tensor, horizon: int) -> Tensor | None:
        raise NotImplementedError(
            "gap-conditioned future prediction requires the official V-JEPA "
            "predictor API (spec section 6.C [HARD TEST])"
        )
