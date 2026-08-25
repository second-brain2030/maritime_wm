"""Arm C: frozen V-JEPA 2.1 encoder + predictor (spec §6.C / §13).

The predictor arm is the world-model hypothesis: given observed pre-gap tokens,
the frozen predictor generates latent tokens for the disappearance interval.
These predicted tokens are concatenated with observed tokens and passed to the
shared Re-ID head — making the matching mechanism literally "predict the latent
state of the not-yet-visible vessel, then match against gallery at reappearance."

Gap-conditioned usage (Arm C7, spec §6.C [HARD TEST]):
    observe query clip up to T_gap → predict tokens for [T_gap, T_gap+horizon]
    → match predicted tokens against gallery tokens at reappearance.

Predictor API (from models/vjepa2/app/vjepa_2_1/models/predictor.py):
    forward(x, masks_x, masks_y, mod="video", mask_index=1)
      x        : context encoder tokens  [B, N_ctx, D]
      masks_x  : list of LongTensor — indices of context patches in full sequence
      masks_y  : list of LongTensor — indices of target  patches in full sequence
    returns predicted tokens for masks_y positions  [B, N_tgt, D]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from .interfaces import TrackletEncoder
from .vjepa_adapter import _load_vjepa2_vitb384, _preprocess

_VJEPA2_ROOT = Path(__file__).resolve().parents[2] / "models" / "vjepa2"

# Spatial patches per frame at 384px / patch_size 16: (384/16)^2 = 576
# Temporal patch size for ViT-B/384: 2 → T frames → T//2 temporal tokens
_SPATIAL_PATCHES = 576   # 24 × 24
_TEMPORAL_STRIDE = 2     # frames per temporal patch


def _temporal_token_indices(t_start: int, t_end: int, n_frames: int) -> Tensor:
    """Return flat token indices for temporal range [t_start, t_end) frames.

    Full token sequence layout: [temporal_token_0, temporal_token_1, ...]
    where each temporal token spans _TEMPORAL_STRIDE frames and covers all
    _SPATIAL_PATCHES spatial patches.

    t_start, t_end are frame indices (not temporal-token indices).
    """
    n_temporal = n_frames // _TEMPORAL_STRIDE
    total_tokens = n_temporal * _SPATIAL_PATCHES

    # Convert frame range to temporal-token range
    ti_start = t_start // _TEMPORAL_STRIDE
    ti_end   = min((t_end + _TEMPORAL_STRIDE - 1) // _TEMPORAL_STRIDE, n_temporal)

    indices = []
    for ti in range(ti_start, ti_end):
        base = ti * _SPATIAL_PATCHES
        indices.extend(range(base, base + _SPATIAL_PATCHES))
    return torch.tensor(indices, dtype=torch.long)


class VJEPAPredictorAdapter(TrackletEncoder):
    """Arm C — frozen V-JEPA 2.1 encoder + predictor.

    Primary use (encode_predicted): split clip into observed prefix and masked
    suffix; run predictor to produce latent tokens for the suffix.

    Gap-conditioned use (predict_future): given pre-gap observed tokens,
    generate tokens for the next `horizon` temporal steps.
    """

    name: str = "vjepa_predictor_vitb384"
    # Predictor projects into the teacher's embedding space (1664d for vitG teacher)
    embedding_dim: int = 1664
    CHECKPOINT: str = "vjepa2_1_vitb_dist_vitG_384"

    def __init__(
        self,
        device: str = "cpu",
        frozen: bool = True,
        obs_fraction: float = 0.6,   # fraction of clip used as context
        predictor_horizon_delta: float = 60.0,  # seconds (logged per trial)
    ) -> None:
        """
        obs_fraction: fraction of temporal tokens treated as observed context.
            The remaining (1 - obs_fraction) are predicted by the predictor.
        """
        encoder, predictor = _load_vjepa2_vitb384()
        self.encoder  = encoder.to(device).eval()
        self.predictor = predictor.to(device).eval()
        self.device = device
        self.obs_fraction = obs_fraction
        self.predictor_horizon_delta = predictor_horizon_delta

        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.predictor.parameters():
                p.requires_grad_(False)

    def preprocess(self, frames: Tensor) -> Tensor:
        return _preprocess(frames).to(self.device)

    @torch.no_grad()
    def encode_observed(
        self, frames: Tensor, frame_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Encode observed (context) tokens only — prefix of the clip.

        Returns [B, N_obs_tokens, 768] — the visible portion of the latent space.
        Concatenate with encode_predicted output before passing to SharedReIDHead.
        """
        x = _preprocess(frames).to(self.device)   # [B, C, T, 384, 384]
        T = x.shape[2]
        n_temporal = T // _TEMPORAL_STRIDE
        n_obs = max(1, int(n_temporal * self.obs_fraction))

        # Observed indices: first n_obs temporal slots × all spatial patches
        obs_idx = torch.arange(n_obs * _SPATIAL_PATCHES, device=self.device)

        # Encode full clip; then select observed tokens
        all_tokens = self.encoder(x)              # [B, N_total, 768]
        return all_tokens[:, obs_idx, :]          # [B, N_obs, 768]

    @torch.no_grad()
    def encode_predicted(
        self, frames: Tensor, frame_mask: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Predict latent tokens for the masked suffix of the clip.

        Returns [B, N_pred_tokens, 768] — predictor's best guess at the
        future/occluded portion, without ever seeing those frames.
        Combine with encode_observed for the full Arm C token sequence.
        """
        x = _preprocess(frames).to(self.device)   # [B, C, T, 384, 384]
        B, C, T, H, W = x.shape
        n_temporal = T // _TEMPORAL_STRIDE
        n_obs  = max(1, int(n_temporal * self.obs_fraction))
        n_pred = n_temporal - n_obs
        if n_pred == 0:
            return None   # no future tokens to predict

        # Encode full clip; split into context / target
        all_tokens = self.encoder(x)              # [B, N_total, 768]
        obs_end  = n_obs  * _SPATIAL_PATCHES
        ctx_tokens = all_tokens[:, :obs_end, :]   # [B, N_obs, 768]

        # Build index masks for predictor — shape must be [B, K] (2D)
        masks_x = [torch.arange(obs_end, device=self.device).unsqueeze(0).expand(B, -1)]
        masks_y = [torch.arange(obs_end, obs_end + n_pred * _SPATIAL_PATCHES,
                                device=self.device).unsqueeze(0).expand(B, -1)]

        # Predictor returns (x_pred, x_context) with return_all_tokens=True
        result = self.predictor(
            ctx_tokens, masks_x=masks_x, masks_y=masks_y, mod="video"
        )
        pred_tokens = result[0] if isinstance(result, (tuple, list)) else result
        return pred_tokens                         # [B, N_pred, predictor_dim]

    @torch.no_grad()
    def predict_future(
        self, observed_tokens: Tensor, horizon: int
    ) -> Optional[Tensor]:
        """Gap-conditioned prediction (Arm C7, spec §6.C [HARD TEST]).

        Given observed_tokens [B, N_obs, 768] from the pre-gap clip,
        generate `horizon` additional temporal token slots.

        horizon: number of temporal token slots (not frames) to predict.
        Returns [B, horizon * SPATIAL_PATCHES, 768] or None if horizon==0.
        """
        if horizon <= 0:
            return None

        B, N_obs, D = observed_tokens.shape
        n_pred = horizon
        dev = observed_tokens.device

        # Masks must be [B, K] (2D)
        masks_x = [torch.arange(N_obs, device=dev).unsqueeze(0).expand(B, -1)]
        masks_y = [torch.arange(N_obs, N_obs + n_pred * _SPATIAL_PATCHES,
                                device=dev).unsqueeze(0).expand(B, -1)]

        result = self.predictor(
            observed_tokens, masks_x=masks_x, masks_y=masks_y, mod="video"
        )
        pred_tokens = result[0] if isinstance(result, (tuple, list)) else result
        return pred_tokens   # [B, n_pred * SPATIAL_PATCHES, predictor_dim]
