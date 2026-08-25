"""Arm B: frozen V-JEPA 2.1 encoder-only probe (spec §6.B / §13).

Loads official facebookresearch/vjepa2 weights via the local hubconf.py
(cloned at models/vjepa2/). No torch.hub network call needed — weights are
already in models/vjepa2/checkpoints/ and symlinked to the hub cache.

Checkpoint: vjepa2_1_vit_base_384
  encoder output: [B, 2304, 768]  (4 temporal × 576 spatial tokens at 384px)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .interfaces import TrackletEncoder

# Absolute path to the cloned vjepa2 repo (submodule)
_VJEPA2_ROOT = Path(__file__).resolve().parents[2] / "models" / "vjepa2"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


_vjepa2_cache: tuple | None = None

# vjepa2's internal submodules that conflict with our own `src.*` namespace.
# We load them explicitly under their vjepa2 paths and register them in
# sys.modules before importing hubconf, so Python never tries to resolve them
# via our project's src/ package.
_VJEPA2_SRC_SHIMS = [
    "src.utils.tensors",
    "src.utils",
    "src.models.vision_transformer",
    "src.models",
]


def _inject_vjepa2_src_shims() -> dict:
    """Pre-register vjepa2's src.* submodules in sys.modules before hubconf load.

    The vjepa2 predictor and vision_transformer both do bare `from src.xxx import`
    which resolves to our project's src/ if we don't intercept first.  We load
    only the two leaf modules actually needed (tensors + masks.utils), register
    them under the `src.*` namespace, then restore after the model is built.

    Returns displaced dict so caller can restore sys.modules afterwards.
    """
    import importlib.util
    import types

    displaced: dict = {}
    vjepa2_src = _VJEPA2_ROOT / "src"

    def _displace_and_register_pkg(key: str, path: Path) -> None:
        displaced[key] = sys.modules.get(key)
        pkg = types.ModuleType(key)
        pkg.__path__ = [str(path)]
        pkg.__package__ = key
        sys.modules[key] = pkg

    def _displace_and_exec_module(key: str, file: Path) -> None:
        displaced[key] = sys.modules.get(key)
        spec = importlib.util.spec_from_file_location(key, file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # Swap top-level src package to point at vjepa2's src/
    _displace_and_register_pkg("src", vjepa2_src)
    _displace_and_register_pkg("src.utils", vjepa2_src / "utils")
    _displace_and_register_pkg("src.masks", vjepa2_src / "masks")

    # Fully execute the two leaf modules that vjepa2's models import at top-level
    _displace_and_exec_module("src.utils.tensors", vjepa2_src / "utils" / "tensors.py")
    _displace_and_exec_module("src.masks.utils", vjepa2_src / "masks" / "utils.py")

    return displaced


def _restore_src_shims(displaced: dict) -> None:
    for key, val in displaced.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


def _load_vjepa2_vitb384() -> tuple:
    """Return (encoder, predictor) from local hubconf, weights from disk cache.

    Resolves the src.* namespace collision between vjepa2's internals and our
    own src/ package by temporarily registering vjepa2's src.utils.tensors
    (and related submodules) in sys.modules before hubconf is imported.
    """
    global _vjepa2_cache
    if _vjepa2_cache is not None:
        return _vjepa2_cache

    # Ensure hubconf is importable from vjepa2 root
    vjepa2_root = str(_VJEPA2_ROOT)
    if vjepa2_root not in sys.path:
        sys.path.insert(0, vjepa2_root)

    displaced = _inject_vjepa2_src_shims()
    try:
        import hubconf  # noqa: PLC0415
        encoder, predictor = hubconf.vjepa2_1_vit_base_384(pretrained=True)
    finally:
        _restore_src_shims(displaced)

    _vjepa2_cache = (encoder, predictor)
    return _vjepa2_cache


def _preprocess(frames: Tensor) -> Tensor:
    """Normalise and resize to 384px.

    Accepts [B, T, C, H, W] or [B, C, H, W] float tensors in [0, 1].
    Returns [B, C, T, 384, 384] (V-JEPA 2 expects channels-first video).
    """
    if frames.dim() == 4:               # [B, C, H, W] → single-frame video
        frames = frames.unsqueeze(2)    # [B, C, 1, H, W]
    elif frames.dim() == 5 and frames.shape[1] != 3:
        # [B, T, C, H, W] → [B, C, T, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)
    # frames: [B, C, T, H, W]
    B, C, T, H, W = frames.shape
    # Resize spatial dims to 384 via bilinear on each frame
    flat = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    if H != 384 or W != 384:
        flat = F.interpolate(flat, size=(384, 384), mode="bilinear", align_corners=False)
    # ImageNet normalise (assumes input in [0,1])
    if flat.max() <= 1.0 + 1e-4:
        mean = torch.tensor(_IMAGENET_MEAN, device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
        flat = (flat - mean) / std
    return flat.reshape(B, T, C, 384, 384).permute(0, 2, 1, 3, 4)  # [B, C, T, 384, 384]


class VJEPAEncoderAdapter(TrackletEncoder):
    """Arm B — frozen V-JEPA 2.1 ViT-B/384 encoder, no predictor.

    encode_observed returns all spatial-temporal tokens [B, N_tokens, 768].
    SharedReIDHead pools these to a fixed-size embedding.
    """

    name: str = "vjepa_encoder_vitb384"
    embedding_dim: int = 768
    CHECKPOINT: str = "vjepa2_1_vitb_dist_vitG_384"

    def __init__(
        self,
        device: str = "cpu",
        frozen: bool = True,
    ) -> None:
        encoder, _predictor = _load_vjepa2_vitb384()
        self.encoder = encoder.to(device).eval()
        self.device = device
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def preprocess(self, frames: Tensor) -> Tensor:
        return _preprocess(frames).to(self.device)

    @torch.no_grad()
    def encode_observed(
        self, frames: Tensor, frame_mask: Optional[Tensor] = None
    ) -> Tensor:
        """[B, T, C, H, W] or [B, C, T, H, W] → [B, N_tokens, 768].

        Runs encoder on all tokens (no masking) — the full observed clip.
        frame_mask is accepted for protocol compatibility but ignored here;
        masking is applied downstream by SharedReIDHead.
        """
        x = _preprocess(frames).to(self.device)   # [B, C, T, 384, 384]
        tokens = self.encoder(x)                  # [B, N_tokens, 768]
        return tokens

    @torch.no_grad()
    def encode_predicted(
        self, frames: Tensor, frame_mask: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Encoder-only arm: no predictor features."""
        return None

    @torch.no_grad()
    def predict_future(
        self, observed_tokens: Tensor, horizon: int
    ) -> Optional[Tensor]:
        """Encoder-only arm: no future prediction."""
        return None
