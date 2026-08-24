"""Canonical temporal sampling (spec section 7).

Modes: uniform | recent | sparse | prefix_only.
Short tracklets are padded by repeating an edge frame (``repeat_last`` /
``repeat_first``). Sampling is deterministic; ``seed`` is accepted for future
jittered variants.
"""
from __future__ import annotations

from typing import Sequence

from .manifest import TrackletManifest

MODES = ("uniform", "recent", "sparse", "prefix_only")
SHORT_POLICIES = ("repeat_last", "repeat_first", "pad_zeros")


def sample_frame_indices(
    n_frames: int,
    frames_per_tracklet: int,
    mode: str = "uniform",
    seed: int | None = None,
    k: int = 1,
    fraction: float = 0.5,
    short_policy: str = "repeat_last",
) -> list[int]:
    """Return frame indices for one tracklet under the requested regime."""
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0, got {n_frames}")
    if frames_per_tracklet <= 0:
        raise ValueError(f"frames_per_tracklet must be > 0, got {frames_per_tracklet}")
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if short_policy not in SHORT_POLICIES:
        raise ValueError(f"short_policy {short_policy!r} not in {SHORT_POLICIES}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if mode == "uniform":
        if n_frames >= frames_per_tracklet:
            if frames_per_tracklet == 1:
                idx = [0]
            else:
                idx = [
                    round(i * (n_frames - 1) / (frames_per_tracklet - 1))
                    for i in range(frames_per_tracklet)
                ]
        else:
            idx = list(range(n_frames))
    elif mode == "recent":
        idx = list(range(max(0, n_frames - frames_per_tracklet), n_frames))
    elif mode == "sparse":
        idx = list(range(0, n_frames, max(1, k)))
    elif mode == "prefix_only":
        n = max(1, round(n_frames * fraction))
        idx = list(range(n))
    else:  # pragma: no cover - guarded above
        raise AssertionError(mode)

    if len(idx) < frames_per_tracklet and mode != "sparse":
        if short_policy == "repeat_last":
            idx = idx + [idx[-1]] * (frames_per_tracklet - len(idx))
        elif short_policy == "repeat_first":
            idx = [idx[0]] * (frames_per_tracklet - len(idx)) + idx
        else:
            raise NotImplementedError("pad_zeros requires frame-level masking")
    if len(idx) > frames_per_tracklet and mode != "sparse":
        idx = idx[:frames_per_tracklet]
    return idx


def sample_frames(
    tracklet: TrackletManifest,
    frames_per_tracklet: int,
    mode: str = "uniform",
    seed: int | None = None,
    k: int = 1,
    fraction: float = 0.5,
    short_policy: str = "repeat_last",
) -> list[str]:
    idx = sample_frame_indices(
        len(tracklet.frame_paths),
        frames_per_tracklet,
        mode=mode,
        seed=seed,
        k=k,
        fraction=fraction,
        short_policy=short_policy,
    )
    return [tracklet.frame_paths[i] for i in idx]
