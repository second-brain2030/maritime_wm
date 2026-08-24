"""Intermittent observation sampler (pilot brief P2).

Temporal frame-skipping (burst observe/gap patterns) and deterministic
synthetic patch masking, used to build long-gap re-acquisition episodes on
single-object tracking data (MVTD) that has no AIS modality.
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np


def intermittent_observation_mask(
    n_frames: int,
    observe_frames: int,
    gap_frames: int,
    seed: int = 42,
) -> list[bool]:
    """Boolean per-frame observation mask: True = observed.

    Pattern: ``observe_frames`` observed, then ``gap_frames`` skipped, repeated
    from a seeded phase offset — simulating intermittent sensor coverage.
    """
    if n_frames <= 0 or observe_frames < 1 or gap_frames < 0:
        raise ValueError("n_frames > 0, observe_frames >= 1, gap_frames >= 0 required")
    rng = random.Random(seed)
    phase = rng.randrange(0, max(1, observe_frames))
    mask: list[bool] = []
    i = 0
    while len(mask) < n_frames:
        take = observe_frames if i % 2 == 0 else gap_frames
        i += 1
        for _ in range(take):
            if len(mask) >= n_frames:
                break
            mask.append(i % 2 == 1)  # odd blocks observed
        if gap_frames == 0 and i % 2 == 1:
            break
    if len(mask) < n_frames:
        # degenerate all-observed fallback
        mask = [True] * n_frames
    mask = mask[:n_frames]
    # phase shift
    return mask[phase:] + mask[:phase] if phase else mask


def find_gaps(mask: Sequence[bool], min_gap_frames: int = 1) -> list[tuple[int, int]]:
    """Contiguous unobserved runs as (start, end-exclusive) frame intervals."""
    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for i, observed in enumerate(mask):
        if not observed and start is None:
            start = i
        elif observed and start is not None:
            if i - start >= min_gap_frames:
                gaps.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_gap_frames:
        gaps.append((start, len(mask)))
    return gaps


def block_patch_mask(height: int, width: int, severity: float, seed: int = 42) -> np.ndarray:
    """Deterministic binary mask (True = masked patch) covering ~severity area.

    Used for synthetic patch masking at evaluation time (spec section 8.4
    block-occlusion diagnostic; brief P2).
    """
    if not (0.0 < severity < 1.0):
        raise ValueError("severity must be in (0, 1)")
    rng = random.Random(seed)
    area = height * width
    target = int(area * severity)
    # single centered-ish random rectangle; grows until it covers target area
    ph = int(round(height * (severity ** 0.5)))
    pw = int(round(width * (severity ** 0.5)))
    ph = min(max(1, ph), height)
    pw = min(max(1, pw), width)
    y0 = rng.randrange(0, height - ph + 1)
    x0 = rng.randrange(0, width - pw + 1)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y0 + ph, x0 : x0 + pw] = True
    return mask
