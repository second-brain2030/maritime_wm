"""Disappearance-gap re-acquisition (DGRA) trial construction.

Implements spec sections 4.4 and 7.1: deterministic construction of
cross-camera / cross-time query->gallery trials, gap-duration binning, and
synthetic full-disappearance trials via contiguous block hold-out.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from .manifest import TrackletManifest

GAP_TYPES = ("natural_cross_camera", "synthetic_within_tracklet")
GAP_DURATION_SOURCES = ("timestamp", "frame_count", "unknown")
MANEUVERS = ("straight", "maneuver", "unknown")
BIN_NAMES = ("short", "medium", "long")
AIS_MODES = ("withheld", "available")


@dataclass
class GapTrial:
    """One query -> gallery re-acquisition trial across a disappearance gap.

    Mirror of the gap-trials manifest schema in spec section 4.4.
    ``pool_size == 0`` with ``distractor_pool_id is None`` means the trial has
    no distractor pool assigned yet (excluded from pool-based analyses).
    """

    trial_id: str
    query_tracklet_id: str
    gallery_tracklet_id: str
    vessel_id: str
    gap_seconds: float | None = None
    gap_duration_source: str = "unknown"
    gap_type: str = "natural_cross_camera"
    maneuver_during_gap: str = "unknown"
    distractor_pool_id: str | None = None
    pool_size: int = 0
    ais_available_at_test: bool = False
    split: str = "test"
    gap_bin: str | None = None

    def validate(self) -> None:
        errors: list[str] = []
        if not self.trial_id:
            errors.append("trial_id empty")
        if not self.query_tracklet_id:
            errors.append("query_tracklet_id empty")
        if not self.gallery_tracklet_id:
            errors.append("gallery_tracklet_id empty")
        if not self.vessel_id:
            errors.append("vessel_id empty")
        if self.gap_type not in GAP_TYPES:
            errors.append(f"gap_type {self.gap_type!r} not in {GAP_TYPES}")
        if self.gap_duration_source not in GAP_DURATION_SOURCES:
            errors.append(
                f"gap_duration_source {self.gap_duration_source!r} not in {GAP_DURATION_SOURCES}"
            )
        if self.maneuver_during_gap not in MANEUVERS:
            errors.append(f"maneuver_during_gap {self.maneuver_during_gap!r} not in {MANEUVERS}")
        if self.gap_bin is not None and self.gap_bin not in BIN_NAMES:
            errors.append(f"gap_bin {self.gap_bin!r} not in {BIN_NAMES}")
        if self.pool_size < 0:
            errors.append("pool_size < 0")
        if self.pool_size == 0 and self.distractor_pool_id is not None:
            errors.append("pool_size == 0 but distractor_pool_id set")
        if self.pool_size > 0 and self.distractor_pool_id is None:
            errors.append("pool_size > 0 but distractor_pool_id missing")
        if self.gap_duration_source == "timestamp" and self.gap_seconds is None:
            errors.append("gap_duration_source=timestamp requires gap_seconds")
        if self.gap_duration_source == "frame_count" and self.gap_seconds is None:
            errors.append("gap_duration_source=frame_count requires estimated gap_seconds")
        if self.gap_duration_source == "unknown" and self.gap_seconds is not None:
            errors.append("gap_duration_source=unknown forbids gap_seconds")
        if errors:
            raise ValueError(f"GapTrial {self.trial_id!r}: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GapTrial":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GapProtocolConfig:
    """DGRA protocol configuration (spec section 7.1)."""

    enabled: bool = True
    gap_bins_seconds: dict[str, list[float | None]] = field(
        default_factory=lambda: {
            "short": [10.0, 30.0],
            "medium": [60.0, 120.0],
            "long": [300.0, None],
        }
    )
    gap_bins_frames: dict[str, list[int | None]] = field(
        default_factory=lambda: {
            "short": [25, 90],
            "medium": [150, 360],
            "long": [900, None],
        }
    )
    pool_sizes: list[int] = field(default_factory=lambda: [5, 10, 20])
    min_pool_similarity: str = "high"
    maneuver_slices: list[str] = field(default_factory=lambda: list(MANEUVERS))
    ais_modes: list[str] = field(default_factory=lambda: list(AIS_MODES))
    synthetic_gap_holdout: float = 0.5
    predictor_horizon_delta: float = 60.0
    nominal_fps: float | None = None
    max_natural_pairs_per_vessel: int | None = None
    seed: int = 42

    def validate(self) -> None:
        errors: list[str] = []
        if not (0.0 < self.synthetic_gap_holdout < 1.0):
            errors.append("synthetic_gap_holdout must be in (0, 1)")
        for bins in (self.gap_bins_seconds, self.gap_bins_frames):
            for name, (lo, hi) in bins.items():
                if name not in BIN_NAMES:
                    errors.append(f"unknown bin name {name!r}")
                if lo is None or (hi is not None and hi <= lo):
                    errors.append(f"bin {name!r} must have lo <= hi")
        if not self.pool_sizes or any(s < 1 for s in self.pool_sizes):
            errors.append("pool_sizes must be non-empty positive integers")
        for mode in self.ais_modes:
            if mode not in AIS_MODES:
                errors.append(f"ais_mode {mode!r} not in {AIS_MODES}")
        if errors:
            raise ValueError("GapProtocolConfig: " + "; ".join(errors))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GapProtocolConfig":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


def assign_gap_bin(
    gap_seconds: float | None,
    gap_frames: int | None,
    gap_duration_source: str,
    config: GapProtocolConfig,
) -> str | None:
    """Assign short/medium/long using seconds (timestamps) or frame counts.

    Returns None for unbinned gaps (unknown source or out-of-range); those are
    kept for the all-gaps composite but excluded from per-bin analyses.
    """
    if gap_duration_source == "timestamp" and gap_seconds is not None:
        for name, (lo, hi) in config.gap_bins_seconds.items():
            hi_ = hi if hi is not None else math.inf
            if lo <= gap_seconds < hi_:
                return name
        return None
    if gap_duration_source == "frame_count" and gap_frames is not None:
        for name, (lo, hi) in config.gap_bins_frames.items():
            hi_ = hi if hi is not None else math.inf
            if lo <= gap_frames < hi_:
                return name
        return None
    return None


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _gap_between_tracklets(q: TrackletManifest, g: TrackletManifest) -> tuple[float | None, str]:
    """Seconds between q's last frame and g's first frame, or (None, 'unknown')."""
    q_end, g_start = _parse_iso(q.timestamp_end), _parse_iso(g.timestamp_start)
    if q_end is not None and g_start is not None:
        return max(0.0, (g_start - q_end).total_seconds()), "timestamp"
    return None, "unknown"


def _with_ais(trial: GapTrial, ais_mode: str) -> GapTrial:
    t = dataclasses.replace(trial)
    t.ais_available_at_test = ais_mode == "available"
    if ais_mode not in t.trial_id:
        t.trial_id = f"{t.trial_id}-{ais_mode}"
    return t


def _assign_pool(trial: GapTrial, pools: Mapping[str, Any] | None, config: GapProtocolConfig) -> None:
    if pools is None or not config.pool_sizes:
        return
    size = config.pool_sizes[0]
    match = next(
        (
            p
            for p in pools.values()
            if p.target_vessel_id == trial.vessel_id and p.size == size
        ),
        None,
    )
    if match is None:
        return
    trial.distractor_pool_id = match.pool_id
    trial.pool_size = match.size


def build_gap_trials(
    tracklets: Sequence[TrackletManifest],
    config: GapProtocolConfig,
    distractor_pools: Mapping[str, Any] | None = None,
) -> list[GapTrial]:
    """Deterministically build DGRA trials from tracklet manifests.

    - Natural trials: same vessel, different camera, query-split query tracklet
      and gallery-split gallery tracklet; gap from timestamps when available.
    - Synthetic trials: contiguous middle-block hold-out of one tracklet; the
      removed block IS the full disappearance.

    Distractor pools are assigned when provided (pool target vessel must match
    and pool size must equal the first configured size); otherwise trials carry
    pool_size 0 and are excluded from pool-based analyses.
    """
    config.validate()
    for t in tracklets:
        t.validate()
    rng = random.Random(config.seed)
    trials: list[GapTrial] = []

    by_vessel: dict[str, list[TrackletManifest]] = {}
    for t in tracklets:
        by_vessel.setdefault(t.vessel_id, []).append(t)

    # --- natural cross-camera trials -------------------------------------
    for vessel_id in sorted(by_vessel):
        group = sorted(by_vessel[vessel_id], key=lambda t: t.tracklet_id)
        queries = [t for t in group if t.split == "query"]
        galleries = [t for t in group if t.split == "gallery"]
        pairs = [(q, g) for q in queries for g in galleries if q.camera_id != g.camera_id]
        if config.max_natural_pairs_per_vessel is not None:
            pairs = pairs[: config.max_natural_pairs_per_vessel]
        for q, g in pairs:
            gap_seconds, source = _gap_between_tracklets(q, g)
            trial = GapTrial(
                trial_id=f"nat-{q.tracklet_id}->{g.tracklet_id}",
                query_tracklet_id=q.tracklet_id,
                gallery_tracklet_id=g.tracklet_id,
                vessel_id=vessel_id,
                gap_seconds=gap_seconds,
                gap_duration_source=source,
                gap_type="natural_cross_camera",
            )
            trial.gap_bin = assign_gap_bin(gap_seconds, None, source, config)
            _assign_pool(trial, distractor_pools, config)
            for ais_mode in config.ais_modes:
                t2 = _with_ais(trial, ais_mode)
                t2.validate()
                trials.append(t2)

    # --- synthetic full-disappearance trials ------------------------------
    for t in sorted(tracklets, key=lambda t: t.tracklet_id):
        if t.split not in ("query", "gallery"):
            continue
        n = len(t.frame_paths)
        if n < 4:
            continue
        block_len = max(1, round(n * config.synthetic_gap_holdout))
        start = round((n - block_len) / 2)
        if start < 1 or start + block_len > n - 1:
            continue  # cannot leave non-empty query and gallery
        gap_frames = block_len
        gap_seconds = round(block_len / config.nominal_fps, 3) if config.nominal_fps else None
        source = "frame_count" if config.nominal_fps else "unknown"
        trial = GapTrial(
            trial_id=f"syn-{t.tracklet_id}-{start}-{block_len}",
            query_tracklet_id=t.tracklet_id,
            gallery_tracklet_id=t.tracklet_id,
            vessel_id=t.vessel_id,
            gap_seconds=gap_seconds,
            gap_duration_source=source,
            gap_type="synthetic_within_tracklet",
        )
        trial.gap_bin = assign_gap_bin(gap_seconds, gap_frames, source, config)
        _assign_pool(trial, distractor_pools, config)
        for ais_mode in config.ais_modes:
            t2 = _with_ais(trial, ais_mode)
            t2.validate()
            trials.append(t2)

    trials.sort(key=lambda tr: tr.trial_id)
    rng.shuffle(trials)  # deterministic given seed; stable final order
    return trials


class GapTrialManifest:
    """JSONL store for gap trials."""

    def __init__(self, trials: Iterable[GapTrial] = ()) -> None:
        self._trials: list[GapTrial] = list(trials)

    def __len__(self) -> int:
        return len(self._trials)

    def __iter__(self):
        return iter(self._trials)

    def append(self, trial: GapTrial) -> None:
        trial.validate()
        self._trials.append(trial)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for t in self._trials:
                f.write(json.dumps(t.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str) -> "GapTrialManifest":
        trials: list[GapTrial] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trials.append(GapTrial.from_dict(json.loads(line)))
        return cls(trials)

    def summary(self) -> dict[str, Any]:
        return {
            "n_trials": len(self._trials),
            "by_gap_type": dict(Counter(t.gap_type for t in self._trials)),
            "by_bin": dict(Counter(t.gap_bin for t in self._trials)),
            "by_pool_size": dict(Counter(t.pool_size for t in self._trials)),
            "by_ais": dict(Counter(t.ais_available_at_test for t in self._trials)),
        }
