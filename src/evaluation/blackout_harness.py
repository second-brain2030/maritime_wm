"""Sensor-blackout harness (pilot brief P1).

Generates controlled evaluation episodes by masking visual observations for
10s/30s/60s/120s, withholding AIS pings inside the blackout window (retaining
them as hidden ground truth), and adding timestamp jitter + dropout to model
asynchronous AIS messaging. Candidate identity matching is then measured at
the reappearance frame.

Episodes are deterministic by seed and serialized to JSONL for reproducible
evaluation runs.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from data.ais import AisTrajectory, split_pings_by_window
from data.manifest import TrackletManifest

DEFAULT_DURATIONS_S = [10.0, 30.0, 60.0, 120.0]


@dataclass
class BlackoutConfig:
    durations_s: list[float] = field(default_factory=lambda: list(DEFAULT_DURATIONS_S))
    min_visible_before_s: float = 5.0
    min_visible_after_s: float = 5.0
    withhold_ais: bool = True
    jitter_ms: int = 0        # +/- timestamp jitter on visible AIS pings
    ais_dropout_p: float = 0.0
    pool_size: int = 5        # co-present candidates at reappearance (incl. target)
    episodes_per_duration_per_tracklet: int = 1
    seed: int = 42

    def validate(self) -> None:
        if not self.durations_s or any(d <= 0 for d in self.durations_s):
            raise ValueError("durations_s must be non-empty positive values")
        if not (0.0 <= self.ais_dropout_p <= 1.0):
            raise ValueError("ais_dropout_p must be in [0, 1]")
        if self.pool_size < 1:
            raise ValueError("pool_size must be >= 1")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BlackoutConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BlackoutEpisode:
    episode_id: str
    sequence_id: str
    vessel_id: str
    split: str
    blackout_duration_s: float
    blackout_start_frame: int
    blackout_start_utc_ms: int
    reappearance_frame: int
    query_frame_indices: list[int]
    candidate_vessel_ids: list[str]          # includes the target vessel
    gt_bbox_at_reappearance: list[float] | None
    gt_lonlat_at_reappearance: list[float] | None  # [lon, lat] from withheld AIS
    withheld_ping_count: int = 0
    ais_withheld: bool = True

    def validate(self) -> None:
        if not self.episode_id or not self.vessel_id:
            raise ValueError("episode_id and vessel_id required")
        if self.vessel_id not in self.candidate_vessel_ids:
            raise ValueError("target vessel must be in candidate_vessel_ids")
        if not self.query_frame_indices:
            raise ValueError("query_frame_indices empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BlackoutEpisode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _visible_frame_map(tracklet: TrackletManifest) -> dict[int, int]:
    """frame_index -> position in the tracklet's observation list."""
    if tracklet.frame_indices is None:
        raise ValueError(f"tracklet {tracklet.tracklet_id!r} has no frame_indices")
    return {fi: i for i, fi in enumerate(tracklet.frame_indices)}


def build_blackout_episodes(
    tracklets: Sequence[TrackletManifest],
    ais_by_vessel: Mapping[str, AisTrajectory] | None = None,
    config: BlackoutConfig | None = None,
) -> list[BlackoutEpisode]:
    """Build deterministic blackout episodes from tracklet manifests.

    Requires per-frame ``frame_indices`` + ``fps`` + ``frame_timestamps_utc_ms``
    (FVessel adapter output). AIS trajectories are matched by ``vessel_id``;
    pings inside the blackout window are withheld and their positions become
    the hidden ground truth for localization drift.
    """
    config = config or BlackoutConfig()
    config.validate()
    rng = random.Random(config.seed)
    episodes: list[BlackoutEpisode] = []

    by_sequence: dict[str, list[TrackletManifest]] = {}
    for t in tracklets:
        by_sequence.setdefault(t.camera_id, []).append(t)

    for tracklet in tracklets:
        if tracklet.frame_indices is None or tracklet.fps is None:
            continue
        if tracklet.frame_timestamps_utc_ms is None:
            continue
        visible = sorted(tracklet.frame_indices)
        lookup = _visible_frame_map(tracklet)
        fps = tracklet.fps

        for duration_s in config.durations_s:
            gap_frames = int(round(duration_s * fps))
            windows: list[tuple[int, int]] = []
            for i, b in enumerate(visible):
                if i == 0:
                    continue
                # before/after spans are TIME-based, not observation-count-based
                # (annotations may be sparse, e.g. 1 Hz GT at 25 fps video)
                span_before_s = (b - visible[0]) / fps
                if span_before_s < config.min_visible_before_s:
                    continue
                reapp = [v for v in visible if v >= b + gap_frames]
                if not reapp:
                    continue
                r = reapp[0]
                span_after_s = (visible[-1] - r) / fps
                if span_after_s < config.min_visible_after_s:
                    continue
                windows.append((b, r))
            windows = windows[: config.episodes_per_duration_per_tracklet]
            for b, r in windows:
                query_idx = [v for v in visible if v < b]
                start_utc = tracklet.frame_timestamps_utc_ms[lookup[b]]
                end_utc = start_utc + int(duration_s * 1000)
                withheld_count = 0
                gt_lonlat: list[float] | None = None
                if config.withhold_ais and ais_by_vessel:
                    traj = ais_by_vessel.get(tracklet.vessel_id)
                    if traj is not None:
                        _, withheld = split_pings_by_window(
                            traj.pings,
                            start_utc,
                            end_utc,
                            jitter_ms=config.jitter_ms,
                            dropout_p=config.ais_dropout_p,
                            seed=config.seed,
                        )
                        withheld_count = len(withheld)
                        reapp_utc = start_utc + int(duration_s * 1000)
                        if withheld:
                            nearest = min(withheld, key=lambda p: abs(p.utc_ms - reapp_utc))
                            gt_lonlat = [nearest.lon, nearest.lat]
                candidates = _co_present_candidates(
                    tracklet, r, by_sequence.get(tracklet.camera_id, []), config.pool_size, rng
                )
                episode = BlackoutEpisode(
                    episode_id=f"blk_{tracklet.tracklet_id}_{int(duration_s)}s_{b}",
                    sequence_id=tracklet.camera_id,
                    vessel_id=tracklet.vessel_id,
                    split=tracklet.split,
                    blackout_duration_s=duration_s,
                    blackout_start_frame=b,
                    blackout_start_utc_ms=start_utc,
                    reappearance_frame=r,
                    query_frame_indices=query_idx,
                    candidate_vessel_ids=candidates,
                    gt_bbox_at_reappearance=tracklet.frame_bboxes[lookup[r]] if tracklet.frame_bboxes else None,
                    gt_lonlat_at_reappearance=gt_lonlat,
                    withheld_ping_count=withheld_count,
                    ais_withheld=config.withhold_ais,
                )
                episode.validate()
                episodes.append(episode)
    episodes.sort(key=lambda e: e.episode_id)
    return episodes


def _co_present_candidates(
    tracklet: TrackletManifest,
    frame: int,
    sequence_tracklets: Sequence[TrackletManifest],
    pool_size: int,
    rng: random.Random,
) -> list[str]:
    """Other vessels visible at the reappearance frame (co-present distractors)."""
    others: list[str] = []
    for t in sequence_tracklets:
        if t.vessel_id == tracklet.vessel_id:
            continue
        if t.frame_indices is not None and frame in set(t.frame_indices):
            others.append(t.vessel_id)
    others = sorted(set(others))
    rng.shuffle(others)
    chosen = [tracklet.vessel_id] + others[: max(0, pool_size - 1)]
    return sorted(set(chosen))


class BlackoutEpisodeManifest:
    """JSONL store for blackout episodes."""

    def __init__(self, episodes: Iterable[BlackoutEpisode] = ()) -> None:
        self._episodes: list[BlackoutEpisode] = list(episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self):
        return iter(self._episodes)

    def append(self, episode: BlackoutEpisode) -> None:
        episode.validate()
        self._episodes.append(episode)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for e in self._episodes:
                f.write(json.dumps(e.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str) -> "BlackoutEpisodeManifest":
        episodes: list[BlackoutEpisode] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(BlackoutEpisode.from_dict(json.loads(line)))
        return cls(episodes)

    def summary(self) -> dict[str, Any]:
        from collections import Counter

        return {
            "n_episodes": len(self._episodes),
            "by_duration_s": dict(Counter(e.blackout_duration_s for e in self._episodes)),
            "mean_pool_size": round(
                sum(len(e.candidate_vessel_ids) for e in self._episodes) / max(1, len(self._episodes)), 2
            ),
            "by_split": dict(Counter(e.split for e in self._episodes)),
        }
