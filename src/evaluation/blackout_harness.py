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
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
    # --- task-spec fields (phase 3b blackout harness) ---
    query_tracklet_id: str = ""               # pre-blackout tracklet
    gallery_tracklet_ids: list[str] = field(default_factory=list)  # candidates at reappearance (target first)
    target_tracklet_id: str = ""              # ground-truth identity at reappearance
    blackout_seconds: float = 0.0             # 10 / 30 / 60 / 120
    ais_ground_truth: list[dict] = field(default_factory=list)  # hidden AIS pings during blackout window
    timestamp_jitter_ms: float = 0.0          # simulated jitter applied to AIS timestamps

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


def _parse_ais_timestamp(value: Any) -> datetime | None:
    """Parse an AIS ping timestamp (datetime or ISO-8601 string) to aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class BlackoutHarness:
    """Task-spec disappearance-gap harness (phase 3b).

    Generates controlled evaluation episodes from normalized tracklets: each
    tracklet is split at its midpoint into pre-gap (query) and post-gap
    (gallery) segments with a ``duration``-second visual blackout in between.
    AIS pings inside the blackout window are withheld and retained as hidden
    ground truth; their timestamps are jittered with Gaussian noise to model
    asynchronous AIS messaging. Deterministic by a fixed seed throughout.
    """

    BLACKOUT_DURATIONS = [10.0, 30.0, 60.0, 120.0]  # seconds

    def __init__(
        self,
        manifest: Sequence[TrackletManifest],
        config: Mapping[str, Any],
        seed: int = 42,
    ) -> None:
        self.manifest = list(manifest)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.blackout_durations = [
            float(d) for d in config.get("blackout_durations", self.BLACKOUT_DURATIONS)
        ]
        lo, hi = config.get("ais_jitter_ms_range", [0, 500])
        self.ais_jitter_ms_range = (float(lo), float(hi))
        self.gallery_pool_size = int(config.get("gallery_pool_size", 10))
        self.synthetic_gap_holdout = float(config.get("synthetic_gap_holdout", 0.5))

    # ------------------------------------------------------------------ API
    def build(self) -> list[BlackoutEpisode]:
        """Build episodes for all configured durations, sorted by
        (blackout_seconds, episode_id). Deterministic: fixed seed throughout."""
        episodes: list[BlackoutEpisode] = []
        for duration in self.blackout_durations:
            episodes.extend(self._build_episodes_for_duration(duration))
        episodes.sort(key=lambda e: (e.blackout_seconds, e.episode_id))
        return episodes

    def save_episodes(self, episodes: Sequence[BlackoutEpisode], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for e in episodes:
                f.write(json.dumps(e.to_dict(), sort_keys=True, default=str) + "\n")

    def load_episodes(self, path: str | Path) -> list[BlackoutEpisode]:
        episodes: list[BlackoutEpisode] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(BlackoutEpisode.from_dict(json.loads(line)))
        return episodes

    @staticmethod
    def evaluate_arm(
        episodes: Sequence[BlackoutEpisode],
        rankings: Mapping[str, Sequence[str]],  # episode_id -> ranked gallery_tracklet_ids
    ) -> dict[float, dict[str, float | int]]:
        """Re-acquisition accuracy per blackout duration bin.

        Returns ``{duration: {top1: float, top5: float, num_episodes: int}}``
        where top1/top5 are the fractions of episodes whose target tracklet is
        ranked first / within the top 5 of the gallery ranking.
        """
        by_duration: dict[float, list[BlackoutEpisode]] = {}
        for ep in episodes:
            by_duration.setdefault(float(ep.blackout_seconds), []).append(ep)
        out: dict[float, dict[str, float | int]] = {}
        for duration in sorted(by_duration):
            eps = by_duration[duration]
            top1 = top5 = 0
            for ep in eps:
                ranked = list(rankings.get(ep.episode_id, []) or [])
                if not ranked:
                    continue
                if ranked[0] == ep.target_tracklet_id:
                    top1 += 1
                if ep.target_tracklet_id in ranked[:5]:
                    top5 += 1
            out[duration] = {
                "top1": top1 / len(eps),
                "top5": top5 / len(eps),
                "num_episodes": len(eps),
            }
        return out

    # ------------------------------------------------------------- internal
    def _build_episodes_for_duration(self, duration: float) -> list[BlackoutEpisode]:
        episodes: list[BlackoutEpisode] = []
        for tracklet in self.manifest:
            fps = tracklet.fps
            n = len(tracklet.frame_paths)
            if fps is None or fps <= 0 or n == 0:
                continue
            if n <= 2 * duration * fps:  # need > 2*duration of visible footage
                continue
            mid_s = (n - 1) / fps / 2.0
            gap_start_s = mid_s - duration / 2.0
            gap_end_s = mid_s + duration / 2.0
            query_idx = [i for i in range(n) if i / fps < gap_start_s]
            gallery_idx = [i for i in range(n) if i / fps > gap_end_s]
            if not query_idx or not gallery_idx:
                continue

            pool = self._gallery_pool(tracklet, self.gallery_pool_size)
            gallery_tracklet_ids = [tracklet.tracklet_id] + [t.tracklet_id for t in pool]
            candidate_vessel_ids = [tracklet.vessel_id] + [t.vessel_id for t in pool]
            candidate_vessel_ids = list(dict.fromkeys(candidate_vessel_ids))

            jitter_ms = self.rng.uniform(*self.ais_jitter_ms_range)
            anchor_ms = self._time_anchor_ms(tracklet)
            ais_gt = self._ais_ground_truth(
                tracklet, gap_start_s, gap_end_s, fps, jitter_ms, anchor_ms
            )

            reappearance_frame = gallery_idx[0]
            reappearance_s = reappearance_frame / fps
            gt_lonlat = self._nearest_lonlat(ais_gt, reappearance_s, anchor_ms)
            gt_bbox = None
            if tracklet.frame_bboxes and reappearance_frame < len(tracklet.frame_bboxes):
                gt_bbox = tracklet.frame_bboxes[reappearance_frame]

            episode = BlackoutEpisode(
                episode_id=f"blk_{tracklet.tracklet_id}_{int(duration)}s",
                sequence_id=tracklet.camera_id,
                vessel_id=tracklet.vessel_id,
                split=tracklet.split,
                blackout_duration_s=duration,
                blackout_start_frame=int(round(gap_start_s * fps)),
                blackout_start_utc_ms=int(gap_start_s * 1000),
                reappearance_frame=reappearance_frame,
                query_frame_indices=query_idx,
                candidate_vessel_ids=candidate_vessel_ids,
                gt_bbox_at_reappearance=gt_bbox,
                gt_lonlat_at_reappearance=gt_lonlat,
                withheld_ping_count=len(ais_gt),
                ais_withheld=True,  # primary evaluation withholds AIS
                query_tracklet_id=tracklet.tracklet_id,
                gallery_tracklet_ids=gallery_tracklet_ids,
                target_tracklet_id=tracklet.tracklet_id,
                blackout_seconds=duration,
                ais_ground_truth=ais_gt,
                timestamp_jitter_ms=round(jitter_ms, 3),
            )
            episodes.append(episode)
        return episodes

    def _gallery_pool(
        self, tracklet: TrackletManifest, pool_size: int
    ) -> list[TrackletManifest]:
        """Sample ``pool_size``-1 candidates from same-split tracklets.

        Candidates sharing the target's ``vessel_type`` are preferred (class
        similarity); the remainder are drawn uniformly. Deterministic via the
        harness's fixed RNG.
        """
        same_split = [
            t
            for t in self.manifest
            if t.split == tracklet.split and t.tracklet_id != tracklet.tracklet_id
        ]
        target_type = tracklet.vessel_type
        similar = [t for t in same_split if target_type and t.vessel_type == target_type]
        rest = [t for t in same_split if t not in similar]
        self.rng.shuffle(similar)
        self.rng.shuffle(rest)
        chosen = similar[: pool_size - 1]
        if len(chosen) < pool_size - 1:
            chosen += rest[: pool_size - 1 - len(chosen)]
        return chosen

    def _ais_ground_truth(
        self,
        tracklet: TrackletManifest,
        gap_start_s: float,
        gap_end_s: float,
        fps: float,
        jitter_ms: float,
        anchor_ms: int | None,
    ) -> list[dict]:
        """AIS pings inside the blackout window with jittered timestamps.

        Frame time is anchored to ``frame_timestamps_utc_ms[0]`` when present,
        else to the earliest AIS ping timestamp; without either anchor no pings
        can be aligned and ``[]`` is returned.
        """
        traj = tracklet.ais_trajectory
        if not traj or anchor_ms is None:
            return []
        std_s = jitter_ms / 1000.0
        out: list[dict] = []
        for ping in traj:
            ts = _parse_ais_timestamp(ping.get("timestamp"))
            lat, lon = ping.get("lat"), ping.get("lon")
            if ts is None or lat is None or lon is None:
                continue
            rel_s = (int(ts.timestamp() * 1000) - anchor_ms) / 1000.0
            if gap_start_s <= rel_s <= gap_end_s:
                jittered = ts + timedelta(seconds=self.rng.gauss(0.0, std_s))
                out.append(
                    {
                        "timestamp": jittered.isoformat(),
                        "mmsi": str(ping.get("mmsi", "")),
                        "lat": float(lat),
                        "lon": float(lon),
                        "sog": float(ping["sog"]) if ping.get("sog") is not None else None,
                        "cog": float(ping["cog"]) if ping.get("cog") is not None else None,
                    }
                )
        return out

    def _time_anchor_ms(self, tracklet: TrackletManifest) -> int | None:
        """Epoch-ms anchor for frame time 0 (frame index 0)."""
        if tracklet.frame_timestamps_utc_ms:
            return tracklet.frame_timestamps_utc_ms[0]
        if tracklet.ais_trajectory:
            stamps = [
                _parse_ais_timestamp(p.get("timestamp"))
                for p in tracklet.ais_trajectory
                if p.get("timestamp") is not None
            ]
            stamps = [s for s in stamps if s is not None]
            if stamps:
                return int(min(stamps).timestamp() * 1000)
        return None

    @staticmethod
    def _nearest_lonlat(
        ais_gt: Sequence[dict], reappearance_s: float, anchor_ms: int | None
    ) -> list[float] | None:
        """[lon, lat] of the withheld ping nearest the reappearance time."""
        if not ais_gt or anchor_ms is None:
            return None
        reapp_ms = anchor_ms + int(reappearance_s * 1000)
        best, best_d = None, float("inf")
        for ping in ais_gt:
            ts = _parse_ais_timestamp(ping.get("timestamp"))
            if ts is None:
                continue
            d = abs(int(ts.timestamp() * 1000) - reapp_ms)
            if d < best_d:
                best_d, best = d, [ping["lon"], ping["lat"]]
        return best
