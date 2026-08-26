"""Maneuver-during-gap stratification (spec §8.3; brief P1/P2).

Labels each blackout episode as ``straight`` / ``maneuver`` / ``unknown`` by
comparing the target vessel's bbox-center heading before the blackout with its
heading after reappearance. Large course change across the gap = maneuver
(dead-reckoning should fail); small = straight (linear extrapolation valid).
Labels are evaluation-only stratification from ground-truth tracks — never
model input.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from data.manifest import TrackletManifest
from evaluation.blackout_harness import BlackoutEpisode
from evaluation.features import tracklet_visible_bboxes

MANEUVER_THRESHOLD_DEG = 30.0


def _heading_deg(pts: list[tuple[float, float]]) -> float | None:
    """Course (deg, 0-360) of the best-fit displacement of the point track."""
    if len(pts) < 2:
        return None
    arr = np.asarray(pts, dtype=float)
    disp = arr[-1] - arr[0]
    norm = float(np.linalg.norm(disp))
    if norm < 1e-6:
        return None
    return math.degrees(math.atan2(disp[1], disp[0])) % 360.0


def _heading_delta(a: float, b: float) -> float:
    d = abs((b - a) % 360.0)
    return min(d, 360.0 - d)


def label_episode_maneuver(
    episode: BlackoutEpisode,
    target: TrackletManifest,
    threshold_deg: float = MANEUVER_THRESHOLD_DEG,
) -> str:
    """straight | maneuver | unknown for one episode.

    Pre-gap heading from the last few visible bbox centers before the blackout
    start; post-gap heading from the first visible bbox centers at/after
    reappearance. Unknown when either side has < 2 samples or stable points.
    """
    fps = target.fps or 25.0
    visible = tracklet_visible_bboxes(target)  # [(frame_idx, [x, y, w, h])]

    pre = [
        (bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0)
        for fi, bb in visible
        if fi < episode.blackout_start_frame
    ][-6:]
    post = [
        (bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0)
        for fi, bb in visible
        if fi >= episode.reappearance_frame
    ][:6]

    h_pre = _heading_deg(pre)
    h_post = _heading_deg(post)
    if h_pre is None or h_post is None:
        return "unknown"
    delta = _heading_delta(h_pre, h_post)
    return "maneuver" if delta >= threshold_deg else "straight"


def label_episodes(
    episodes,
    tracklet_map: Mapping[tuple[str, str], TrackletManifest],
) -> dict[str, str]:
    """episode_id -> label for every labelable episode."""
    out: dict[str, str] = {}
    for ep in episodes:
        t = tracklet_map.get((ep.sequence_id, ep.vessel_id))
        if t is None:
            out[ep.episode_id] = "unknown"
            continue
        out[ep.episode_id] = label_episode_maneuver(ep, t)
    return out