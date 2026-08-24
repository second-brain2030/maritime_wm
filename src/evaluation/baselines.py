"""External baseline rankers (brief P3: Arms F/G/H) on blackout episodes.

- Arm F: motion-only dead-reckoning — extrapolate the query bbox center,
  rank candidates by pixel distance of their reappearance bbox.
- Arm G: conventional tracker appearance embedding — cosine rank of raw
  (unprobed) frozen-backbone features at query vs reappearance.
- Arm H: AIS-fused upper bound — extrapolate the query AIS position, rank
  candidates by Haversine distance of their AIS position at reappearance.

All return (ranked_candidate_ids, drift_m) and share the re-acquisition
result format so baselines and probe arms are compared on identical trials.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from data.ais import AisPing
from data.manifest import TrackletManifest
from evaluation.blackout_harness import BlackoutEpisode
from evaluation.features import embed_frames, tracklet_visible_bboxes
from evaluation.reacquisition_eval import (
    ais_drift_m,
    bbox_center,
    pixel_drift_m,
    predict_bbox_center,
    predict_lonlat_from_pings,
    rank_by_cosine,
)
from evaluation.tracking_metrics import haversine_m


def deadreckon_rank(
    episode: BlackoutEpisode,
    target: TrackletManifest,
    candidates: Mapping[str, TrackletManifest],
    max_query_observations: int = 8,
) -> tuple[list[str], float | None]:
    """Arm F: position-only dead-reckoning ranking."""
    fps = target.fps or 25.0
    obs = [
        (fi / fps, bb)
        for fi, bb in tracklet_visible_bboxes(target)
        if fi < episode.blackout_start_frame
    ][-max_query_observations:]
    predicted = predict_bbox_center(obs, episode.blackout_duration_s)

    cand_pos: dict[str, np.ndarray] = {}
    for cid, ct in candidates.items():
        visible = tracklet_visible_bboxes(ct)
        if not visible:
            continue
        fi, bb = min(visible, key=lambda vb: abs(vb[0] - episode.reappearance_frame))
        cand_pos[cid] = bbox_center(bb)
    if predicted is None:
        ranked = list(cand_pos)
    else:
        ranked = sorted(cand_pos, key=lambda cid: float(np.linalg.norm(cand_pos[cid] - predicted)))
    drift = pixel_drift_m(predicted, episode.gt_bbox_at_reappearance)
    return ranked, drift


def appearance_rank(
    episode: BlackoutEpisode,
    target: TrackletManifest,
    candidates: Mapping[str, TrackletManifest],
    encoder,
    window_frames: int = 8,
    max_frames: int = 16,
) -> tuple[list[str], None]:
    """Arm G: raw frozen-backbone appearance embedding (no probe), cosine rank."""
    fps = target.fps or 25.0
    q_frames = [
        fi for fi, _ in tracklet_visible_bboxes(target) if fi < episode.blackout_start_frame
    ][-max_frames:]
    q_paths = [
        p for fi, p in zip(target.frame_indices or [], target.frame_paths) if fi in q_frames
    ]
    q_emb = embed_frames(encoder, None, q_paths, fps=fps)

    cand_embs: dict[str, np.ndarray] = {}
    for cid, ct in candidates.items():
        visible = tracklet_visible_bboxes(ct)
        if not visible:
            continue
        near = [fi for fi, _ in visible if abs(fi - episode.reappearance_frame) <= window_frames]
        if not near:
            fi, _ = min(visible, key=lambda vb: abs(vb[0] - episode.reappearance_frame))
            near = [fi]
        cand_embs[cid] = embed_frames(encoder, None, [p for fi, p in zip(ct.frame_indices or [], ct.frame_paths) if fi in near], fps=fps)
    return rank_by_cosine(q_emb, cand_embs), None


def ais_rank(
    episode: BlackoutEpisode,
    target_visible_pings: Sequence[AisPing],
    candidates_pings: Mapping[str, Sequence[AisPing]],
) -> tuple[list[str], float | None]:
    """Arm H: AIS-fused upper bound — Haversine rank of reappearance positions."""
    predicted = predict_lonlat_from_pings(target_visible_pings, episode.blackout_duration_s)
    reapp_utc = episode.blackout_start_utc_ms + int(episode.blackout_duration_s * 1000)
    cand_pos: dict[str, tuple[float, float]] = {}
    for cid, pings in candidates_pings.items():
        if not pings:
            continue
        near = min(pings, key=lambda p: abs(p.utc_ms - reapp_utc))
        cand_pos[cid] = (near.lon, near.lat)
    if predicted is None:
        ranked = list(cand_pos)
    else:
        ranked = sorted(
            cand_pos,
            key=lambda cid: haversine_m(predicted[0], predicted[1], cand_pos[cid][0], cand_pos[cid][1]),
        )
    drift = ais_drift_m(predicted, episode.gt_lonlat_at_reappearance)
    return ranked, drift
