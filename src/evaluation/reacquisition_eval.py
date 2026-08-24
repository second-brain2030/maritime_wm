"""Re-acquisition evaluation glue (brief P1/P4).

Pure helpers: cosine ranking of candidates at the reappearance frame,
per-episode result records, and position drift estimation from AIS pings or
bounding-box trajectories. Kept free of torch so baseline and probe-based
evaluations share one code path.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from data.ais import AisPing
from evaluation.blackout_harness import BlackoutEpisode
from evaluation.tracking_metrics import haversine_m
from models.kinematic import predict_across_gap


def rank_by_cosine(
    query_embedding: np.ndarray, candidates: Mapping[str, np.ndarray]
) -> list[str]:
    """Rank candidate ids by cosine similarity to the query (descending)."""
    q = np.asarray(query_embedding, dtype=float).ravel()
    qn = q / (np.linalg.norm(q) + 1e-9)
    scored: list[tuple[str, float]] = []
    for cid, emb in candidates.items():
        e = np.asarray(emb, dtype=float).ravel()
        e = e / (np.linalg.norm(e) + 1e-9)
        scored.append((cid, float(qn @ e)))
    scored.sort(key=lambda t: -t[1])
    return [cid for cid, _ in scored]


def episode_result(
    episode: BlackoutEpisode, ranked: Sequence[str], drift_m: float | None = None
) -> dict:
    """Standard per-episode result record for summarize_reacquisition."""
    rank = None
    if episode.vessel_id in ranked:
        rank = ranked.index(episode.vessel_id) + 1
    return {
        "episode_id": episode.episode_id,
        "duration_s": episode.blackout_duration_s,
        "rank_of_correct": rank,
        "n_candidates": len(ranked),
        "drift_m": drift_m,
    }


# ---------------------------------------------------------------------------
# Drift estimation
# ---------------------------------------------------------------------------


def predict_lonlat_from_pings(
    pings: Sequence[AisPing], gap_s: float
) -> tuple[float, float] | None:
    """Extrapolate lon/lat after ``gap_s`` from the last two visible pings."""
    if len(pings) < 2:
        return None
    p1, p2 = pings[-2], pings[-1]
    dt_s = (p2.utc_ms - p1.utc_ms) / 1000.0
    if dt_s <= 0:
        return None
    vlon = (p2.lon - p1.lon) / dt_s
    vlat = (p2.lat - p1.lat) / dt_s
    return (p2.lon + vlon * gap_s, p2.lat + vlat * gap_s)


def ais_drift_m(
    predicted_lonlat: tuple[float, float] | None,
    gt_lonlat: Sequence[float] | None,
) -> float | None:
    if predicted_lonlat is None or gt_lonlat is None:
        return None
    return haversine_m(predicted_lonlat[0], predicted_lonlat[1], gt_lonlat[0], gt_lonlat[1])


def predict_bbox_center(
    observations: Sequence[tuple[float, Sequence[float]]], gap_s: float
) -> np.ndarray | None:
    """Constant-velocity prediction of the bbox center after ``gap_s``.

    ``observations``: [(t_seconds, [x, y, w, h]), ...] from before the blackout.
    """
    if len(observations) < 2:
        return None
    pts = [
        (t, bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0) for t, bb in observations
    ]
    pred, _ = predict_across_gap(pts, gap_s)
    return pred


def bbox_center(bb: Sequence[float]) -> np.ndarray:
    return np.array([bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0], dtype=float)


def pixel_drift_m(
    predicted: np.ndarray | None,
    gt_bbox: Sequence[float] | None,
) -> float | None:
    if predicted is None or gt_bbox is None:
        return None
    return float(np.linalg.norm(predicted - bbox_center(gt_bbox)))
