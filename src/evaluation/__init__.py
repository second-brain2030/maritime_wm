"""Evaluation package: re-acquisition, blackout, track metrics, reports.

Exports the blackout harness, leakage-safe re-acquisition evaluation helpers,
external baseline rankers, on-demand feature embedding, plus CC's merged
metrics layer (Re-ID retrieval, track metrics, bootstrap, degradation,
ReportWriter).
"""
from .blackout_harness import (
    BlackoutConfig,
    BlackoutEpisode,
    BlackoutEpisodeManifest,
    BlackoutHarness,
    build_blackout_episodes,
)
from .reacquisition_eval import (
    ais_drift_m,
    episode_result,
    pixel_drift_m,
    predict_bbox_center,
    predict_lonlat_from_pings,
    rank_by_cosine,
)
from .baselines import ais_rank, appearance_rank, deadreckon_rank
from .features import embed_frames, tracklet_visible_bboxes
from .reid_metrics import compute_cmc, compute_distmat, compute_map, compute_metrics
from .track_metrics import TrackFrame, compute_hota, compute_idf1, compute_idsw
from .bootstrap import bootstrap_arm_diff, bootstrap_ci
from .degradation import DegradationResult, compare_slopes, compute_degradation
from .reports import ReportWriter

__all__ = [
    # blackout harness
    "BlackoutConfig",
    "BlackoutEpisode",
    "BlackoutEpisodeManifest",
    "BlackoutHarness",
    "build_blackout_episodes",
    # re-acquisition helpers
    "ais_drift_m",
    "episode_result",
    "pixel_drift_m",
    "predict_bbox_center",
    "predict_lonlat_from_pings",
    "rank_by_cosine",
    # external baselines
    "ais_rank",
    "appearance_rank",
    "deadreckon_rank",
    # on-demand embedding
    "embed_frames",
    "tracklet_visible_bboxes",
    # Re-ID retrieval metrics (CC)
    "compute_distmat",
    "compute_map",
    "compute_cmc",
    "compute_metrics",
    # track metrics (CC)
    "TrackFrame",
    "compute_idsw",
    "compute_idf1",
    "compute_hota",
    # bootstrap (CC)
    "bootstrap_ci",
    "bootstrap_arm_diff",
    # degradation (CC)
    "DegradationResult",
    "compute_degradation",
    "compare_slopes",
    # reports (CC)
    "ReportWriter",
]
