"""Evaluation package: Re-ID/track metrics, bootstrap, degradation, reports.

All modules import via full ``src.xxx`` prefixes (src/ is not a namespace
root). The legacy score-map APIs (``cmc``/``mean_average_precision``/
``rank_gallery``, ``tracking_metrics``, ``generate_report``) were superseded
by the embedding/distmat evaluation layer and are no longer exported; the
``tracking_metrics.py`` file itself is kept for backward reference.
"""
from .blackout_harness import (
    BlackoutConfig,
    BlackoutEpisode,
    BlackoutEpisodeManifest,
    BlackoutHarness,
    build_blackout_episodes,
)
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
    # Re-ID retrieval metrics
    "compute_distmat",
    "compute_map",
    "compute_cmc",
    "compute_metrics",
    # track metrics
    "TrackFrame",
    "compute_idsw",
    "compute_idf1",
    "compute_hota",
    # bootstrap
    "bootstrap_ci",
    "bootstrap_arm_diff",
    # degradation
    "DegradationResult",
    "compute_degradation",
    "compare_slopes",
    # reports
    "ReportWriter",
]
