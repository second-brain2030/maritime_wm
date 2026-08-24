"""Evaluation package: metrics, robustness, bootstrap, degradation, harness."""
from .reid_metrics import cmc, mean_average_precision, rank_gallery
from .bootstrap import bootstrap_ci, paired_bootstrap_difference
from .degradation import chance_normalized, degradation_curve, degradation_slope
from .tracking_metrics import (
    haversine_m,
    hota,
    id_switches,
    idf1,
    pixel_distance,
    reacquisition_topk,
    summarize_reacquisition,
)
from .blackout_harness import (
    BlackoutConfig,
    BlackoutEpisode,
    BlackoutEpisodeManifest,
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

__all__ = [
    "cmc",
    "mean_average_precision",
    "rank_gallery",
    "bootstrap_ci",
    "paired_bootstrap_difference",
    "chance_normalized",
    "degradation_curve",
    "degradation_slope",
    "haversine_m",
    "hota",
    "id_switches",
    "idf1",
    "pixel_distance",
    "reacquisition_topk",
    "summarize_reacquisition",
    "BlackoutConfig",
    "BlackoutEpisode",
    "BlackoutEpisodeManifest",
    "build_blackout_episodes",
    "ais_drift_m",
    "episode_result",
    "pixel_drift_m",
    "predict_bbox_center",
    "predict_lonlat_from_pings",
    "rank_by_cosine",
    "ais_rank",
    "appearance_rank",
    "deadreckon_rank",
    "embed_frames",
    "tracklet_visible_bboxes",
]
