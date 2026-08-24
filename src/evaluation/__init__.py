"""Evaluation package: metrics, robustness, bootstrap, degradation, reports."""
from .reid_metrics import cmc, mean_average_precision, rank_gallery
from .bootstrap import bootstrap_ci, paired_bootstrap_difference
from .degradation import degradation_curve, degradation_slope

__all__ = [
    "cmc",
    "mean_average_precision",
    "rank_gallery",
    "bootstrap_ci",
    "paired_bootstrap_difference",
    "degradation_curve",
    "degradation_slope",
]
