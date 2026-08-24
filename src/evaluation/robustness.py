"""Robustness metrics for diagnostic stress suites (spec section 9).

For each stressor and severity: absolute_score, score_drop_from_clean,
relative_retained_performance, area_under_corruption_curve.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def robustness_report(
    clean_score: float,
    stressed_scores: Mapping[str, float],
) -> dict[str, Any]:
    """Per-severity robustness metrics; NotImplemented until the stress run lands."""
    raise NotImplementedError("robustness curves land with run_stress_suite (later commit)")
