"""Bootstrap confidence intervals at the vessel-identity level.

Resampling happens over vessel identities (the natural sampling unit of the
trial protocol), not over individual frames, so correlated frames inside one
vessel's trajectory do not inflate the effective sample size.

- ``bootstrap_ci``: mean + percentile CI of a per-vessel score map.
- ``bootstrap_arm_diff``: paired bootstrap of ``score_b - score_a`` over the
  shared vessel IDs; the CI excluding zero indicates a significant difference.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _percentile_interval(values: np.ndarray, ci: float) -> tuple[float, float]:
    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    return float(lo), float(hi)


def bootstrap_ci(
    scores: dict[str, float],   # vessel_id -> score
    n_samples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Resample vessel identities with replacement and return
    ``(mean, lower_bound, upper_bound)`` of the mean score.
    """
    arr = np.asarray(list(scores.values()), dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci requires at least one vessel score")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = _percentile_interval(means, ci)
    return float(arr.mean()), lo, hi


def bootstrap_arm_diff(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    n_samples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap of ``(score_b - score_a)`` over shared vessel IDs.

    Returns ``(mean_diff, lower, upper)``; if the CI excludes zero the arms
    differ significantly (B better when the interval is entirely positive).
    """
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        raise ValueError("bootstrap_arm_diff requires at least one shared vessel ID")
    diffs = np.asarray([scores_b[v] - scores_a[v] for v in shared], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_samples, diffs.size))
    sample_means = diffs[idx].mean(axis=1)
    lo, hi = _percentile_interval(sample_means, ci)
    return float(diffs.mean()), lo, hi
