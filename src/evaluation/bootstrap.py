"""Bootstrap statistics (spec section 9).

Bootstrap at vessel identity level, not frame level; use paired bootstrap
differences between arms on identical queries/trials.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_samples: int = 2000,
    seed: int = 42,
    statistic: Callable = np.mean,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Mean + 95% percentile bootstrap CI over ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci requires non-empty values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    stats = statistic(arr[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(statistic(arr)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def paired_bootstrap_difference(
    a: Sequence[float],
    b: Sequence[float],
    n_samples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap CI of mean(a - b) on identical trials (spec §9)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired bootstrap requires equal-length non-empty arrays")
    diffs = a - b
    ci = bootstrap_ci(diffs.tolist(), n_samples=n_samples, seed=seed, statistic=np.mean)
    return {
        "mean_diff": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "excludes_zero": ci["ci_low"] > 0 or ci["ci_high"] < 0,
    }
