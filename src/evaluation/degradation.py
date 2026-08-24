"""Gap-degradation curves and slopes (spec section 9 [HARD TEST]).

The headline signal is the DEGRADATION CURVE: accuracy vs gap-duration bin,
one curve per arm, plus the linear-fit slope of chance-normalized accuracy vs
log gap. Flatter slope = more robust; the "advantage widens with gap" gate
compares slopes across arms (spec sections 8.5, 9).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

BIN_ORDER = {"short": 0, "medium": 1, "long": 2}
# Representative gap durations for log-gap x-axis (seconds), matching the
# default bins in configs/gap/viv_reid_dgra.yaml.
BIN_CENTER_SECONDS = {"short": 20.0, "medium": 90.0, "long": 600.0}


def degradation_curve(accs: Mapping[str, float]) -> list[tuple[str, float]]:
    """Sort per-bin accuracies short -> medium -> long (unknown bins dropped)."""
    known = [(name, accs[name]) for name in BIN_ORDER if name in accs]
    return sorted(known, key=lambda kv: BIN_ORDER[kv[0]])


def degradation_slope(accs: Mapping[str, float], bin_centers: Mapping[str, float] | None = None) -> float:
    """Linear-fit slope of accuracy vs log gap; requires >= 2 bins."""
    centers = bin_centers or BIN_CENTER_SECONDS
    curve = degradation_curve(accs)
    if len(curve) < 2:
        raise ValueError("degradation_slope requires at least 2 gap bins")
    x = np.log([centers[name] for name, _ in curve])
    y = np.array([acc for _, acc in curve], dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def chance_normalized(accuracy: float, pool_size: int) -> float:
    """(acc - 1/K) / (1 - 1/K); 0 = chance, 1 = perfect (spec section 9)."""
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")
    chance = 1.0 / pool_size
    if chance == 1.0:
        raise ValueError("pool_size 1 has no chance-normalization range")
    return (accuracy - chance) / (1.0 - chance)
