"""Gap-degradation curves: accuracy vs gap-duration bin, per arm.

The headline signal is the degradation curve: chance-normalized accuracy per
gap bin plus the linear-fit slope of accuracy against log gap. A flatter
(less negative) slope means the arm degrades less as the disappearance gap
grows — the "advantage widens with gap" gate compares slopes across arms.

Bins are either the DGRA names (``short``/``medium``/``long``) or blackout
durations keyed by their labels (``10s``/``30s``/``60s``/``120s``). Numeric
labels are placed on a log-seconds axis; non-numeric labels fall back to
their bin index.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

BIN_ORDER = ["short", "medium", "long"]


@dataclass
class DegradationResult:
    arm_name: str
    bin_labels: list[str]            # ["short","medium","long"] or ["10s","30s","60s","120s"]
    accuracies: list[float]          # top-1 per bin
    chance_normalized: list[float]   # (acc - 1/K) / (1 - 1/K)
    slope: float                     # linear fit on log-gap axis
    auc: float                       # trapezoid
    ci_lower: float
    ci_upper: float


def _ordered_labels(results_by_bin: dict[str, list[bool]]) -> list[str]:
    """Order bins: named DGRA bins first by canonical order, otherwise by the
    numeric prefix of the label (ascending), non-numeric labels last."""
    labels = list(results_by_bin)
    if labels and all(l in BIN_ORDER for l in labels):
        return sorted(labels, key=lambda l: BIN_ORDER.index(l))

    def key(label: str):
        m = re.search(r"\d+(?:\.\d+)?", label)
        return (0, float(m.group())) if m else (1, label)

    return sorted(labels, key=key)


def _gap_axis(labels: list[str]) -> np.ndarray:
    """x axis for slope/AUC: log-seconds when every label carries a number,
    else plain bin index."""
    nums = []
    for l in labels:
        m = re.search(r"\d+(?:\.\d+)?", l)
        nums.append(float(m.group()) if m else None)
    if labels and all(n is not None for n in nums):
        return np.log(np.asarray(nums, dtype=float))
    return np.arange(len(labels), dtype=float)


def _chance_normalized(accuracy: float, pool_size: int) -> float:
    if pool_size < 2:
        raise ValueError("pool_size must be >= 2 for chance normalization")
    chance = 1.0 / pool_size
    return (accuracy - chance) / (1.0 - chance)


def compute_degradation(
    results_by_bin: dict[str, list[bool]],   # bin_label -> list of top1_correct bools
    pool_size: int,
    arm_name: str,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> DegradationResult:
    """Per-bin accuracy, chance normalization, log-linear slope, trapezoid
    AUC and a bootstrap CI on the mean chance-normalized accuracy.

    The slope is a linear fit of accuracy vs the gap axis (log-seconds when
    labels are numeric, bin index otherwise); a more negative slope means
    faster degradation. With fewer than two bins the slope is ``NaN``.
    """
    labels = _ordered_labels(results_by_bin)
    if not labels:
        raise ValueError("results_by_bin must contain at least one bin")
    accs = []
    for label in labels:
        hits = results_by_bin[label]
        accs.append(float(np.mean(hits)) if hits else 0.0)
    cns = [_chance_normalized(a, pool_size) for a in accs]

    x = _gap_axis(labels)
    slope = float(np.polyfit(x, np.asarray(accs, dtype=float), 1)[0]) if len(labels) >= 2 else float("nan")
    auc = float(_trapezoid(x, np.asarray(cns, dtype=float)))

    rng = np.random.default_rng(seed)
    per_bin = [np.asarray(results_by_bin[l], dtype=bool) for l in labels]
    sample_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        acc_sample = []
        for b in per_bin:
            if b.size == 0:
                acc_sample.append(0.0)
                continue
            idx = rng.integers(0, b.size, size=(b.size,))
            acc_sample.append(float(b[idx].mean()))
        cns_sample = [_chance_normalized(a, pool_size) for a in acc_sample]
        sample_means[i] = float(np.mean(cns_sample))
    lo, hi = np.percentile(sample_means, [2.5, 97.5])

    return DegradationResult(
        arm_name=arm_name,
        bin_labels=labels,
        accuracies=accs,
        chance_normalized=cns,
        slope=slope,
        auc=auc,
        ci_lower=float(lo),
        ci_upper=float(hi),
    )


def _trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal rule; works with any numpy version."""
    if x.shape[0] < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def compare_slopes(a: DegradationResult, b: DegradationResult) -> dict:
    """Compare degradation slopes: a higher (less negative) slope means less
    degradation. Returns ``{"slope_diff": float, "interpretation": str}``.
    """
    slope_diff = float(b.slope - a.slope)
    interpretation = "B_flatter" if b.slope > a.slope else "A_flatter"
    return {"slope_diff": slope_diff, "interpretation": interpretation}
