import numpy as np
import pytest

from evaluation.bootstrap import bootstrap_ci, paired_bootstrap_difference
from evaluation.degradation import (
    BIN_CENTER_SECONDS,
    chance_normalized,
    degradation_curve,
    degradation_slope,
)


def test_degradation_curve_order():
    assert degradation_curve({"long": 0.4, "short": 0.8, "medium": 0.6}) == [
        ("short", 0.8),
        ("medium", 0.6),
        ("long", 0.4),
    ]


def test_degradation_slope_known():
    # log-gap centers equally spaced -> slope is exact two-point fit
    centers = {"short": 1.0, "medium": np.e, "long": np.e ** 2}
    slope = degradation_slope(
        {"short": 0.8, "medium": 0.6, "long": 0.4}, bin_centers=centers
    )
    assert slope == pytest.approx(-0.2, abs=1e-9)


def test_degradation_slope_needs_two_bins():
    with pytest.raises(ValueError):
        degradation_slope({"short": 0.5})


def test_chance_normalized():
    assert chance_normalized(0.5, 10) == pytest.approx((0.5 - 0.1) / 0.9)
    assert chance_normalized(0.1, 10) == pytest.approx(0.0)  # chance -> 0
    assert chance_normalized(1.0, 10) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        chance_normalized(0.5, 0)


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, 200)
    r = bootstrap_ci(vals, n_samples=1000, seed=1)
    assert r["ci_low"] <= r["mean"] <= r["ci_high"]
    assert r["ci_low"] < 0 < r["ci_high"]


def test_paired_diff_excludes_zero():
    a = [1.0] * 50
    b = [0.0] * 50
    r = paired_bootstrap_difference(a, b, n_samples=1000, seed=1)
    assert r["mean_diff"] > 0
    assert r["ci_low"] > 0
    assert r["excludes_zero"]


def test_paired_diff_needs_equal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1.0, 2.0], [1.0])


def test_bin_centers_defined():
    assert set(BIN_CENTER_SECONDS) == {"short", "medium", "long"}
