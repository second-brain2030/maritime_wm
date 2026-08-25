"""Bootstrap (identity-level CI and paired arm-difference) tests.

Covers bootstrap_ci and bootstrap_arm_diff from src.evaluation.bootstrap.
The legacy score-map degradation APIs (degradation_curve, degradation_slope,
chance_normalized, BIN_CENTER_SECONDS) were removed; degradation-curve tests
live in test_degradation.py.
"""
import pytest

from src.evaluation.bootstrap import bootstrap_arm_diff, bootstrap_ci


def test_bootstrap_ci_mean_and_interval():
    scores = {"v1": 0.5, "v2": 1.0, "v3": 1.5}
    mean, lo, hi = bootstrap_ci(scores, n_samples=200, seed=1)
    assert mean == pytest.approx(1.0)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_ci_deterministic():
    scores = {"v1": 0.1, "v2": 0.9, "v3": 0.4, "v4": 0.7}
    a = bootstrap_ci(scores, n_samples=100, seed=42)
    b = bootstrap_ci(scores, n_samples=100, seed=42)
    assert a == b


def test_bootstrap_ci_requires_scores():
    with pytest.raises(ValueError):
        bootstrap_ci({})


def test_bootstrap_arm_diff_constant():
    # all shared identities differ by exactly -1.0 -> CI excludes zero
    a = {"v1": 1.0, "v2": 1.0, "v3": 1.0}
    b = {"v1": 0.0, "v2": 0.0, "v3": 0.0}
    mean_diff, lo, hi = bootstrap_arm_diff(a, b, n_samples=100, seed=1)
    assert mean_diff == pytest.approx(-1.0)
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(-1.0)
    assert hi < 0  # entirely negative -> B significantly worse than A


def test_bootstrap_arm_diff_requires_shared_identities():
    with pytest.raises(ValueError):
        bootstrap_arm_diff({"v1": 0.5}, {"v2": 0.5})
