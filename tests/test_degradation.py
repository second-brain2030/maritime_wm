"""Gap-degradation curve tests (spec: degradation vs gap bin per arm).

compute_degradation takes per-bin top1-correct booleans and returns per-bin
accuracy, chance-normalized accuracy, a log-linear slope and a trapezoid AUC.
"""
import pytest

from evaluation.degradation import compare_slopes, compute_degradation


def degrade(accuracies, arm_name="A", pool_size=10):
    """Build per-bin bool lists with the requested mean accuracy."""
    results = {}
    for label, acc in zip(["short", "medium", "long"], accuracies):
        n = 10
        n_hits = round(acc * n)
        results[label] = [True] * n_hits + [False] * (n - n_hits)
    return compute_degradation(results, pool_size=pool_size, arm_name=arm_name)


def test_slope_direction():
    res = degrade([0.8, 0.6, 0.3])
    assert res.accuracies == pytest.approx([0.8, 0.6, 0.3])
    assert res.slope < 0  # accuracy drops across bins -> degradation


def test_chance_normalized_at_chance():
    # accuracy == 1/pool_size means the arm performs at chance -> 0.0
    pool_size = 10
    res = compute_degradation(
        {"short": [True] + [False] * 9},
        pool_size=pool_size,
        arm_name="A",
    )
    assert res.accuracies[0] == pytest.approx(1.0 / pool_size)
    assert res.chance_normalized[0] == pytest.approx(0.0, abs=1e-9)


def test_chance_normalized_perfect():
    res = compute_degradation(
        {"short": [True] * 10},
        pool_size=10,
        arm_name="A",
    )
    assert res.accuracies[0] == pytest.approx(1.0)
    assert res.chance_normalized[0] == pytest.approx(1.0, abs=1e-9)


def test_auc_monotone():
    # monotonically decreasing accuracy -> AUC below the trivial upper bound
    # (num_bins * max_acc) and below the AUC of a flat perfect curve
    dec = degrade([1.0, 0.5, 0.1], arm_name="dec")
    flat = degrade([1.0, 1.0, 1.0], arm_name="flat")
    assert dec.auc < len(dec.bin_labels) * max(dec.accuracies)
    assert dec.auc < flat.auc


def test_compare_slopes_label():
    a = degrade([0.8, 0.7, 0.6], arm_name="A")  # slope -0.1 (flatter)
    b = degrade([0.8, 0.5, 0.2], arm_name="B")  # slope -0.3 (steeper)
    assert a.slope == pytest.approx(-0.1, abs=1e-9)
    assert b.slope == pytest.approx(-0.3, abs=1e-9)
    out = compare_slopes(a, b)
    # higher (less negative) slope degrades less -> A is labelled flatter
    assert out["interpretation"] == "A_flatter"
    assert out["slope_diff"] == pytest.approx(b.slope - a.slope)
