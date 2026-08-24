import pytest

from data.intermittent import block_patch_mask, find_gaps, intermittent_observation_mask


def test_mask_length_and_determinism():
    a = intermittent_observation_mask(100, 5, 5, seed=1)
    b = intermittent_observation_mask(100, 5, 5, seed=1)
    assert len(a) == 100
    assert a == b
    c = intermittent_observation_mask(100, 5, 5, seed=2)
    assert a != c  # different seeds -> different phase


def test_mask_observed_fraction():
    m = intermittent_observation_mask(1000, 5, 5, seed=1)
    frac = sum(m) / len(m)
    assert 0.4 <= frac <= 0.6


def test_find_gaps_consistent_with_mask():
    m = intermittent_observation_mask(1000, 5, 5, seed=1)
    gaps = find_gaps(m, min_gap_frames=1)
    assert gaps
    for s, e in gaps:
        assert e > s
        assert all(not m[i] for i in range(s, e))
    # total masked equals union of gaps
    masked = sum(1 for v in m if not v)
    assert sum(e - s for s, e in gaps) == masked


def test_find_gaps_all_observed():
    m = [True] * 50
    assert find_gaps(m) == []


def test_find_gaps_min_length():
    m = [True, False, True, False, False, True]
    assert find_gaps(m, min_gap_frames=2) == [(3, 5)]
    assert find_gaps(m, min_gap_frames=1) == [(1, 2), (3, 5)]


def test_block_patch_mask():
    m = block_patch_mask(100, 100, 0.25, seed=3)
    assert m.shape == (100, 100)
    frac = float(m.mean())
    assert 0.15 <= frac <= 0.35
    assert (block_patch_mask(100, 100, 0.25, seed=3) == m).all()


def test_block_patch_mask_validation():
    with pytest.raises(ValueError):
        block_patch_mask(100, 100, 1.5, seed=1)
    with pytest.raises(ValueError):
        block_patch_mask(100, 100, 0.0, seed=1)
