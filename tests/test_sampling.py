import pytest

from src.data.sampling import MODES, sample_frame_indices


def test_uniform_length_and_order():
    idx = sample_frame_indices(100, 16, mode="uniform")
    assert len(idx) == 16
    assert idx == sorted(idx)


def test_uniform_deterministic():
    a = sample_frame_indices(100, 16, mode="uniform", seed=1)
    b = sample_frame_indices(100, 16, mode="uniform", seed=1)
    assert a == b


def test_recent_last_window():
    assert sample_frame_indices(100, 16, mode="recent") == list(range(84, 100))


def test_sparse_every_k():
    assert sample_frame_indices(100, 16, mode="sparse", k=7) == list(range(0, 100, 7))


def test_prefix_only_fraction():
    # prefix is capped at frames_per_tracklet (fixed-length contract)
    assert sample_frame_indices(100, 16, mode="prefix_only", fraction=0.25) == list(range(16))


def test_prefix_only_short_pads():
    idx = sample_frame_indices(100, 16, mode="prefix_only", fraction=0.05)
    assert len(idx) == 16
    assert idx[:5] == list(range(5))


def test_sparse_keeps_variable_length():
    # sparse is the variable-length regime: no padding to frames_per_tracklet
    idx = sample_frame_indices(100, 16, mode="sparse", k=7)
    assert idx == list(range(0, 100, 7))
    assert len(idx) == 15  # not padded to 16


def test_short_repeat_last():
    idx = sample_frame_indices(3, 8, mode="uniform", short_policy="repeat_last")
    assert len(idx) == 8
    assert idx[-1] == 2


def test_short_repeat_first():
    idx = sample_frame_indices(3, 8, mode="uniform", short_policy="repeat_first")
    assert len(idx) == 8
    assert idx[0] == 0


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        sample_frame_indices(10, 4, mode="nope")


def test_bad_args_raise():
    with pytest.raises(ValueError):
        sample_frame_indices(0, 4)
    with pytest.raises(ValueError):
        sample_frame_indices(10, 4, k=0)
    with pytest.raises(ValueError):
        sample_frame_indices(10, 4, fraction=0.0)


def test_modes_defined():
    assert set(MODES) == {"uniform", "recent", "sparse", "prefix_only"}
