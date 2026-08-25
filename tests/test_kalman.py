"""Arm F: motion-only Kalman dead-reckoning baseline tests.

rank() is pure numpy (centroid + velocity * gap extrapolation); predict_position()
fits a constant-velocity Kalman filter and needs the optional filterpy
dependency, so it is skipped when filterpy is unavailable.
"""
import numpy as np
import pytest

from src.models.baselines.kalman_deadreckon import KalmanDeadReckon


def test_rank_nearest_first():
    kf = KalmanDeadReckon()
    ranking = kf.rank(
        query_centroid=np.array([0.0, 0.0]),
        query_velocity=np.array([1.0, 0.0]),
        gap_seconds=5.0,
        gallery_centroids={
            "near": np.array([5.0, 0.0]),    # exactly at the extrapolated point
            "far": np.array([100.0, 100.0]),
        },
    )
    assert ranking == ["near", "far"]


def test_rank_pure_extrapolation():
    kf = KalmanDeadReckon()
    centroid = np.array([10.0, -5.0])
    velocity = np.array([2.0, 0.5])
    gap = 10.0
    predicted = centroid + velocity * gap  # [30.0, 0.0]
    gallery = {
        "exact": predicted.copy(),
        "off": predicted + np.array([40.0, 0.0]),
    }
    ranking = kf.rank(centroid, velocity, gap, gallery)
    assert ranking[0] == "exact"
    assert np.allclose(predicted, centroid + velocity * gap)


def test_predict_position_shape():
    pytest.importorskip("filterpy")  # optional dependency
    kf = KalmanDeadReckon()
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    )
    xy, std = kf.predict_position(positions, gap_steps=5)
    assert xy.shape == (2,)
    assert std.shape == (2,)
    assert np.isfinite(xy).all() and np.isfinite(std).all()
    # constant-velocity track: the prediction keeps moving forward along x
    assert xy[0] > positions[-1, 0]


def test_deterministic():
    kf = KalmanDeadReckon()
    args = (
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        5.0,
        {
            "a": np.array([5.0, 0.0]),
            "b": np.array([10.0, 0.0]),
            "c": np.array([1.0, 1.0]),
        },
    )
    assert kf.rank(*args) == kf.rank(*args)
