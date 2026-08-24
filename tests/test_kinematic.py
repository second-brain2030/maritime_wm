import numpy as np
import pytest

from models.kinematic import (
    ConstantVelocityKalman,
    predict_across_gap,
    propose_search_window,
)


def test_kalman_predict_constant_velocity():
    kf = ConstantVelocityKalman()
    kf.init(x=0.0, y=0.0, vx=1.0, vy=0.0)
    pred = kf.predict(dt=10.0)
    assert np.allclose(pred, [10.0, 0.0], atol=1e-6)


def test_kalman_uncertainty_grows():
    kf = ConstantVelocityKalman(process_noise=0.1)
    kf.init(x=0.0, y=0.0, vx=0.5, vy=0.0)
    kf.predict(dt=1.0)
    r1 = kf.uncertainty_radius()
    kf.predict(dt=10.0)
    r2 = kf.uncertainty_radius()
    assert r2 > r1


def test_kalman_update_pulls_toward_measurement():
    kf = ConstantVelocityKalman(measurement_noise=1e-6)
    kf.init(x=0.0, y=0.0, vx=0.0, vy=0.0)
    kf.update(x=10.0, y=0.0)
    assert abs(float(kf.state[0]) - 10.0) < 1e-3


def test_kalman_not_initialized_raises():
    kf = ConstantVelocityKalman()
    with pytest.raises(RuntimeError):
        kf.predict(1.0)


def test_predict_across_gap():
    obs = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
    pred, radius = predict_across_gap(obs, gap_s=10.0)
    assert np.allclose(pred, [11.0, 0.0])
    assert radius > 0.0


def test_predict_across_gap_needs_two():
    with pytest.raises(ValueError):
        predict_across_gap([(0.0, 0.0, 0.0)], 10.0)


def test_search_window_gates_candidates():
    predicted = np.array([0.0, 0.0])
    candidates = [
        ("a", np.array([1.0, 1.0])),
        ("b", np.array([100.0, 100.0])),
    ]
    assert propose_search_window(predicted, 5.0, candidates) == ["a"]
    assert propose_search_window(predicted, 0.5, candidates) == []


def test_search_window_factor():
    predicted = np.array([0.0, 0.0])
    candidates = [("a", np.array([3.0, 0.0]))]
    assert propose_search_window(predicted, 2.0, candidates, window_factor=2.0) == ["a"]
    assert propose_search_window(predicted, 2.0, candidates, window_factor=1.0) == []
