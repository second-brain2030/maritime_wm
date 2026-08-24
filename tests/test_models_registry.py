import pytest

from models import encoder_registry
from models.baselines import AisUpperBound, KalmanDeadReckoning, TrackerReidBaseline
from models.baselines.ais_upper_bound import AisUpperBound as _AUB
from data.gap_trials import GapTrial


def test_encoder_registry_has_arms():
    assert encoder_registry.names() == ["cnn_reid", "openvla_vision", "vjepa_encoder", "vjepa_predictor"]


def test_encoder_registry_create():
    enc = encoder_registry.create("cnn_reid", backbone="osnet_x1_0")
    assert enc.name == "cnn_reid_osnet_x1_0"
    assert enc.embedding_dim == 2048


def test_cnn_encode_fails_loudly():
    enc = encoder_registry.create("cnn_reid")
    with pytest.raises(NotImplementedError):
        enc.encode_observed(None, None)  # type: ignore[arg-type]


def test_predictor_arm_is_not_silent():
    enc = encoder_registry.create("vjepa_predictor")
    assert enc.predictor_horizon_delta == 60.0
    with pytest.raises(NotImplementedError):
        enc.encode_predicted(None, None)  # type: ignore[arg-type]


def test_openvla_feature_source_validated():
    with pytest.raises(ValueError):
        encoder_registry.create("openvla_vision", feature_source="nope")


def test_ais_upper_bound_requires_ais():
    baseline = _AUB()
    trial = GapTrial(
        trial_id="x", query_tracklet_id="a", gallery_tracklet_id="b",
        vessel_id="v", ais_available_at_test=False,
    )
    with pytest.raises(ValueError):
        baseline.rank(trial, {})


def test_baselines_instantiate():
    assert KalmanDeadReckoning().name == "kalman_deadreckon"
    assert TrackerReidBaseline().name == "tracker_reid"
    assert AisUpperBound().name == "ais_upper_bound"
