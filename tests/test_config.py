import json

import pytest

from utils.config import deep_merge, load_config


def test_deep_merge():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    over = {"b": {"y": 3}, "c": 4}
    out = deep_merge(base, over)
    assert out == {"a": 1, "b": {"x": 1, "y": 3}, "c": 4}


def test_deep_merge_scalar_overwrites():
    out = deep_merge({"a": {"x": 1}}, {"a": 5})
    assert out == {"a": 5}


def test_load_experiment_resolves_extends():
    cfg = load_config("experiments", "vivreid_vjepa_encoder")
    assert cfg["experiment"]["name"] == "vivreid_vjepa_encoder"
    assert cfg["training"]["epochs"] == 60  # from _base
    assert cfg["model"]["arm"] == "vjepa_encoder"
    assert cfg["evaluation"]["bootstrap_samples"] == 2000


def test_load_gap_config():
    cfg = load_config("gap", "viv_reid_dgra")
    proto = cfg["gap_protocol"]
    assert proto["pool_sizes"] == [5, 10, 20]
    assert proto["long" if False else "gap_bins_seconds"]["long"] == [300, None]


def test_load_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_config("experiments", "does_not_exist")
