import pytest

from src.data.gap_trials import (
    GapProtocolConfig,
    GapTrial,
    GapTrialManifest,
    assign_gap_bin,
    build_gap_trials,
)
from src.data.manifest import TrackletManifest


def make_tracklet(tid, vid, cam, split, n_frames=10, t_start=None, t_end=None):
    return TrackletManifest(
        tracklet_id=tid,
        vessel_id=vid,
        camera_id=cam,
        split=split,
        frame_paths=[f"{tid}_{i}.jpg" for i in range(n_frames)],
        timestamp_start=t_start,
        timestamp_end=t_end,
    )


def test_bin_timestamp():
    cfg = GapProtocolConfig()
    assert assign_gap_bin(15.0, None, "timestamp", cfg) == "short"
    assert assign_gap_bin(90.0, None, "timestamp", cfg) == "medium"
    assert assign_gap_bin(600.0, None, "timestamp", cfg) == "long"
    assert assign_gap_bin(5.0, None, "timestamp", cfg) is None


def test_bin_frame_count():
    cfg = GapProtocolConfig()
    assert assign_gap_bin(None, 50, "frame_count", cfg) == "short"
    assert assign_gap_bin(None, 200, "frame_count", cfg) == "medium"
    assert assign_gap_bin(None, 1000, "frame_count", cfg) == "long"


def test_bin_unknown():
    cfg = GapProtocolConfig()
    assert assign_gap_bin(None, None, "unknown", cfg) is None


def test_synthetic_trials_deterministic():
    cfg = GapProtocolConfig(synthetic_gap_holdout=0.5, nominal_fps=25.0, ais_modes=["withheld"])
    tracklets = [
        make_tracklet("q1", "v1", "cam0", "query", n_frames=20),
        make_tracklet("g2", "v1", "cam1", "gallery", n_frames=20),
        make_tracklet("q3", "v2", "cam0", "query", n_frames=20),
    ]
    a = build_gap_trials(tracklets, cfg)
    b = build_gap_trials(tracklets, cfg)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]
    syn = [t for t in a if t.gap_type == "synthetic_within_tracklet"]
    assert syn
    assert all(t.gap_duration_source == "frame_count" for t in syn)
    assert all(t.gap_seconds == pytest.approx(10.0 / 25.0) for t in syn)


def test_natural_trial_gap_seconds():
    cfg = GapProtocolConfig(ais_modes=["withheld"])
    tracklets = [
        make_tracklet(
            "q1", "v1", "cam0", "query",
            t_start="2024-01-01T00:00:00Z", t_end="2024-01-01T00:01:00Z",
        ),
        make_tracklet(
            "g1", "v1", "cam1", "gallery",
            t_start="2024-01-01T00:02:00Z", t_end="2024-01-01T00:03:00Z",
        ),
    ]
    trials = build_gap_trials(tracklets, cfg)
    nat = [t for t in trials if t.gap_type == "natural_cross_camera"]
    assert len(nat) == 1
    assert nat[0].gap_seconds == pytest.approx(60.0)
    assert nat[0].gap_duration_source == "timestamp"
    assert nat[0].gap_bin == "medium"


def test_same_camera_natural_pairs_excluded():
    cfg = GapProtocolConfig(ais_modes=["withheld"])
    tracklets = [
        make_tracklet("q1", "v1", "cam0", "query"),
        make_tracklet("g1", "v1", "cam0", "gallery"),
    ]
    trials = build_gap_trials(tracklets, cfg)
    assert not [t for t in trials if t.gap_type == "natural_cross_camera"]


def test_validation_errors():
    with pytest.raises(ValueError):
        GapTrial(
            trial_id="x", query_tracklet_id="a", gallery_tracklet_id="b",
            vessel_id="v", gap_duration_source="timestamp", gap_seconds=None,
        ).validate()
    with pytest.raises(ValueError):
        GapTrial(
            trial_id="x", query_tracklet_id="a", gallery_tracklet_id="b",
            vessel_id="v", pool_size=5, distractor_pool_id=None,
        ).validate()


def test_manifest_roundtrip(tmp_path):
    t = GapTrial(
        trial_id="x", query_tracklet_id="a", gallery_tracklet_id="b",
        vessel_id="v", gap_duration_source="unknown",
    )
    m = GapTrialManifest([t])
    p = tmp_path / "trials.jsonl"
    m.save(str(p))
    m2 = GapTrialManifest.load(str(p))
    assert len(m2) == 1
    assert m2.summary()["n_trials"] == 1


def test_config_validation():
    with pytest.raises(ValueError):
        GapProtocolConfig(synthetic_gap_holdout=1.5).validate()
    with pytest.raises(ValueError):
        GapProtocolConfig(pool_sizes=[]).validate()
