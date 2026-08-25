"""DGRA gap-trial construction tests (spec section 4.4 / 7.1).

Covers the GapTrial schema, deterministic synthetic-trial construction with
contiguous block hold-out, gap-duration binning, and seed determinism.
"""
import pytest

from data.gap_trials import (
    GapProtocolConfig,
    GapTrial,
    assign_gap_bin,
    build_gap_trials,
)
from data.manifest import TrackletManifest


def make_tracklet(tid="t1", vid="v1", cam="cam0", split="query", n_frames=30):
    return TrackletManifest(
        tracklet_id=tid,
        vessel_id=vid,
        camera_id=cam,
        split=split,
        frame_paths=[f"{tid}_{i:03d}.jpg" for i in range(n_frames)],
    )


def test_gap_trial_fields():
    t = GapTrial(
        trial_id="syn-t1-8-15",
        query_tracklet_id="t1",
        gallery_tracklet_id="t1",
        vessel_id="v1",
        gap_seconds=0.6,
        gap_duration_source="frame_count",
        gap_type="synthetic_within_tracklet",
        distractor_pool_id="p1",
        pool_size=5,
        ais_available_at_test=True,
        split="test",
        gap_bin="medium",
    )
    t.validate()  # must not raise
    assert t.trial_id == "syn-t1-8-15"
    assert t.vessel_id == "v1"
    assert t.gap_seconds == pytest.approx(0.6)
    assert t.gap_bin == "medium"
    assert t.gap_type == "synthetic_within_tracklet"
    assert t.pool_size == 5
    assert t.ais_available_at_test is True


def test_synthetic_trial_disjoint():
    # A 30-frame tracklet with default 0.5 holdout produces a synthetic trial
    # whose query prefix + removed block + gallery suffix tile the tracklet.
    cfg = GapProtocolConfig(ais_modes=["withheld"])  # one trial per tracklet
    n = 30
    tracklet = make_tracklet(n_frames=n)
    trials = build_gap_trials([tracklet], cfg)
    syn = [t for t in trials if t.gap_type == "synthetic_within_tracklet"]
    assert len(syn) == 1
    t = syn[0]
    assert t.query_tracklet_id == t.gallery_tracklet_id == "t1"
    assert t.vessel_id == "v1"

    # trial_id encodes the held-out block: syn-{tracklet_id}-{start}-{block_len}
    # with an optional -{ais_mode} suffix appended by _with_ais.
    parts = t.trial_id.split("-")
    assert parts[0] == "syn" and parts[1] == "t1"
    nums = [int(p) for p in parts if p.isdigit()]
    start, block_len = nums[-2], nums[-1]
    assert 1 <= start < start + block_len <= n - 1

    query = set(range(0, start))
    removed = set(range(start, start + block_len))
    gallery = set(range(start + block_len, n))
    # pairwise disjoint and covering all 30 frames
    assert query & removed == set()
    assert removed & gallery == set()
    assert query & gallery == set()
    assert query | removed | gallery == set(range(n))


def test_bin_assignment():
    cfg = GapProtocolConfig()
    # seconds bins are half-open [lo, hi): short [10, 30), medium [60, 120),
    # long [300, inf)
    assert assign_gap_bin(20.0, None, "timestamp", cfg) == "short"
    assert assign_gap_bin(90.0, None, "timestamp", cfg) == "medium"
    assert assign_gap_bin(400.0, None, "timestamp", cfg) == "long"
    # lower edge is included, upper edge of each bin is excluded
    assert assign_gap_bin(10.0, None, "timestamp", cfg) == "short"
    assert assign_gap_bin(30.0, None, "timestamp", cfg) is None
    assert assign_gap_bin(60.0, None, "timestamp", cfg) == "medium"
    assert assign_gap_bin(120.0, None, "timestamp", cfg) is None
    assert assign_gap_bin(300.0, None, "timestamp", cfg) == "long"
    # frame-count bins use their own thresholds
    assert assign_gap_bin(None, 50, "frame_count", cfg) == "short"
    assert assign_gap_bin(None, 200, "frame_count", cfg) == "medium"
    assert assign_gap_bin(None, 1000, "frame_count", cfg) == "long"


def test_determinism():
    cfg = GapProtocolConfig(seed=42, ais_modes=["withheld"])
    tracklets = [
        make_tracklet("t1", "v1", "cam0", "query"),
        make_tracklet("g2", "v1", "cam1", "gallery"),
        make_tracklet("t3", "v2", "cam0", "query"),
        make_tracklet("g4", "v2", "cam1", "gallery"),
    ]
    a = build_gap_trials(tracklets, cfg)
    b = build_gap_trials(tracklets, cfg)
    assert [t.trial_id for t in a] == [t.trial_id for t in b]
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]
    # a different seed may reorder trials but keeps the same trial set
    c = build_gap_trials(tracklets, GapProtocolConfig(seed=43, ais_modes=["withheld"]))
    assert sorted(t.trial_id for t in a) == sorted(t.trial_id for t in c)
