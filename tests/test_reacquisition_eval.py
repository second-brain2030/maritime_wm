import numpy as np
import pytest

from data.ais import AisPing
from evaluation.blackout_harness import BlackoutConfig, BlackoutEpisode, build_blackout_episodes
from evaluation.reacquisition_eval import (
    ais_drift_m,
    episode_result,
    pixel_drift_m,
    predict_bbox_center,
    predict_lonlat_from_pings,
    rank_by_cosine,
)
from data.manifest import TrackletManifest


def test_rank_by_cosine():
    q = np.array([1.0, 0.0])
    cands = {"a": np.array([0.99, 0.1]), "b": np.array([0.0, 1.0]), "c": np.array([1.0, 0.0])}
    assert rank_by_cosine(q, cands) == ["c", "a", "b"]


def make_episode(vessel_id="v1"):
    return BlackoutEpisode(
        episode_id="e1", sequence_id="S", vessel_id=vessel_id, split="query",
        blackout_duration_s=10.0, blackout_start_frame=50, blackout_start_utc_ms=5000,
        reappearance_frame=150, query_frame_indices=[0, 25, 50],
        candidate_vessel_ids=["v1", "v2"], gt_bbox_at_reappearance=[0, 0, 10, 10],
        gt_lonlat_at_reappearance=[114.0, 30.0],
    )


def test_episode_result_ranks():
    ep = make_episode()
    r = episode_result(ep, ["v2", "v1"], drift_m=12.5)
    assert r["rank_of_correct"] == 2
    assert r["duration_s"] == 10.0
    assert r["drift_m"] == 12.5
    r2 = episode_result(ep, ["v2"], drift_m=None)
    assert r2["rank_of_correct"] is None


def test_predict_lonlat_from_pings():
    pings = [
        AisPing(utc_ms=0, mmsi="1", lon=0.0, lat=0.0),
        AisPing(utc_ms=1000, mmsi="1", lon=0.001, lat=0.0),
    ]
    pred = predict_lonlat_from_pings(pings, gap_s=10.0)
    assert pred is not None
    assert pred[0] == pytest.approx(0.011)  # 0.001 deg/s * 10s + 0.001
    assert predict_lonlat_from_pings(pings[:1], 10.0) is None


def test_ais_drift_m():
    assert ais_drift_m((0.0, 0.0), [0.0, 1.0]) == pytest.approx(111194.9, rel=1e-3)
    assert ais_drift_m(None, [0.0, 1.0]) is None


def test_predict_bbox_center():
    obs = [(0.0, [0.0, 0.0, 10.0, 10.0]), (1.0, [10.0, 0.0, 10.0, 10.0])]
    pred = predict_bbox_center(obs, gap_s=5.0)
    # centers 5 -> 15 at vx=10; +5s -> 65
    assert pred is not None
    assert pred[0] == pytest.approx(65.0)
    assert predict_bbox_center(obs[:1], 5.0) is None


def test_pixel_drift_m():
    pred = np.array([5.0, 5.0])
    assert pixel_drift_m(pred, [0.0, 0.0, 10.0, 10.0]) == pytest.approx(0.0)
    assert pixel_drift_m(None, [0, 0, 10, 10]) is None


def test_end_to_end_target_ranks_first():
    # target moves at constant velocity; candidates far away -> dead-reckon finds it
    def make(vid, x):
        n = 200
        idx = list(range(n))
        return TrackletManifest(
            tracklet_id=f"t_{vid}", vessel_id=vid, camera_id="S", split="query",
            frame_paths=[f"v.mp4#{i}" for i in idx], fps=10.0, video_path="v.mp4",
            frame_indices=idx,
            frame_timestamps_utc_ms=[i * 100 for i in idx],
            frame_bboxes=[[x + i, 10.0, 20.0, 20.0] for i in range(n)],
            source_dataset="fvessel",
        )

    target = make("v1", 0.0)
    c2, c3 = make("v2", 500.0), make("v3", -500.0)
    cfg = BlackoutConfig(durations_s=[1.0], min_visible_before_s=3.0, pool_size=5, seed=1)
    ep = build_blackout_episodes([target, c2, c3], config=cfg)[0]
    from evaluation.baselines import deadreckon_rank
    ranked, drift = deadreckon_rank(
        ep, target, {"v1": target, "v2": c2, "v3": c3}
    )
    assert ranked[0] == "v1"
    assert drift is not None and drift < 10.0
