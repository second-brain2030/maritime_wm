import pytest

from data.ais import AisPing, AisTrajectory
from data.manifest import TrackletManifest
from evaluation.blackout_harness import (
    BlackoutConfig,
    BlackoutEpisodeManifest,
    build_blackout_episodes,
)


def make_tracklet(vessel_id="v1", seq="Video-01", n_frames=200, fps=10.0, start_utc=0):
    idx = list(range(n_frames))
    return TrackletManifest(
        tracklet_id=f"fv_{seq}_t0",
        vessel_id=vessel_id,
        camera_id=seq,
        split="query",
        frame_paths=[f"v.mp4#{i}" for i in idx],
        fps=fps,
        video_path="v.mp4",
        frame_indices=idx,
        frame_timestamps_utc_ms=[start_utc + int(i / fps * 1000) for i in idx],
        frame_bboxes=[[10.0, 10.0, 50.0, 50.0]] * n_frames,
        source_dataset="fvessel",
    )


def make_ais(vessel_id, n_pings=50, start_utc=0, step_ms=500, lon=114.0, lat=30.0):
    return AisTrajectory(
        trajectory_id=f"t_{vessel_id}",
        vessel_id=vessel_id,
        sequence_id="Video-01",
        pings=[AisPing(utc_ms=start_utc + i * step_ms, mmsi=vessel_id, lon=lon, lat=lat) for i in range(n_pings)],
    )


def test_episodes_built_for_each_duration(tmp_path):
    tracklet = make_tracklet()
    cfg = BlackoutConfig(
        durations_s=[1.0, 2.0],
        min_visible_before_s=3.0,
        min_visible_after_s=1.0,
        withhold_ais=False,
        pool_size=5,
        seed=42,
    )
    episodes = build_blackout_episodes([tracklet], config=cfg)
    assert len(episodes) == 2
    d1 = next(e for e in episodes if e.blackout_duration_s == 1.0)
    # fps 10, gap 10 frames; first window with >=30 query frames -> b=30, r=40
    assert d1.blackout_start_frame == 30
    assert d1.reappearance_frame == 40
    assert d1.query_frame_indices == list(range(30))
    assert d1.candidate_vessel_ids == ["v1"]  # degenerate pool (only vessel)
    assert d1.gt_bbox_at_reappearance == [10.0, 10.0, 50.0, 50.0]
    assert d1.split == "query"


def test_ais_withheld_and_gt_lonlat():
    tracklet = make_tracklet(start_utc=0)
    ais = make_ais("v1", start_utc=0, step_ms=100)
    cfg = BlackoutConfig(
        durations_s=[1.0],
        min_visible_before_s=3.0,
        min_visible_after_s=1.0,
        withhold_ais=True,
        pool_size=5,
        seed=42,
    )
    episodes = build_blackout_episodes([tracklet], {"v1": ais}, config=cfg)
    e = episodes[0]
    # blackout window [3000, 4000) ms; pings every 100ms -> 10 withheld
    assert e.withheld_ping_count == 10
    assert e.gt_lonlat_at_reappearance == pytest.approx([114.0, 30.0])


def test_deterministic():
    tracklet = make_tracklet()
    cfg = BlackoutConfig(durations_s=[1.0, 2.0, 3.0], seed=42)
    a = build_blackout_episodes([tracklet], config=cfg)
    b = build_blackout_episodes([tracklet], config=cfg)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]


def test_insufficient_visibility_no_episodes():
    tracklet = make_tracklet(n_frames=20)  # 2s total, less than min_before(3s)+gap
    cfg = BlackoutConfig(durations_s=[10.0], min_visible_before_s=3.0, seed=1)
    assert build_blackout_episodes([tracklet], config=cfg) == []


def test_manifest_roundtrip(tmp_path):
    tracklet = make_tracklet()
    cfg = BlackoutConfig(durations_s=[1.0], seed=1)
    episodes = build_blackout_episodes([tracklet], config=cfg)
    m = BlackoutEpisodeManifest(episodes)
    p = tmp_path / "episodes.jsonl"
    m.save(str(p))
    m2 = BlackoutEpisodeManifest.load(str(p))
    assert len(m2) == len(episodes)
    s = m2.summary()
    assert s["n_episodes"] == 1
    assert s["by_duration_s"] == {1.0: 1}


def test_sparse_observations_time_based_spans():
    # 1 Hz annotations on 25 fps video (FVessel GT style): before/after spans
    # are computed in TIME, not observation count.
    idx = [s * 25 for s in range(120)]  # seconds 0..119
    tracklet = TrackletManifest(
        tracklet_id="fv_Video-01_t0",
        vessel_id="v1",
        camera_id="Video-01",
        split="query",
        frame_paths=[f"v.mp4#{i}" for i in idx],
        fps=25.0,
        video_path="v.mp4",
        frame_indices=idx,
        frame_timestamps_utc_ms=[i * 40 for i in idx],
        frame_bboxes=[[10.0, 10.0, 50.0, 50.0]] * len(idx),
        source_dataset="fvessel",
    )
    cfg = BlackoutConfig(
        durations_s=[10.0, 30.0],
        min_visible_before_s=5.0,
        min_visible_after_s=1.0,
        seed=42,
    )
    episodes = build_blackout_episodes([tracklet], config=cfg)
    assert len(episodes) == 2
    d10 = next(e for e in episodes if e.blackout_duration_s == 10.0)
    assert d10.blackout_start_frame == 5 * 25  # earliest window with >=5s before
    assert d10.reappearance_frame == 15 * 25


def test_co_present_candidates():
    target = make_tracklet(vessel_id="v1", n_frames=200)
    other = make_tracklet(vessel_id="v2", seq="Video-01", n_frames=200)
    cfg = BlackoutConfig(durations_s=[1.0], min_visible_before_s=3.0, pool_size=5, seed=1)
    episodes = build_blackout_episodes([target, other], config=cfg)
    e = episodes[0]
    assert "v1" in e.candidate_vessel_ids
    assert "v2" in e.candidate_vessel_ids  # co-present at reappearance frame
    assert len(e.candidate_vessel_ids) == 2


def test_config_validation():
    with pytest.raises(ValueError):
        BlackoutConfig(durations_s=[0.0]).validate()
    with pytest.raises(ValueError):
        BlackoutConfig(ais_dropout_p=1.5).validate()
