import pytest

from src.data.ais import AisPing, AisTrajectory, AisTrajectoryManifest, split_pings_by_window


def make_ping(utc_ms, mmsi="100000000", lon=114.3, lat=30.6):
    return AisPing(utc_ms=utc_ms, mmsi=mmsi, lon=lon, lat=lat)


def test_ping_validate():
    make_ping(0).validate()
    with pytest.raises(ValueError):
        make_ping(-1).validate()
    with pytest.raises(ValueError):
        make_ping(0, lon=200).validate()


def test_trajectory_requires_sorted():
    t = AisTrajectory(
        trajectory_id="t", vessel_id="v", sequence_id="s",
        pings=[make_ping(200), make_ping(100)],
    )
    with pytest.raises(ValueError):
        t.validate()


def test_manifest_roundtrip(tmp_path):
    traj = AisTrajectory(
        trajectory_id="t1", vessel_id="v1", sequence_id="s1",
        pings=[make_ping(100), make_ping(200)],
    )
    m = AisTrajectoryManifest([traj])
    p = tmp_path / "ais.jsonl"
    m.save(str(p))
    m2 = AisTrajectoryManifest.load(str(p))
    assert len(m2) == 1
    assert m2.get("v1").to_dict() == traj.to_dict()
    assert m2.get("missing") is None


def test_split_pings_by_window():
    pings = [make_ping(100), make_ping(200), make_ping(300), make_ping(400)]
    visible, withheld = split_pings_by_window(pings, 200, 400)
    assert [p.utc_ms for p in withheld] == [200, 300]
    assert [p.utc_ms for p in visible] == [100, 400]


def test_split_pings_dropout_and_jitter():
    pings = [make_ping(1000 + i * 100) for i in range(20)]
    visible, withheld = split_pings_by_window(
        pings, 2000, 3000, jitter_ms=50, dropout_p=0.5, seed=7
    )
    # withheld pings are untouched (hidden ground truth)
    assert all(2000 <= p.utc_ms < 3000 for p in withheld)
    assert all(p.utc_ms in range(2000, 3000) for p in withheld)
    # visible pings outside the window, jittered, possibly dropped
    assert len(visible) < len(pings) - len(withheld)
    a = split_pings_by_window(pings, 2000, 3000, jitter_ms=50, dropout_p=0.5, seed=7)
    b = split_pings_by_window(pings, 2000, 3000, jitter_ms=50, dropout_p=0.5, seed=7)
    assert [p.utc_ms for p in a[0]] == [p.utc_ms for p in b[0]]
