import pytest

from data.manifest import TrackletManifest, load_manifests, save_manifests


def make(**kw):
    base = dict(
        tracklet_id="t1",
        vessel_id="v1",
        camera_id="cam0",
        split="train",
        frame_paths=["a.jpg", "b.jpg"],
    )
    base.update(kw)
    return TrackletManifest(**base)


def test_valid_manifest_ok():
    make().validate()


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        make(split="val").validate()


def test_invalid_occlusion_raises():
    with pytest.raises(ValueError):
        make(occlusion_level="heavy").validate()


def test_missing_frames_raises():
    with pytest.raises(ValueError):
        make(frame_paths=[]).validate()


def test_fingerprint_deterministic_and_stable():
    a = make().fingerprint()
    b = make().fingerprint()
    assert a == b
    assert len(a) == 64


def test_fingerprint_sensitive_to_fields():
    assert make().fingerprint() != make(quality_score=0.5).fingerprint()


def test_roundtrip(tmp_path):
    m = make(quality_score=0.9, source_dataset="viv_reid")
    p = tmp_path / "m.jsonl"
    save_manifests(str(p), [m])
    loaded = load_manifests(str(p))
    assert len(loaded) == 1
    assert loaded[0].to_dict() == m.to_dict()


def test_from_dict_ignores_unknown_keys():
    m = TrackletManifest.from_dict({**make().to_dict(), "extra": 1})
    assert m.tracklet_id == "t1"
