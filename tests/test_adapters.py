"""Dataset adapter tests (spec section 16: adapter produces stable IDs and
no overlap violations; query/gallery manifest schema validation)."""
import pytest

from data.adapters import adapter_registry, get_adapter
from data.adapters.viv_reid import ViVReidAdapter
from data.manifest import TrackletManifest, load_manifests, save_manifests
from data.splits import identity_sets, validate_identity_disjointness

LAYOUT = {
    # Spec §12: train identities must be disjoint from query/gallery;
    # query and gallery may share identities (official Re-ID protocol).
    "train": ["v001_cam1_t1", "v001_cam1_t2", "v002_cam2_t1"],
    "query": ["v003_cam3_t1"],
    "gallery": ["v003_cam3_t2", "v004_cam4_t1"],
}

GOOD_CONFIG = {
    "layout": {
        "tracklet_identity_pattern": r"^(?P<vessel_id>v\d+)_.*$",
        "camera_pattern": r"cam(?P<camera_id>\d+)",
    }
}


def _make_raw_layout(tmp_path, layout=None):
    root = tmp_path / "viv-reid"
    for split, dirs in (layout or LAYOUT).items():
        for d in dirs:
            td = root / split / d
            td.mkdir(parents=True)
            for i in range(5):
                (td / f"frame_{i:04d}.jpg").touch()
    return root


def test_adapter_registered():
    assert "viv_reid" in adapter_registry


def test_get_adapter_via_registry(tmp_path):
    root = _make_raw_layout(tmp_path)
    adapter = get_adapter("viv_reid", {"root": str(root), **GOOD_CONFIG})
    assert isinstance(adapter, ViVReidAdapter)
    ms = adapter.build_manifests()
    assert len(ms) == 6  # 3 train + 1 query + 2 gallery
    for m in ms:
        m.validate()


def test_build_manifests_parses_ids(tmp_path):
    root = _make_raw_layout(tmp_path)
    ms = ViVReidAdapter({"root": str(root), **GOOD_CONFIG}).build_manifests()
    assert {m.split for m in ms} == {"train", "query", "gallery"}
    v001 = [m for m in ms if m.vessel_id == "v001"]
    assert len(v001) == 2  # both in train
    assert all(m.camera_id == "1" for m in v001)
    assert all(len(m.frame_paths) == 5 for m in ms)
    assert all(m.source_dataset == "viv_reid" for m in ms)
    # stable, unique tracklet ids
    ids = [m.tracklet_id for m in ms]
    assert len(ids) == len(set(ids))
    # deterministic ordering
    assert [m.tracklet_id for m in ms] == [m.tracklet_id for m in ViVReidAdapter(
        {"root": str(root), **GOOD_CONFIG}
    ).build_manifests()]


def test_missing_split_dir_raises(tmp_path):
    root = tmp_path / "viv-reid"
    (root / "train").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        ViVReidAdapter({"root": str(root)}).build_manifests()


def test_empty_root_raises(tmp_path):
    root = tmp_path / "viv-reid"
    root.mkdir()
    # no split dirs at all -> the first missing split dir fails loudly
    with pytest.raises(FileNotFoundError):
        ViVReidAdapter({"root": str(root)}).build_manifests()


def test_unparseable_identity_raises(tmp_path):
    root = _make_raw_layout(tmp_path)
    bad = root / "train" / "unparseable_thing"
    bad.mkdir()
    (bad / "x.jpg").touch()
    with pytest.raises(ValueError, match="cannot parse vessel identity"):
        ViVReidAdapter({"root": str(root), **GOOD_CONFIG}).build_manifests()


def test_empty_tracklet_dir_skipped(tmp_path):
    root = _make_raw_layout(tmp_path)
    empty = root / "train" / "v009_cam9_t1"
    empty.mkdir(parents=True)
    ms = ViVReidAdapter({"root": str(root), **GOOD_CONFIG}).build_manifests()
    assert len(ms) == 6  # empty dir skipped


def test_train_query_overlap_raises(tmp_path):
    # Same vessel identity in train and query -> split-hygiene violation (spec §12).
    layout = {
        "train": ["v001_cam1_t1"],
        "query": ["v001_cam1_t3"],
        "gallery": ["v002_cam2_t1"],
    }
    root = _make_raw_layout(tmp_path, layout)
    with pytest.raises(ValueError, match="train identities must not appear"):
        ViVReidAdapter({"root": str(root), **GOOD_CONFIG}).build_manifests()


def test_camera_unknown_without_pattern(tmp_path):
    root = _make_raw_layout(tmp_path)
    cfg = {"root": str(root), "layout": {"tracklet_identity_pattern": r"^(?P<vessel_id>v\d+)_.*$"}}
    ms = ViVReidAdapter(cfg).build_manifests()
    assert all(m.camera_id == "unknown" for m in ms)


# ---------------------------------------------------------------------------
# TrackletManifest schema / persistence / fingerprint (spec section 4.3, 13)
# ---------------------------------------------------------------------------

def _manifest(tid="t1", vid="v1", split="train", n_frames=3, **kw):
    return TrackletManifest(
        tracklet_id=tid,
        vessel_id=vid,
        camera_id="cam0",
        split=split,
        frame_paths=[f"{tid}_{i:03d}.jpg" for i in range(n_frames)],
        **kw,
    )


def test_manifest_schema_valid():
    m = _manifest(
        timestamp_start="2024-01-01T00:00:00Z",
        timestamp_end="2024-01-01T00:00:02Z",
        fps=25.0,
        quality_score=0.9,
        occlusion_level="partial",
    )
    m.validate()  # must not raise


def test_manifest_roundtrip(tmp_path):
    ms = [
        _manifest("t1", "v1", "train", n_frames=3),
        _manifest("t2", "v2", "query", n_frames=2, frame_timestamps_utc_ms=[0, 100]),
        _manifest("t3", "v3", "gallery", n_frames=1, source_dataset="mvtd"),
    ]
    path = tmp_path / "manifests.jsonl"
    save_manifests(str(path), ms)
    loaded = load_manifests(str(path))
    assert [m.to_dict() for m in loaded] == [m.to_dict() for m in ms]
    for m in loaded:
        m.validate()


def test_manifest_hash_stable():
    # fingerprint() is the stable content hash (feature-cache keys)
    a = _manifest("t1").fingerprint()
    b = _manifest("t1").fingerprint()  # identical content -> identical hash
    assert a == b
    c = _manifest("t2").fingerprint()  # different tracklet_id -> different hash
    assert a != c
    d = _manifest("t1", n_frames=5).fingerprint()  # different content -> different hash
    assert a != d


def test_split_no_overlap():
    # Spec section 12: train identities must be disjoint from query+gallery.
    ms = [
        _manifest("t1", "v001", "train"),
        _manifest("t2", "v001", "train"),
        _manifest("q1", "v002", "query"),
        _manifest("g1", "v002", "gallery"),
        _manifest("g2", "v003", "gallery"),
    ]
    sets = identity_sets(ms)
    assert sets["train"] & (sets["query"] | sets["gallery"]) == set()
    report = validate_identity_disjointness(ms)
    assert report["train_query_overlap"] == []
    assert report["train_gallery_overlap"] == []
    assert report["train_identity_count"] == 1
