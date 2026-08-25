import pytest

from src.data.manifest import TrackletManifest
from src.data.splits import validate_identity_disjointness


def m(tid, vid, split):
    return TrackletManifest(
        tracklet_id=tid, vessel_id=vid, camera_id="c0", split=split,
        frame_paths=["a.jpg"],
    )


def test_no_overlap():
    r = validate_identity_disjointness(
        [m("t1", "v1", "train"), m("q1", "v2", "query"), m("g1", "v3", "gallery")]
    )
    assert r["train_query_overlap"] == []
    assert r["train_gallery_overlap"] == []
    assert r["train_identity_count"] == 1


def test_train_query_overlap_raises():
    with pytest.raises(ValueError):
        validate_identity_disjointness([m("t1", "v1", "train"), m("q1", "v1", "query")])


def test_train_gallery_overlap_raises():
    with pytest.raises(ValueError):
        validate_identity_disjointness([m("t1", "v1", "train"), m("g1", "v1", "gallery")])


def test_query_gallery_shared_allowed():
    # Official Re-ID protocol allows query/gallery identity sharing.
    r = validate_identity_disjointness([m("q1", "v1", "query"), m("g1", "v1", "gallery")])
    assert r["train_query_overlap"] == []
