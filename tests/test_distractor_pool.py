import pytest

from data.distractor_pool import DistractorPool, DistractorPoolManifest, chance_baseline


def make_pool(k=5):
    return DistractorPool(
        pool_id="p1",
        target_vessel_id="v1",
        candidate_tracklet_ids=[f"g{i}" for i in range(k)],
        reference_embedding_name="dino_vitb16",
        seed=42,
    )


def test_chance_baseline():
    assert make_pool(10).size == 10
    assert make_pool(10).chance_top1 == pytest.approx(0.1)
    assert chance_baseline(5) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        chance_baseline(0)


def test_duplicates_raise():
    p = make_pool(3)
    p.candidate_tracklet_ids[1] = p.candidate_tracklet_ids[0]
    with pytest.raises(ValueError):
        p.validate()


def test_empty_pool_raises():
    p = make_pool(0)
    with pytest.raises(ValueError):
        p.validate()


def test_roundtrip(tmp_path):
    p = make_pool(5)
    m = DistractorPoolManifest([p])
    path = tmp_path / "pools.jsonl"
    m.save(str(path))
    m2 = DistractorPoolManifest.load(str(path))
    assert len(m2) == 1
    assert m2.get("p1").to_dict() == p.to_dict()
    assert m2.get("missing") is None
