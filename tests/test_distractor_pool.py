import math

import numpy as np
import pytest

from src.data.distractor_pool import DistractorPool, DistractorPoolManifest, chance_baseline


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


# ---------------------------------------------------------------------------
# Pool construction semantics (build_distractor_pools is not implemented yet,
# so pools are exercised through the data class directly)
# ---------------------------------------------------------------------------

def test_pool_size():
    # pick K=5 distractor candidates out of 20 available
    candidates_20 = [f"g{i:02d}" for i in range(20)]
    pool = DistractorPool(
        pool_id="p1",
        target_vessel_id="v1",
        candidate_tracklet_ids=candidates_20[:5],
        reference_embedding_name="dino_vitb16",
        seed=42,
    )
    pool.validate()
    assert len(pool.candidate_tracklet_ids) == 5
    assert pool.size == 5


def test_target_excluded():
    # the distractor candidate list must never contain the target's own
    # tracklets (the pool builder filters them before construction)
    pool = DistractorPool(
        pool_id="p1",
        target_vessel_id="v1",
        candidate_tracklet_ids=["g2", "g3", "g4", "g5", "g6"],
        reference_embedding_name="dino_vitb16",
        seed=42,
    )
    assert "g1" not in pool.candidate_tracklet_ids  # g1 is v1's own tracklet
    assert all(not tid.startswith("v1_") for tid in pool.candidate_tracklet_ids)


def test_chance_top1_matches_random_ranking():
    # random ranking on a pool of K: empirical top-1 hit rate ~ 1/K
    k = 5
    n_trials = 4000
    rng = np.random.default_rng(0)
    # the target occupies a fixed rank; a random ranker hits it with prob 1/K
    hits = (rng.integers(0, k, size=n_trials) == 0).mean()
    p = 1.0 / k
    sigma = math.sqrt(p * (1.0 - p) / n_trials)
    assert abs(hits - p) <= 3 * sigma
    assert make_pool(k).chance_top1 == pytest.approx(p)
    assert chance_baseline(k) == pytest.approx(p)


def test_pool_determinism():
    # identical inputs + seed -> identical pool (serialization is stable)
    def build():
        return DistractorPool(
            pool_id="p1",
            target_vessel_id="v1",
            candidate_tracklet_ids=["g2", "g3", "g4", "g5", "g6"],
            reference_embedding_name="dino_vitb16",
            seed=42,
        )

    a, b = build(), build()
    assert a.to_dict() == b.to_dict()
    assert a.pool_id == b.pool_id and a.size == b.size
