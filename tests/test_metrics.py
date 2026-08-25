"""Re-ID retrieval metrics over embedding distance matrices.

Covers compute_distmat / compute_map / compute_cmc / compute_metrics
(src.evaluation.reid_metrics) and identity-level bootstrap_ci
(src.evaluation.bootstrap). Uses tiny numpy tensors directly — no model
mocking.
"""
import numpy as np
import pytest

from evaluation.bootstrap import bootstrap_ci
from evaluation.reid_metrics import (
    compute_cmc,
    compute_distmat,
    compute_map,
    compute_metrics,
)


def test_compute_map_perfect():
    # query embedding identical to one gallery embedding with the same pid
    q = np.array([[0.0, 0.0]])
    g = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0]])
    distmat = compute_distmat(q, g)
    assert distmat.shape == (1, 3)
    assert compute_map(distmat, [0], [0, 1, 2]) == pytest.approx(1.0)


def test_compute_map_chance():
    # random embeddings + random pids: each query has exactly one relevant
    # gallery item among G=60; at chance the expected AP is ~ H_G / G << 1.
    # Loose statistical bound: far below the perfect 1.0 but still positive.
    rng = np.random.default_rng(7)
    n_q, n_g, dim = 30, 60, 32
    q_embs = rng.normal(size=(n_q, dim))
    g_embs = rng.normal(size=(n_g, dim))
    q_pids = list(range(n_q))
    g_pids = list(range(n_g))  # all distinct from query pids -> 1 relevant each
    ap = compute_map(compute_distmat(q_embs, g_embs), q_pids, g_pids)
    assert 0.0 < ap < 0.5
    # sanity: chance is nowhere near the perfect value
    assert ap < 0.25


def test_cmc_rank1():
    # perfect match at rank 1 -> CMC[1] == 1.0
    q = np.array([[0.0, 0.0]])
    g = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
    cmc = compute_cmc(compute_distmat(q, g), [0], [0, 1, 2], ranks=[1, 5, 10])
    assert cmc[1] == pytest.approx(1.0)
    assert cmc[5] == pytest.approx(1.0)


def test_compute_metrics_returns_keys():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(4, 8))
    g = rng.normal(size=(6, 8))
    out = compute_metrics(q, g, [0, 1, 2, 3], [0, 1, 2, 3, 4, 5])
    assert set(out) == {"mAP", 1, 5, 10}
    assert 0.0 <= out["mAP"] <= 1.0
    for rank in (1, 5, 10):
        assert 0.0 <= out[rank] <= 1.0


def test_bootstrap_identity_level():
    # bootstrap_ci resamples at the vessel-identity level: input is a
    # dict of vessel_id -> score (one score per identity), not a flat list.
    scores = {"v1": 1.0, "v2": 0.0}
    mean, lo, hi = bootstrap_ci(scores, n_samples=100, seed=42)
    assert mean == pytest.approx(0.5)
    assert lo <= mean <= hi
    # identity-level resampling is deterministic for a fixed seed
    assert bootstrap_ci(scores, n_samples=100, seed=42) == (mean, lo, hi)
    # with only 2 identities the resampled mean stays in {0.0, 0.5, 1.0},
    # so the CI is exactly [0, 1] regardless of n_samples — i.e. the CI is
    # driven by the number of identities, not by any per-identity score count
    mean2, lo2, hi2 = bootstrap_ci(scores, n_samples=500, seed=42)
    assert (mean2, lo2, hi2) == (mean, lo, hi)
