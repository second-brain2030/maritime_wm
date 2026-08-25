"""Re-ID retrieval metrics over embedding distance matrices.

The evaluation layer consumes query/gallery embeddings and identity labels:

- ``compute_distmat``: squared-Euclidean distance matrix [Q, G].
- ``compute_map``: mean Average Precision with optional camera-id exclusion
  (same pid, different camid counts as relevant — the cross-camera protocol).
- ``compute_cmc``: CMC curve at the requested ranks.
- ``compute_metrics``: convenience wrapper returning mAP + CMC{1,5,10}.

Queries with no relevant gallery item are skipped for mAP (they carry no
information) and counted as misses for CMC.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_distmat(query_embs: np.ndarray, gallery_embs: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix [Q, G].

    Uses the expansion ``||q - g||^2 = ||q||^2 - 2 q.g + ||g||^2`` for a
    memory-friendly single matrix product; tiny negative values from floating
    point are clamped to 0.
    """
    q = np.asarray(query_embs, dtype=float)
    g = np.asarray(gallery_embs, dtype=float)
    if q.ndim != 2 or g.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got query {q.ndim}D, gallery {g.ndim}D")
    if q.shape[1] != g.shape[1]:
        raise ValueError(
            f"embedding dims differ: query {q.shape[1]} vs gallery {g.shape[1]}"
        )
    if q.shape[1] == 0:
        raise ValueError("embeddings must have at least one feature dimension")
    q_norm = np.sum(q * q, axis=1, keepdims=True)
    g_norm = np.sum(g * g, axis=1, keepdims=True)
    dist = q_norm - 2.0 * (q @ g.T) + g_norm.T
    return np.maximum(dist, 0.0)


def _relevant_mask(
    q_pids: Sequence, g_pids: Sequence, q_camids, g_camids
) -> list[np.ndarray]:
    """Per-query boolean gallery relevance: same pid, different camid."""
    q_camids = q_camids if q_camids is not None else [None] * len(q_pids)
    g_camids = g_camids if g_camids is not None else [None] * len(g_pids)
    if len(q_pids) != len(q_camids):
        raise ValueError("q_pids and q_camids must have equal length")
    if len(g_pids) != len(g_camids):
        raise ValueError("g_pids and g_camids must have equal length")
    g_pids_arr = np.asarray(g_pids)
    g_camids_arr = np.asarray(g_camids, dtype=object)
    masks = []
    for pid, camid in zip(q_pids, q_camids):
        same_pid = g_pids_arr == pid
        if camid is None or all(c is None for c in g_camids):
            masks.append(same_pid)
        else:
            masks.append(same_pid & (g_camids_arr != camid))
    return masks


def compute_map(
    distmat: np.ndarray,
    q_pids: list,
    g_pids: list,
    q_camids: list | None = None,
    g_camids: list | None = None,
) -> float:
    """Mean Average Precision.

    For each query the gallery is sorted by distance (ascending); AP is the
    mean precision at each relevant gallery position. Queries with no relevant
    gallery items are skipped. Returns 0.0 if no query has any relevant item.
    """
    distmat = np.asarray(distmat, dtype=float)
    masks = _relevant_mask(q_pids, g_pids, q_camids, g_camids)
    aps: list[float] = []
    for d, rel in zip(distmat, masks):
        if not rel.any():
            continue
        order = np.argsort(d, kind="stable")
        rel_sorted = rel[order]
        n_rel = int(rel_sorted.sum())
        cum_hits = np.cumsum(rel_sorted)
        # precision at positions where the gallery item is relevant
        precisions = cum_hits[rel_sorted] / np.arange(1, len(rel_sorted) + 1)[rel_sorted]
        aps.append(float(precisions.sum()) / n_rel)
    return float(np.mean(aps)) if aps else 0.0


def compute_cmc(
    distmat: np.ndarray,
    q_pids: list,
    g_pids: list,
    ranks: list[int] = [1, 5, 10],
    q_camids: list | None = None,
    g_camids: list | None = None,
) -> dict[int, float]:
    """CMC curve: ``{rank: fraction of queries correct within rank}``.

    A query counts as a miss at every rank if it has no relevant gallery item
    or its first relevant item lies beyond the rank.
    """
    distmat = np.asarray(distmat, dtype=float)
    masks = _relevant_mask(q_pids, g_pids, q_camids, g_camids)
    ranks = sorted(int(r) for r in ranks)
    n_queries = distmat.shape[0]
    first_hit = np.full(n_queries, np.inf, dtype=float)
    for i, (d, rel) in enumerate(zip(distmat, masks)):
        if not rel.any():
            continue
        order = np.argsort(d, kind="stable")
        hit_pos = int(np.flatnonzero(rel[order])[0]) + 1  # 1-based rank
        first_hit[i] = hit_pos
    out: dict[int, float] = {}
    for r in ranks:
        out[r] = float(np.mean(first_hit <= r))
    return out


def compute_metrics(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    q_pids: list,
    g_pids: list,
    q_camids: list | None = None,
    g_camids: list | None = None,
) -> dict:
    """Convenience wrapper: distmat + mAP + CMC{1,5,10} as a flat dict.

    Keys: ``"mAP"``, ``1``, ``5``, ``10``.
    """
    distmat = compute_distmat(query_embs, gallery_embs)
    cmc = compute_cmc(distmat, q_pids, g_pids, ranks=[1, 5, 10], q_camids=q_camids, g_camids=g_camids)
    return {
        "mAP": compute_map(distmat, q_pids, g_pids, q_camids=q_camids, g_camids=g_camids),
        1: cmc[1],
        5: cmc[5],
        10: cmc[10],
    }
