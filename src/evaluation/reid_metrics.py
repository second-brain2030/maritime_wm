"""Re-ID retrieval metrics (spec section 9).

Convention: ``scores_by_query[q]`` maps gallery tracklet id -> score (higher
is better); ``gt[q]`` is the set of relevant gallery ids.
"""
from __future__ import annotations

from typing import Mapping, Sequence


def rank_gallery(scores: Mapping[str, float], relevant: set[str] | None = None) -> list[str]:
    """Rank gallery ids by score, descending (ties broken by id for stability)."""
    return sorted(scores, key=lambda gid: (scores[gid], gid), reverse=True)


def cmc(
    scores_by_query: Mapping[str, Mapping[str, float]],
    gt: Mapping[str, set[str]],
    max_rank: int | None = None,
) -> list[float]:
    """CMC curve: rank-k accuracy for k = 1..max_rank."""
    queries = list(scores_by_query)
    if not queries:
        return []
    k_max = max_rank or max(len(s) for s in scores_by_query.values())
    hits = [0] * k_max
    for q in queries:
        relevant = gt.get(q, set())
        ranked = rank_gallery(scores_by_query[q])
        first = next((i + 1 for i, gid in enumerate(ranked) if gid in relevant), None)
        if first is not None:
            for r in range(first - 1, k_max):
                hits[r] += 1
    return [h / len(queries) for h in hits]


def mean_average_precision(
    scores_by_query: Mapping[str, Mapping[str, float]],
    gt: Mapping[str, set[str]],
) -> float:
    """Mean average precision over queries (spec section 9 primary metric)."""
    aps: list[float] = []
    for q, scores in scores_by_query.items():
        relevant = gt.get(q, set())
        if not relevant:
            continue
        ranked = rank_gallery(scores)
        hits = 0
        sum_prec = 0.0
        for i, gid in enumerate(ranked, start=1):
            if gid in relevant:
                hits += 1
                sum_prec += hits / i
        aps.append(sum_prec / len(relevant))
    return sum(aps) / len(aps) if aps else 0.0
