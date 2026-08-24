"""Tracking + re-acquisition metrics (pilot brief P4; spec section 9).

- Re-acquisition accuracy: Top-1 / Top-5 rank of the correct candidate at
  reappearance after a blackout.
- Identity continuity: track ID switches (IDSW), IDF1, HOTA.
- Localization drift: Haversine (geo) or pixel distance between the predicted
  state and the ground-truth coordinate at reappearance.
- Degradation slope: reuse evaluation.degradation.degradation_slope for the
  per-arm accuracy drop across blackout-duration bins (the pilot's primary
  hypothesis: Arm D keeps a flatter degradation curve).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Localization drift
# ---------------------------------------------------------------------------


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters (WGS84)."""
    r_earth = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r_earth * math.asin(math.sqrt(a))


def pixel_distance(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    """Euclidean distance between two [x, y, w, h] box centers."""
    cx_a, cy_a = bbox_a[0] + bbox_a[2] / 2, bbox_a[1] + bbox_a[3] / 2
    cx_b, cy_b = bbox_b[0] + bbox_b[2] / 2, bbox_b[1] + bbox_b[3] / 2
    return float(math.hypot(cx_a - cx_b, cy_a - cy_b))


# ---------------------------------------------------------------------------
# Re-acquisition accuracy
# ---------------------------------------------------------------------------


def reacquisition_topk(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of episodes where the correct candidate ranked <= k at
    reappearance. ``None`` rank = correct candidate not in the candidate pool
    (counted as a miss)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


# ---------------------------------------------------------------------------
# Identity continuity
# ---------------------------------------------------------------------------


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two [x, y, w, h] boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def id_switches(aligned: Sequence[tuple[Any, Any]]) -> int:
    """Count ID switches over per-frame aligned (gt_id, pred_id) matches.

    A switch is counted when a ground-truth identity's matched prediction id
    changes between consecutive frames in which the identity is present.
    Caller must pass the matched identity pairs per frame.
    """
    last: dict[Any, Any] = {}
    switches = 0
    for gt_id, pred_id in aligned:
        if gt_id in last and last[gt_id] != pred_id:
            switches += 1
        last[gt_id] = pred_id
    return switches


def _cumulative_iou_weights(
    gt_frames: Sequence[Sequence[tuple[Any, Sequence[float]]]],
    pred_frames: Sequence[Sequence[tuple[Any, Sequence[float]]]],
) -> tuple[list[Any], list[Any], np.ndarray, dict[Any, int], dict[Any, int]]:
    gt_ids = sorted({i for f in gt_frames for i, _ in f})
    pred_ids = sorted({i for f in pred_frames for i, _ in f})
    weight = np.zeros((len(gt_ids), len(pred_ids)))
    gt_det_count = {i: 0 for i in gt_ids}
    pred_det_count = {i: 0 for i in pred_ids}
    for gf, pf in zip(gt_frames, pred_frames):
        gmap = dict(gf)
        pmap = dict(pf)
        for gid in gmap:
            gt_det_count[gid] += 1
        for pid in pmap:
            pred_det_count[pid] += 1
        for gi, gid in enumerate(gt_ids):
            if gid not in gmap:
                continue
            for pj, pid in enumerate(pred_ids):
                if pid in pmap:
                    weight[gi, pj] += iou(gmap[gid], pmap[pid])
    return gt_ids, pred_ids, weight, gt_det_count, pred_det_count


def idf1(
    gt_frames: Sequence[Sequence[tuple[Any, Sequence[float]]]],
    pred_frames: Sequence[Sequence[tuple[Any, Sequence[float]]]],
) -> float:
    """Identity F1 (Ristani et al. 2016) over aligned per-frame detection lists.

    ``gt_frames[t]`` = [(id, [x, y, w, h]), ...]; ``pred_frames`` likewise.
    """
    gt_ids, pred_ids, weight, gt_det_count, pred_det_count = _cumulative_iou_weights(
        gt_frames, pred_frames
    )
    if not gt_ids or not pred_ids:
        return 0.0
    rows, cols = linear_sum_assignment(-weight)
    idtp = float(weight[rows, cols].sum())
    idfp = float(sum(pred_det_count.values())) - idtp
    idfn = float(sum(gt_det_count.values())) - idtp
    denom = 2 * idtp + idfp + idfn
    return 0.0 if denom == 0 else 2 * idtp / denom


def hota(
    gt_frames: Sequence[Sequence[tuple[Any, Sequence[float]]]],
    pred_frames: Sequence[Sequence[tuple[Any, Sequence[float]]]],
    alphas: Sequence[float] | None = None,
) -> float:
    """HOTA (Luiten et al. 2021), mean over IoU association thresholds.

    Per alpha: per-frame optimal matching with IoU >= alpha, then
    DetA = TP/(TP+FP+FN) and AssA = mean trajectory similarity over matched
    pairs, HOTA(alpha) = sqrt(AssA * DetA); final = mean over alphas.
    """
    if alphas is None:
        alphas = [round(0.05 + 0.05 * i, 2) for i in range(19)]  # 0.05..0.95
    scores: list[float] = []
    for alpha in alphas:
        total_tp = total_fp = total_fn = 0
        pair_tp: dict[tuple[Any, Any], int] = {}
        gt_total: dict[Any, int] = {}
        pred_total: dict[Any, int] = {}
        for gf, pf in zip(gt_frames, pred_frames):
            gmap = dict(gf)
            pmap = dict(pf)
            for gid in gmap:
                gt_total[gid] = gt_total.get(gid, 0) + 1
            for pid in pmap:
                pred_total[pid] = pred_total.get(pid, 0) + 1
            gids, pids = list(gmap), list(pmap)
            if gids and pids:
                cost = np.array([[iou(gmap[g], pmap[p]) for p in pids] for g in gids])
                rows, cols = linear_sum_assignment(-cost)
                matched = [(gids[r], pids[c]) for r, c in zip(rows, cols) if cost[r, c] >= alpha]
            else:
                matched = []
            for g, p in matched:
                pair_tp[(g, p)] = pair_tp.get((g, p), 0) + 1
            total_tp += len(matched)
            total_fp += len(pids) - len(matched)
            total_fn += len(gids) - len(matched)
        det_a = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0.0
        sims: list[float] = []
        for (g, p), tp in pair_tp.items():
            fn_pair = gt_total[g] - tp
            fp_pair = pred_total[p] - tp
            sims.append(tp / (tp + fn_pair + fp_pair) if (tp + fn_pair + fp_pair) else 0.0)
        ass_a = float(np.mean(sims)) if sims else 0.0
        scores.append(math.sqrt(ass_a * det_a))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Re-acquisition summary (feeds degradation curves)
# ---------------------------------------------------------------------------


def summarize_reacquisition(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-episode re-acquisition results by blackout duration.

    Each result: ``{duration_s, rank_of_correct (int|None), n_candidates,
    drift_m (float|None)}``. Returns per-duration Top-1/Top-5, mean drift,
    chance baseline, plus an overall degradation-ready accuracy map.
    """
    by_duration: dict[float, list[Mapping[str, Any]]] = {}
    for r in results:
        by_duration.setdefault(float(r["duration_s"]), []).append(r)

    durations = sorted(by_duration)
    per_duration: list[dict[str, Any]] = []
    for d in durations:
        rs = by_duration[d]
        ranks = [r.get("rank_of_correct") for r in rs]
        pools = [r.get("n_candidates", 0) for r in rs]
        drifts = [r.get("drift_m") for r in rs if r.get("drift_m") is not None]
        chance = 1.0 / max(1, int(np.mean(pools))) if pools else None
        per_duration.append(
            {
                "duration_s": d,
                "n_episodes": len(rs),
                "top1": reacquisition_topk(ranks, 1),
                "top5": reacquisition_topk(ranks, 5),
                "mean_drift_m": float(np.mean(drifts)) if drifts else None,
                "mean_pool_size": float(np.mean(pools)) if pools else None,
                "chance_top1": chance,
            }
        )
    acc_by_bin = {f"{int(d)}s": row["top1"] for d, row in zip(durations, per_duration)}
    return {"per_duration": per_duration, "top1_by_duration": acc_by_bin}
