"""Identity-continuity metrics over per-frame track associations.

Consumes a flat list of :class:`TrackFrame` records — one (frame, predicted
track, ground-truth vessel) triple per frame — and computes:

- ``compute_idsw``: identity switches of the predicted track per GT identity.
- ``compute_idf1``: ID F1 with a global Hungarian assignment of predicted
  track IDs to GT IDs maximising matched-frame overlap (Ristani et al. 2016).
- ``compute_hota``: simplified HOTA (Luiten et al. 2021) at a single alpha
  threshold over the same global assignment.

In this single-association model each frame carries exactly one predicted
track and one GT vessel, so per-frame IoU thresholds do not apply; ``alpha``
is kept for API parity with the multi-detection formulation and does not
filter matches.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackFrame:
    frame_id: int
    track_id: str       # predicted track ID
    gt_id: str          # ground-truth vessel ID
    score: float = 1.0


def _frame_lists(frames: list[TrackFrame]):
    """GT/track id sets, cumulative overlap matrix and per-id frame counts."""
    gt_ids = sorted({f.gt_id for f in frames})
    track_ids = sorted({f.track_id for f in frames})
    gt_index = {gid: i for i, gid in enumerate(gt_ids)}
    tr_index = {tid: i for i, tid in enumerate(track_ids)}
    weight = np.zeros((len(gt_ids), len(track_ids)), dtype=float)
    gt_total = np.zeros(len(gt_ids), dtype=float)
    tr_total = np.zeros(len(track_ids), dtype=float)
    for f in frames:
        gi, ti = gt_index[f.gt_id], tr_index[f.track_id]
        weight[gi, ti] += 1.0
        gt_total[gi] += 1.0
        tr_total[ti] += 1.0
    return gt_ids, track_ids, weight, gt_total, tr_total


def compute_idsw(frames: list[TrackFrame]) -> int:
    """Count identity switches: for each gt_id, the number of times its
    assigned track_id changes between consecutive frames of that identity.

    The first track_id seen for a gt_id does not count as a switch.
    """
    by_gt: dict[str, list[TrackFrame]] = defaultdict(list)
    for f in frames:
        by_gt[f.gt_id].append(f)
    switches = 0
    for gid, fs in by_gt.items():
        fs = sorted(fs, key=lambda f: f.frame_id)
        last_track: str | None = None
        for f in fs:
            if last_track is not None and f.track_id != last_track:
                switches += 1
            last_track = f.track_id
    return switches


def compute_idf1(frames: list[TrackFrame]) -> float:
    """ID F1 score: harmonic mean of IDP and IDR.

    A Hungarian assignment (``scipy.optimize.linear_sum_assignment``) maps
    predicted track IDs to GT IDs maximising the total matched-frame count
    (IDTP). IDFP = predicted frames not in the best match, IDFN = GT frames
    not matched. Returns 0.0 when there is nothing to match.
    """
    if not frames:
        return 0.0
    _, _, weight, gt_total, tr_total = _frame_lists(frames)
    rows, cols = linear_sum_assignment(-weight)
    idtp = float(weight[rows, cols].sum())
    idfp = float(tr_total.sum()) - idtp
    idfn = float(gt_total.sum()) - idtp
    denom = 2.0 * idtp + idfp + idfn
    return 0.0 if denom == 0 else 2.0 * idtp / denom


def compute_hota(frames: list[TrackFrame], alpha: float = 0.5) -> float:
    """Simplified HOTA at a single association threshold.

    Uses the same global Hungarian assignment as :func:`compute_idf1`:
    ``TP = matched frames``, ``DetA = TP/(TP+FP+FN)``. ``AssA`` is the mean
    per-matched-pair similarity ``tp/(tp+fn+fp)`` over assigned (gt, track)
    pairs, and ``HOTA = sqrt(DetA * AssA)``. Returns 0.0 when nothing is
    matched.
    """
    if not frames:
        return 0.0
    _, _, weight, gt_total, tr_total = _frame_lists(frames)
    rows, cols = linear_sum_assignment(-weight)
    idtp = float(weight[rows, cols].sum())
    idfp = float(tr_total.sum()) - idtp
    idfn = float(gt_total.sum()) - idtp
    det_a = idtp / (idtp + idfp + idfn) if (idtp + idfp + idfn) > 0 else 0.0
    sims: list[float] = []
    for r, c in zip(rows, cols):
        tp = float(weight[r, c])
        if tp <= 0:
            continue
        fn_pair = float(gt_total[r]) - tp
        fp_pair = float(tr_total[c]) - tp
        denom = tp + fn_pair + fp_pair
        sims.append(tp / denom if denom > 0 else 0.0)
    ass_a = float(np.mean(sims)) if sims else 0.0
    return math.sqrt(det_a * ass_a)
