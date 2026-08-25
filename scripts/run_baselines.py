#!/usr/bin/env python3
"""Run external baseline controls: Arm F motion-only dead-reckoning over the
DGRA gap trials (spec section 6.F / §21).

Hydra entry point::

    python scripts/run_baselines.py --config-name gap/viv_reid_dgra
    python scripts/run_baselines.py --config-name gap/fvessel_blackout

For every gap trial the query tracklet's centroid + velocity are extrapolated
across the gap with ``KalmanDeadReckon.rank`` and the gallery (all gallery-split
tracklets) is ranked by predicted position. Reports top-1 accuracy per gap bin
and checks the dead-reckoning sanity gate: long-gap Top-1 must not beat the
chance baseline more than 2x (chance = 1/pool_size; when trials carry no
distractor pool yet, the global gallery size is used as the documented proxy).
Results are written to ``outputs/baseline_<config>_<ts>/baseline_kalman.json``.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.data.gap_trials import GapTrialManifest
from src.data.manifest import load_manifests
from src.models.baselines.kalman_deadreckon import KalmanDeadReckon
from src.utils.reproducibility import seed_everything


def _original_cwd() -> Path:
    return Path(hydra.utils.get_original_cwd())


def _dump_resolved_config(cfg, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (out_dir / "resolved_config.yaml").write_text(
        yaml.dump(resolved, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _run_dir(orig: Path, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = orig / "outputs" / f"{tag}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed(cfg) -> int:
    seed = OmegaConf.select(cfg, "experiment.seed", default=None)
    if seed is None:
        seed = OmegaConf.select(cfg, "gap_protocol.seed", default=42)
    return int(seed)


def _path_opt(cfg, key: str, default: str) -> Path:
    """Top-level path key, falling back to the same key under gap_protocol."""
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        value = OmegaConf.select(cfg, f"gap_protocol.{key}", default=default)
    return Path(value)


def _centroid(m) -> np.ndarray | None:
    """Mean bbox center [x, y]; None when the tracklet has no bboxes."""
    boxes = [b for b in (m.frame_bboxes or []) if b]
    if not boxes:
        return None
    xs = [b[0] + b[2] / 2.0 for b in boxes]
    ys = [b[1] + b[3] / 2.0 for b in boxes]
    return np.array([float(np.mean(xs)), float(np.mean(ys))])


def _velocity(m) -> np.ndarray | None:
    """Constant-velocity estimate from first/last bbox centers over time."""
    boxes = [b for b in (m.frame_bboxes or []) if b]
    if len(boxes) < 2:
        return None
    first, last = boxes[0], boxes[-1]
    fps = m.fps or 25.0
    if m.frame_indices and len(m.frame_indices) >= 2:
        dt = (m.frame_indices[-1] - m.frame_indices[0]) / fps
    else:
        dt = (len(boxes) - 1) / fps
    if dt <= 0:
        return None
    dx = (last[0] + last[2] / 2.0) - (first[0] + first[2] / 2.0)
    dy = (last[1] + last[3] / 2.0) - (first[1] + first[3] / 2.0)
    return np.array([dx / dt, dy / dt])


@hydra.main(config_path="../configs", config_name="gap/viv_reid_dgra", version_base=None)
def main(cfg) -> None:
    orig = _original_cwd()
    os.chdir(orig)
    seed_everything(_seed(cfg))

    config_name = (HydraConfig.get().job.config_name or "gap/viv_reid_dgra").split("/")[-1]
    run_dir = _run_dir(orig, f"baseline_{config_name}")
    _dump_resolved_config(cfg, run_dir)

    manifest_path = _path_opt(cfg, "manifest_path", "data/manifests/viv_reid.jsonl")
    trials_path = orig / _path_opt(cfg, "trials_out", f"data/gap_trials/{config_name}.jsonl")
    manifests = load_manifests(str(orig / manifest_path))
    trials = list(GapTrialManifest.load(str(trials_path)))
    by_id = {m.tracklet_id: m for m in manifests}
    print(f"loaded {len(trials)} gap trials, {len(manifests)} tracklets")

    kf = KalmanDeadReckon()
    gallery_centroids = {
        m.tracklet_id: c
        for m in manifests
        if m.split == "gallery" and (c := _centroid(m)) is not None
    }
    print(f"gallery candidates with positions: {len(gallery_centroids)}")

    per_bin: dict[str, list[bool]] = {}
    results: list[dict] = []
    skipped = 0
    for t in trials:
        q = by_id.get(t.query_tracklet_id)
        if q is None:
            skipped += 1
            continue
        qc, qv = _centroid(q), _velocity(q)
        if qc is None or qv is None or not gallery_centroids:
            skipped += 1
            continue
        ranked = kf.rank(qc, qv, t.gap_seconds or 0.0, gallery_centroids)
        correct = bool(ranked and ranked[0] == t.gallery_tracklet_id)
        per_bin.setdefault(t.gap_bin or "unknown", []).append(correct)
        results.append(
            {
                "trial_id": t.trial_id,
                "gap_bin": t.gap_bin,
                "gap_seconds": t.gap_seconds,
                "gap_type": t.gap_type,
                "pool_size": t.pool_size,
                "top1_correct": correct,
                "ranked_head": ranked[:5],
            }
        )

    bin_stats = {
        bin_name: {
            "top1": float(np.mean(hits)),
            "num_trials": len(hits),
        }
        for bin_name, hits in per_bin.items()
    }
    print("per-bin top-1:", json.dumps(bin_stats, default=str))

    # --- dead-reckoning sanity gate ---------------------------------------
    long_hits = per_bin.get("long", [])
    long_top1 = float(np.mean(long_hits)) if long_hits else None
    pool_size = next((t.pool_size for t in trials if t.pool_size >= 2), 0)
    if pool_size >= 2:
        chance = 1.0 / pool_size
        chance_note = f"trial pool_size={pool_size}"
    else:
        # Trials carry pool_size 0 until distractor pools land; the global
        # gallery size is the only documented chance proxy available.
        chance = 1.0 / max(1, len(gallery_centroids))
        chance_note = (
            f"no distractor pools (pool_size=0); chance proxy 1/|gallery| = "
            f"1/{len(gallery_centroids)}"
        )
    if long_top1 is None:
        gate = {"status": "SKIP", "reason": "no long-gap trials scored"}
        print("dead-reckoning sanity gate: SKIP (no long-gap trials)")
    else:
        gate_pass = long_top1 <= 2.0 * chance + 1e-9
        gate = {
            "status": "PASS" if gate_pass else "FAIL",
            "long_gap_top1": long_top1,
            "chance": chance,
            "two_x_chance": 2.0 * chance,
            "note": chance_note,
        }
        print(f"dead-reckoning sanity gate: {'PASS' if gate_pass else 'FAIL'} "
              f"(long-gap top-1={long_top1:.3f} vs 2x chance={2.0 * chance:.3f} [{chance_note}])")

    payload = {
        "arm": "kalman_deadreckon",
        "config_name": config_name,
        "seed": _seed(cfg),
        "num_trials": len(results),
        "skipped_no_position": skipped,
        "gallery_candidates": len(gallery_centroids),
        "per_bin_top1": bin_stats,
        "sanity_gate": gate,
    }
    out_path = run_dir / "baseline_kalman.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote baseline results -> {out_path}")


if __name__ == "__main__":
    main()
