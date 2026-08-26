#!/usr/bin/env python3
"""Run external baseline controls on blackout episodes (brief P3; spec §21).

Arms:
  F  kalman_deadreckon  motion-only dead-reckoning (no appearance)
  G  tracker_reid       raw frozen-backbone appearance embedding (no probe)
  H  ais_upper_bound    AIS-fused association (AIS available; separate table)

Outputs the same per-duration re-acquisition format as evaluate.py so arms
compare on identical trials.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.ais import AisTrajectoryManifest, split_pings_by_window
from data.manifest import load_manifests
from evaluation.baselines import ais_rank, appearance_rank, deadreckon_rank
from evaluation.blackout_harness import BlackoutConfig, BlackoutEpisodeManifest
from evaluation.degradation import compute_degradation
from evaluation.features import embed_frames, tracklet_visible_bboxes
from evaluation.reacquisition_eval import episode_result
from evaluation.tracking_metrics import summarize_reacquisition
from models import encoder_registry
from utils.config import load_config

BASELINES = ("kalman_deadreckon", "tracker_reid", "ais_upper_bound")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", choices=BASELINES, required=True)
    ap.add_argument("--config-name", default="fvessel_cnn")
    ap.add_argument("--episodes", default="data/gap_trials/blackout.jsonl")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--ais-manifest", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--appearance-arm", default="cnn_reid", help="Arm G appearance encoder")
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("experiments", args.config_name)
    manifests = load_manifests(args.manifest or cfg["data"]["manifest_path"])
    index = {(m.camera_id, m.vessel_id): m for m in manifests}
    episodes = BlackoutEpisodeManifest.load(args.episodes)
    if args.max_episodes is not None:
        episodes = BlackoutEpisodeManifest(list(episodes)[: args.max_episodes])

    ais_by = {}
    ais_path = args.ais_manifest or cfg.get("ais_manifest_path")
    if ais_path and Path(ais_path).is_file():
        ais_by = {t.vessel_id: t for t in AisTrajectoryManifest.load(ais_path)}

    bcfg = BlackoutConfig()
    sidecar = Path(args.episodes).with_suffix(Path(args.episodes).suffix + ".config.json")
    if sidecar.is_file():
        bcfg = BlackoutConfig.from_dict(json.loads(sidecar.read_text()))

    encoder = None
    if args.baseline == "tracker_reid":
        model_cfg = dict(cfg["model"])
        arm = args.appearance_arm
        model_cfg.pop("arm", None)
        for key in ("frozen_backbone", "temporal_head", "embedding_dim", "input_size", "token_dim", "pool_tokens"):
            model_cfg.pop(key, None)
        encoder = encoder_registry.create(arm, **model_cfg)

    results = []
    for ep in episodes:
        target = index.get((ep.sequence_id, ep.vessel_id))
        if target is None:
            continue
        candidates = {
            cid: index[(ep.sequence_id, cid)]
            for cid in ep.candidate_vessel_ids
            if (ep.sequence_id, cid) in index
        }  # includes the target: its reappearance observation is the correct match
        if args.baseline == "kalman_deadreckon":
            ranked, drift = deadreckon_rank(ep, target, candidates)
        elif args.baseline == "tracker_reid":
            ranked, drift = appearance_rank(ep, target, candidates, encoder)
        else:  # ais_upper_bound
            visible, _ = split_pings_by_window(
                ais_by[ep.vessel_id].pings,
                ep.blackout_start_utc_ms,
                ep.blackout_start_utc_ms + int(ep.blackout_duration_s * 1000),
                jitter_ms=bcfg.jitter_ms,
                dropout_p=bcfg.ais_dropout_p,
                seed=bcfg.seed,
            ) if ep.vessel_id in ais_by else ([], [])
            cand_pings = {
                cid: ais_by[cid].pings
                for cid in ep.candidate_vessel_ids
                if cid in ais_by
            }
            ranked, drift = ais_rank(ep, visible, cand_pings)
        r = episode_result(ep, ranked, drift)
        r["self_cosine"] = None  # kinematic/AIS baselines have no appearance
        results.append(r)

    summary = summarize_reacquisition(results)
    self_cos_by_duration: dict[str, float] = {}
    if args.baseline == "tracker_reid" and encoder is not None:
        # recompute appearance self-cosine per episode (stable-feature signal)
        for ep in episodes:
            target = index.get((ep.sequence_id, ep.vessel_id))
            if target is None:
                continue
            q_frames = [
                fi for fi, _ in tracklet_visible_bboxes(target) if fi < ep.blackout_start_frame
            ][-16:]
            q_paths = [p for fi, p in zip(target.frame_indices or [], target.frame_paths) if fi in q_frames]
            near = [
                fi for fi, _ in tracklet_visible_bboxes(target)
                if abs(fi - ep.reappearance_frame) <= int(0.5 * (target.fps or 25))
            ]
            if not q_paths or not near:
                continue
            q_emb = embed_frames(encoder, None, q_paths, fps=target.fps)
            t_emb = embed_frames(encoder, None, [p for fi, p in zip(target.frame_indices or [], target.frame_paths) if fi in near], fps=target.fps)
            qn = q_emb / (float(np.linalg.norm(q_emb)) + 1e-9)
            tn = t_emb / (float(np.linalg.norm(t_emb)) + 1e-9)
            key = f"{int(ep.blackout_duration_s)}s"
            self_cos_by_duration.setdefault(key, []).append(float(qn @ tn))
        self_cos_by_duration = {k: float(np.mean(v)) for k, v in self_cos_by_duration.items()}
    by_bin: dict[str, list[bool]] = {}
    pools: list[int] = []
    for r in results:
        key = f"{int(r['duration_s'])}s"
        by_bin.setdefault(key, []).append(bool(r["rank_of_correct"] == 1))
        pools.append(int(r["n_candidates"]))
    deg = None
    # chance normalization requires pool_size >= 2 (single-object tracking,
    # e.g. MVTD, has pool=1 and thus no meaningful re-acquisition accuracy)
    if by_bin:
        pool_size = max(1, int(round(sum(pools) / len(pools))))
        if pool_size >= 2:
            deg = compute_degradation(by_bin, pool_size=pool_size, arm_name=args.baseline)
    slope = deg.slope if deg is not None else None

    out = Path(args.output or f"outputs/baselines/{args.baseline}/reacquisition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": args.baseline,
        "dataset": cfg["data"].get("dataset", "unknown"),
        "summary": summary,
        "self_cosine_by_duration": self_cos_by_duration,
        "degradation": asdict(deg) if deg is not None else None,
        "degradation_slope": slope,
        "n_episodes": len(results),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    with open(out.with_suffix(".results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(payload["summary"]["per_duration"], indent=2, default=str))
    print(f"degradation_slope={slope} -> {out}")


if __name__ == "__main__":
    main()
