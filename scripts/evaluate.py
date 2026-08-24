#!/usr/bin/env python3
"""Evaluate a trained probe on blackout episodes (brief P1/P4).

Per episode: embed the pre-blackout query frames with the arm encoder + probe
head, embed each candidate's frames at reappearance, rank by cosine, record
the correct candidate's rank and localization drift. Outputs a per-duration
re-acquisition summary + degradation slope, identical in format to the
baseline runs so arms compare on the same trials.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.ais import AisTrajectoryManifest, split_pings_by_window
from data.manifest import load_manifests
from evaluation.blackout_harness import BlackoutConfig, BlackoutEpisodeManifest
from evaluation.degradation import degradation_slope
from evaluation.features import embed_frames, tracklet_visible_bboxes
from evaluation.reacquisition_eval import (
    ais_drift_m,
    episode_result,
    pixel_drift_m,
    predict_bbox_center,
    predict_lonlat_from_pings,
    rank_by_cosine,
)
from evaluation.tracking_metrics import summarize_reacquisition
from models import encoder_registry
from training.probe import ProbeArtifacts, build_head
from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="fvessel_cnn")
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--ais-manifest", default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("experiments", args.config_name)
    manifests = load_manifests(cfg["data"]["manifest_path"])
    index = {(m.camera_id, m.vessel_id): m for m in manifests}

    episodes = BlackoutEpisodeManifest.load(
        args.episodes or cfg.get("episodes_path", "data/gap_trials/blackout.jsonl")
    )
    if args.max_episodes is not None:
        episodes = BlackoutEpisodeManifest(list(episodes)[: args.max_episodes])

    model_cfg = dict(cfg["model"])
    arm = model_cfg.pop("arm")
    encoder = encoder_registry.create(arm, **model_cfg)
    probe_path = args.probe or f"outputs/probes/{arm}/probe.pt"
    head = build_head(ProbeArtifacts.load(probe_path))
    print(f"arm={arm} probe={probe_path} episodes={len(episodes)}")

    ais_by = {}
    ais_path = args.ais_manifest or cfg.get("ais_manifest_path")
    if ais_path and Path(ais_path).is_file():
        ais_by = {t.vessel_id: t for t in AisTrajectoryManifest.load(ais_path)}

    bcfg = BlackoutConfig()
    cfg_path = Path(args.episodes or "data/gap_trials/blackout.jsonl")
    sidecar = cfg_path.with_suffix(cfg_path.suffix + ".config.json")
    if sidecar.is_file():
        bcfg = BlackoutConfig.from_dict(json.loads(sidecar.read_text()))

    results = []
    for ep in episodes:
        target = index.get((ep.sequence_id, ep.vessel_id))
        if target is None:
            continue
        q_frames = [
            fi for fi, _ in tracklet_visible_bboxes(target) if fi < ep.blackout_start_frame
        ][-16:]
        q_paths = [p for fi, p in zip(target.frame_indices or [], target.frame_paths) if fi in q_frames]
        if not q_paths:
            continue
        q_emb = embed_frames(encoder, head, q_paths, fps=target.fps)

        cand_embs = {}
        for cid in ep.candidate_vessel_ids:
            ct = index.get((ep.sequence_id, cid))
            if ct is None:
                continue
            near = [
                fi for fi, _ in tracklet_visible_bboxes(ct)
                if abs(fi - ep.reappearance_frame) <= int(0.5 * (ct.fps or 25))
            ]
            if not near:
                fi, _ = min(tracklet_visible_bboxes(ct), key=lambda vb: abs(vb[0] - ep.reappearance_frame))
                near = [fi]
            cand_embs[cid] = embed_frames(
                encoder, head,
                [p for fi, p in zip(ct.frame_indices or [], ct.frame_paths) if fi in near],
                fps=ct.fps,
            )
        ranked = rank_by_cosine(q_emb, cand_embs)

        drift = None
        if ep.gt_lonlat_at_reappearance is not None and ep.vessel_id in ais_by:
            visible, _ = split_pings_by_window(
                ais_by[ep.vessel_id].pings,
                ep.blackout_start_utc_ms,
                ep.blackout_start_utc_ms + int(ep.blackout_duration_s * 1000),
                jitter_ms=bcfg.jitter_ms,
                dropout_p=bcfg.ais_dropout_p,
                seed=bcfg.seed,
            )
            drift = ais_drift_m(
                predict_lonlat_from_pings(visible, ep.blackout_duration_s),
                ep.gt_lonlat_at_reappearance,
            )
        elif ep.gt_bbox_at_reappearance is not None:
            obs = [
                (fi / (target.fps or 25), bb)
                for fi, bb in tracklet_visible_bboxes(target)
                if fi < ep.blackout_start_frame
            ][-8:]
            drift = pixel_drift_m(
                predict_bbox_center(obs, ep.blackout_duration_s),
                ep.gt_bbox_at_reappearance,
            )
        results.append(episode_result(ep, ranked, drift))

    summary = summarize_reacquisition(results)
    acc = summary["top1_by_duration"]
    centers = {k: float(k[:-1]) for k in acc}
    slope = degradation_slope(acc, bin_centers=centers) if len(acc) >= 2 else None

    out = Path(args.output or f"outputs/eval/{arm}/reacquisition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"arm": arm, "summary": summary, "degradation_slope": slope, "n_episodes": len(results)}
    out.write_text(json.dumps(payload, indent=2, default=str))
    with open(out.with_suffix(".results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(payload["summary"]["per_duration"], indent=2, default=str))
    print(f"degradation_slope={slope} -> {out}")


if __name__ == "__main__":
    main()
