#!/usr/bin/env python3
"""Evaluate a trained probe: overall Re-ID metrics + optional DGRA/blackout
gap-protocol evaluation (spec section 15 / §21).

Hydra entry point::

    python scripts/evaluate.py --config-name experiments/vivreid_cnn
    python scripts/evaluate.py --config-name experiments/vivreid_cnn \
        evaluation.protocol=dgra gap_trials_path=data/gap_trials/viv_reid_dgra.jsonl
    python scripts/evaluate.py --config-name experiments/vivreid_cnn \
        evaluation.protocol=blackout episodes_path=data/gap_trials/fvessel_blackout.jsonl

Loads the probe checkpoint (``checkpoint=...`` override, else the newest
``outputs/train_*/best_probe.pt``) and the cached features, computes
query-vs-gallery metrics via ``src.evaluation.reid_metrics.compute_metrics``,
optionally runs gap-degradation / blackout evaluation, and writes a full
report (resolved config, metrics.json, degradation_curves.csv,
degradation_slopes.json, report.md) into ``outputs/evaluate_<exp>_<ts>/``.
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
import torch
import yaml
from omegaconf import OmegaConf

from src.data.gap_trials import GapTrialManifest
from src.data.manifest import load_manifests
from src.evaluation.blackout_harness import BlackoutHarness
from src.evaluation.degradation import compute_degradation
from src.evaluation.reid_metrics import compute_metrics
from src.evaluation.reports import ReportWriter
from src.models.common_head import SharedReIDHead
from src.training.callbacks import FeatureCache
from src.utils.reproducibility import content_addressed_key, seed_everything


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
    return int(OmegaConf.select(cfg, "experiment.seed", default=42))


def _manifest_hash(manifests) -> str:
    parts = {"manifests": [m.fingerprint() for m in sorted(manifests, key=lambda m: m.tracklet_id)]}
    return content_addressed_key(parts)


def _encoder_name(cfg) -> str:
    backbone = OmegaConf.select(cfg, "model.backbone", default="resnet50")
    pretrained = bool(OmegaConf.select(cfg, "model.pretrained", default=True))
    return f"cnn_reid/{backbone}/pretrained={pretrained}"


def _preprocess_cfg(cfg) -> dict:
    return {"input_resolution": int(OmegaConf.select(cfg, "data.input_resolution", default=224))}


def _sample_cfg(cfg) -> dict:
    return {
        "frames_per_tracklet": int(OmegaConf.select(cfg, "data.frames_per_tracklet", default=16)),
        "sample_mode": str(OmegaConf.select(cfg, "data.sample_mode", default="uniform")),
        "short_tracklet_policy": str(
            OmegaConf.select(cfg, "data.short_tracklet_policy", default="repeat_last")
        ),
        "long_tracklet_policy": str(
            OmegaConf.select(cfg, "data.long_tracklet_policy", default="uniform_subsample")
        ),
    }


def _feature_cache_key(cfg, cache: FeatureCache, manifest_hash: str) -> str:
    return cache._cache_key(  # noqa: SLF001 - documented helper of FeatureCache
        _encoder_name(cfg), _preprocess_cfg(cfg), _sample_cfg(cfg), manifest_hash
    )


def _find_checkpoint(orig: Path, exp_name: str) -> Path:
    candidates = sorted(
        (orig / "outputs").glob(f"train_{exp_name}_*/best_probe.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no outputs/train_{exp_name}_*/best_probe.pt found; "
            "run scripts/train_probe.py first or pass checkpoint=<path>"
        )
    return candidates[0]


@hydra.main(config_path="../configs", config_name="experiments/vivreid_cnn", version_base=None)
def main(cfg) -> None:
    orig = _original_cwd()
    os.chdir(orig)
    seed_everything(_seed(cfg))

    exp_name = OmegaConf.select(cfg, "experiment.name", default="experiment")
    run_dir = _run_dir(orig, f"evaluate_{exp_name}")
    _dump_resolved_config(cfg, run_dir)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dataset = OmegaConf.select(cfg, "data.dataset", default="viv_reid")
    manifest_path = OmegaConf.select(cfg, "data.manifest_path", default=f"data/manifests/{dataset}.jsonl")
    manifests = load_manifests(str(orig / manifest_path))
    manifest_hash = _manifest_hash(manifests)

    cache = FeatureCache(orig / "outputs" / "feature_cache")
    key = _feature_cache_key(cfg, cache, manifest_hash)
    features = cache.load(key)
    if features is None:
        raise FileNotFoundError(
            f"feature cache outputs/feature_cache/{key}.pt not found; "
            "run scripts/extract_features.py with the same config first"
        )

    device = OmegaConf.select(cfg, "device", default="cuda" if torch.cuda.is_available() else "cpu")
    ckpt_rel = OmegaConf.select(cfg, "checkpoint", default=None)
    ckpt_path = orig / ckpt_rel if ckpt_rel else _find_checkpoint(orig, exp_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"loaded checkpoint {ckpt_path} (val_mAP={ckpt.get('val_mAP', 'n/a')})")

    input_dim = int(next(iter(features.values())).shape[-1])
    label_map = {v: i for i, v in enumerate(sorted({m.vessel_id for m in manifests}))}
    num_classes = len(label_map) or 1
    pooler = OmegaConf.select(cfg, "model.temporal_head", default="attention")
    head = SharedReIDHead(input_dim=input_dim, num_classes=num_classes, pooler=pooler)
    head.load_state_dict(ckpt["head_state_dict"])
    head.to(device).eval()

    def embed(tracklet_id: str) -> np.ndarray:
        tokens = features[tracklet_id].unsqueeze(0).to(device)  # [1, T, D]
        return head.get_embedding(tokens)[0].detach().cpu().numpy()

    query = [m for m in manifests if m.split == "query" and m.tracklet_id in features]
    gallery = [m for m in manifests if m.split == "gallery" and m.tracklet_id in features]
    if not query or not gallery:
        raise RuntimeError(
            f"need query ({len(query)}) and gallery ({len(gallery)}) tracklets with cached features"
        )
    q_embs = np.stack([embed(m.tracklet_id) for m in query])
    g_embs = np.stack([embed(m.tracklet_id) for m in gallery])
    q_pids = [m.vessel_id for m in query]
    g_pids = [m.vessel_id for m in gallery]
    q_camids = [m.camera_id for m in query]
    g_camids = [m.camera_id for m in gallery]

    metrics = compute_metrics(q_embs, g_embs, q_pids, g_pids, q_camids, g_camids)
    print("overall metrics:", json.dumps(metrics, default=str))

    # --- protocol-specific gap evaluation ---------------------------------
    protocol = OmegaConf.select(cfg, "evaluation.protocol", default=None)
    arm_name = _encoder_name(cfg)
    degradation_results = []
    blackout_results = None

    if protocol == "dgra":
        trials_path = orig / OmegaConf.select(
            cfg, "gap_trials_path", default="data/gap_trials/viv_reid_dgra.jsonl"
        )
        trials = list(GapTrialManifest.load(str(trials_path)))
        gallery_ids = [m.tracklet_id for m in gallery]
        by_bin: dict[str, list[bool]] = {}
        pooled_size = 0
        for t in trials:
            if t.gap_bin is None:
                continue
            if t.query_tracklet_id not in features or t.gallery_tracklet_id not in gallery_ids:
                continue
            if t.pool_size >= 2:
                pooled_size = pooled_size or t.pool_size
            q = embed(t.query_tracklet_id)
            dists = np.linalg.norm(g_embs - q, axis=1)
            ranked = [gallery_ids[i] for i in np.argsort(dists)]
            by_bin.setdefault(t.gap_bin, []).append(ranked[0] == t.gallery_tracklet_id)
        if by_bin:
            # Chance baseline: the trial's distractor pool when assigned,
            # otherwise the global gallery size is the only available proxy.
            pool_size = pooled_size if pooled_size >= 2 else len(gallery_ids)
            if pool_size >= 2:
                degradation_results = [
                    compute_degradation(by_bin, pool_size, arm_name, seed=_seed(cfg))
                ]
                print(
                    f"dgra degradation: bins={degradation_results[0].bin_labels} "
                    f"acc={degradation_results[0].accuracies} slope={degradation_results[0].slope:.3f}"
                )
            else:
                print("[warn] gallery too small for chance-normalized degradation; skipped")
        else:
            print("[warn] no binned DGRA trials scored; skipping degradation")
    elif protocol == "blackout":
        episodes_path = orig / OmegaConf.select(
            cfg, "episodes_path", default="data/gap_trials/fvessel_blackout.jsonl"
        )
        harness = BlackoutHarness([], {}, seed=_seed(cfg))
        episodes = harness.load_episodes(str(episodes_path))
        rankings: dict[str, list[str]] = {}
        scored = 0
        for ep in episodes:
            if ep.query_tracklet_id not in features:
                continue
            gids = [g for g in ep.gallery_tracklet_ids if g in features]
            if not gids:
                continue
            q = embed(ep.query_tracklet_id)
            gemb = np.stack([embed(g) for g in gids])
            dists = np.linalg.norm(gemb - q, axis=1)
            rankings[ep.episode_id] = [gids[i] for i in np.argsort(dists)]
            scored += 1
        if scored:
            blackout_results = harness.evaluate_arm(episodes, rankings)
            print("blackout results:", json.dumps(blackout_results, default=str))
        else:
            print("[warn] no blackout episodes scored; skipping blackout evaluation")

    # --- report -----------------------------------------------------------
    writer = ReportWriter(run_dir)
    writer.write(
        config=cfg_dict,
        metrics=metrics,
        metrics_by_slice={},
        degradation_results=degradation_results,
        blackout_results=blackout_results,
    )
    print(f"report written to {run_dir}")

    summary = {
        "mAP": metrics["mAP"],
        "rank1": metrics[1],
        "rank5": metrics[5],
        "rank10": metrics[10],
    }
    print("summary metrics:", json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
