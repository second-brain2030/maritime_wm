#!/usr/bin/env python3
"""Extract frozen-backbone tracklet features (spec §21; brief P3).

Loads the experiment config, builds the arm encoder from the registry, samples
`frames_per_tracklet` frames per tracklet, and caches per-tracklet features to
outputs/features/<arm>/<content-addressed-key>.pt. Cache keys include the
checkpoint, sampling policy, and manifest fingerprint (spec §13), so re-runs
with identical inputs are no-ops.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torchvision.transforms import ToTensor

from data.manifest import load_manifests
from data.sampling import sample_frame_indices
from models import encoder_registry
from utils.config import load_config
from utils.media import load_frame
from utils.reproducibility import content_addressed_key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="vivreid_vjepa_encoder")
    ap.add_argument("--max-tracklets", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("experiments", args.config_name)
    model_cfg = dict(cfg["model"])
    arm = model_cfg.pop("arm")
    # metadata keys consumed by the pipeline, not passed to the adapter
    model_cfg.pop("embedding_dim", None)
    model_cfg.pop("input_size", None)
    model_cfg.pop("token_dim", None)
    encoder = encoder_registry.create(arm, **model_cfg)
    print(f"arm={arm} encoder={encoder.name} dim={encoder.embedding_dim}")

    data_cfg = cfg["data"]
    manifests = load_manifests(data_cfg["manifest_path"])
    if args.max_tracklets is not None:
        manifests = manifests[: args.max_tracklets]
    print(f"loaded {len(manifests)} tracklets")

    frames_per = int(data_cfg.get("frames_per_tracklet", 16))
    mode = data_cfg.get("sample_mode", "uniform")
    features_dir = Path(cfg.get("features_dir", "outputs/features")) / arm
    features_dir.mkdir(parents=True, exist_ok=True)

    cached = skipped = 0
    for m in manifests:
        key = content_addressed_key(
            {
                "arm": arm,
                "encoder": encoder.name,
                "checkpoint": getattr(encoder, "checkpoint", None),
                "sampling": {"frames_per_tracklet": frames_per, "mode": mode},
                "manifest_fingerprint": m.fingerprint(),
            }
        )
        out_path = features_dir / f"{key}.pt"
        if out_path.exists():
            cached += 1
            continue
        idx = sample_frame_indices(
            len(m.frame_paths), frames_per, mode=mode, seed=cfg["experiment"].get("seed", 42)
        )
        try:
            frames = torch.stack(
                [ToTensor()(load_frame(m.frame_paths[i], fps=m.fps)) for i in idx]
            )  # [T, 3, H, W] in [0, 1]
        except Exception as e:  # missing media file etc.
            print(f"[warn] skipping {m.tracklet_id}: {e}")
            skipped += 1
            continue
        tokens = encoder.encode_observed(frames.unsqueeze(0), None)  # [1, T, D]
        torch.save(
            {
                "key": key,
                "tracklet_id": m.tracklet_id,
                "vessel_id": m.vessel_id,
                "split": m.split,
                "frame_indices": idx,
                "arm": arm,
                "features": tokens[0],
            },
            out_path,
        )
        if (cached + skipped) % 50 == 0:
            print(f"  processed {cached + skipped} / {len(manifests)}")
    print(f"done: {cached} cached, {skipped} skipped, features in {features_dir}")


if __name__ == "__main__":
    main()
