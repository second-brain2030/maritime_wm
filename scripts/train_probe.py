#!/usr/bin/env python3
"""Train a shared Re-ID probe on cached frozen features (spec §21; brief P3)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import load_manifests
from training.probe import train_probe
from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="mvtd_cnn")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--token-dim", type=int, default=None)
    ap.add_argument("--max-tracklets", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("experiments", args.config_name)
    manifests = load_manifests(cfg["data"]["manifest_path"])
    arm = cfg["model"]["arm"]
    features_dir = args.features_dir or str(Path(cfg.get("features_dir", "outputs/features")) / arm)
    token_dim = args.token_dim or int(
        cfg.get("model", {}).get("embedding_dim", 2048)
    )
    out = Path(args.output or f"outputs/probes/{arm}/probe.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    artifacts = train_probe(
        features_dir=features_dir,
        manifests=manifests,
        token_dim=token_dim,
        epochs=args.epochs or int(cfg["training"]["epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        lr=float(cfg["training"]["lr"]),
        id_ce_weight=float(cfg["training"]["losses"]["id_ce_weight"]),
        triplet_weight=float(cfg["training"]["losses"]["triplet_weight"]),
        seed=args.seed or int(cfg["experiment"]["seed"]),
        max_tracklets=args.max_tracklets,
    )
    artifacts.save(out)
    print(f"saved probe -> {out}")
    print(f"classes={len(artifacts.class_map)} epochs={artifacts.config['epochs_run']} "
          f"final_loss={artifacts.config['loss_history'][-1]:.4f}")


if __name__ == "__main__":
    main()
