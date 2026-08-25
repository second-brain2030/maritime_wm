#!/usr/bin/env python3
"""Train a shared Re-ID probe on frozen backbone features (spec section 12 / §21).

Hydra entry point::

    python scripts/train_probe.py --config-name experiments/vivreid_cnn

Loads the feature cache written by ``extract_features.py`` (same content
addressed key), builds a ``SharedReIDHead`` + ``CombinedLoss`` + ``ProbeTrainer``
and runs ``trainer.fit(...)``. Because every tracklet is already cached, the
probe never re-runs the frozen encoder: a thin dataset serves the cached token
tensors through ``ProbeTrainer``'s FeatureCache path. The best checkpoint is
saved to ``outputs/train_<exp>_<ts>/best_probe.pt`` together with
``resolved_config.yaml`` and per-epoch metrics.
"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import torch
import yaml
from omegaconf import OmegaConf

from src.data.manifest import load_manifests
from src.evaluation.reid_metrics import compute_metrics
from src.models.common_head import SharedReIDHead
from src.training.callbacks import FeatureCache
from src.training.losses import CombinedLoss
from src.training.trainer import ProbeTrainer
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


def _select(obj, key: str, default=None):
    """OmegaConf.select that also accepts plain dicts (config subsets)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return OmegaConf.select(obj, key, default=default)


def _manifest_hash(manifests) -> str:
    parts = {"manifests": [m.fingerprint() for m in sorted(manifests, key=lambda m: m.tracklet_id)]}
    return content_addressed_key(parts)


def _encoder_name(cfg) -> str:
    backbone = OmegaConf.select(cfg, "model.backbone", default="resnet50")
    pretrained = bool(OmegaConf.select(cfg, "model.pretrained", default=True))
    return f"cnn_reid/{backbone}/pretrained={pretrained}"


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


def _preprocess_cfg(cfg) -> dict:
    return {"input_resolution": int(OmegaConf.select(cfg, "data.input_resolution", default=224))}


def _feature_cache_key(cfg, cache: FeatureCache, manifest_hash: str) -> str:
    return cache._cache_key(  # noqa: SLF001 - documented helper of FeatureCache
        _encoder_name(cfg), _preprocess_cfg(cfg), _sample_cfg(cfg), manifest_hash
    )


class _CachedFeatureDataset(torch.utils.data.Dataset):
    """Serve cached token features through ProbeTrainer's batch contract.

    Every returned tracklet id is present in the FeatureCache, so
    ``ProbeTrainer._get_tokens`` reads tokens from the cache and never invokes
    the frozen encoder; the dummy ``frames`` tensor only keeps the batch
    contract shape ``[B, T, C, H, W]``.
    """

    def __init__(
        self,
        features,
        manifests,
        split: str,
        frames_per_tracklet: int,
        label_map: dict,
        manifest_hash: str,
        preprocess_cfg: dict,
        sample_cfg: dict,
    ) -> None:
        self.T = int(frames_per_tracklet)
        self.label_map = label_map
        # ProbeTrainer._feature_key derives the cache key from these dataset
        # attributes; they must match extract_features.py's inputs so the
        # cached token tensors are found.
        self.manifest_hash = manifest_hash
        self.preprocess_cfg = preprocess_cfg
        self.sample_cfg = sample_cfg
        self.items = [
            (m.tracklet_id, m.vessel_id)
            for m in manifests
            if m.split == split and m.tracklet_id in features
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        tracklet_id, vessel_id = self.items[i]
        return {
            "frames": torch.zeros(self.T, 3, 224, 224),
            "labels": torch.tensor(self.label_map[vessel_id], dtype=torch.long),
            "tracklet_ids": tracklet_id,
        }


class _StubEncoder:
    """Placeholder encoder for ProbeTrainer when all features are cached.

    ``ProbeTrainer._feature_key`` reads ``encoder.name`` to derive the cache
    key, but ``encode_observed`` is never invoked while every batch is a cache
    hit; this stub carries the same name extract_features.py uses, so keys
    match and the frozen backbone is never loaded.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.embedding_dim = 0

    def encode_observed(self, frames, frame_mask=None):
        raise RuntimeError(
            "cached-feature probe must not invoke the encoder; "
            "run scripts/extract_features.py with the same config first"
        )


def _val_metric(embeddings: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Proxy validation metric over all val embeddings (self-retrieval mAP)."""
    embs = embeddings.detach().cpu().numpy()
    labs = labels.detach().cpu().numpy().tolist()
    m = compute_metrics(embs, embs, labs, labs)
    return {"mAP": float(m["mAP"]), "rank1": float(m[1])}


@hydra.main(config_path="../configs", config_name="experiments/vivreid_cnn", version_base=None)
def main(cfg) -> None:
    orig = _original_cwd()
    os.chdir(orig)
    seed_everything(_seed(cfg))

    arm = OmegaConf.select(cfg, "model.arm", default="cnn_reid")
    if arm != "cnn_reid":
        raise NotImplementedError(
            f"probe training for arm={arm!r} is not implemented; only 'cnn_reid' "
            "features (produced by extract_features.py) are supported"
        )

    exp_name = OmegaConf.select(cfg, "experiment.name", default="experiment")
    run_dir = _run_dir(orig, f"train_{exp_name}")
    _dump_resolved_config(cfg, run_dir)

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
    print(f"loaded {len(features)} cached tracklet features (key={key})")

    first = next(iter(features.values()))
    input_dim = int(first.shape[-1])
    label_map = {v: i for i, v in enumerate(sorted({m.vessel_id for m in manifests}))}
    num_classes = len(label_map)
    print(f"input_dim={input_dim} num_classes={num_classes}")

    pooler = OmegaConf.select(cfg, "model.temporal_head", default="attention")
    head = SharedReIDHead(input_dim=input_dim, num_classes=num_classes, pooler=pooler)

    losses_cfg = OmegaConf.select(cfg, "training.losses", default={})
    loss_fn = CombinedLoss(
        num_classes=num_classes,
        ce_weight=float(_select(losses_cfg, "id_ce_weight", 1.0)),
        triplet_weight=float(_select(losses_cfg, "triplet_weight", 1.0)),
    )

    training_cfg = OmegaConf.select(cfg, "training", default={})
    optimizer_cfg = {
        "name": str(_select(training_cfg, "optimizer", "adamw")),
        "lr": float(_select(training_cfg, "lr", 3e-4)),
    }
    scheduler_cfg = {
        "warmup_epochs": int(_select(training_cfg, "warmup_epochs", 5)),
    }
    device = OmegaConf.select(cfg, "device", default="cuda" if torch.cuda.is_available() else "cpu")

    trainer = ProbeTrainer(
        head=head,
        encoder=_StubEncoder(_encoder_name(cfg)),  # cache-hit path only
        loss_fn=loss_fn,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        cache=cache,
        device=device,
        seed=_seed(cfg),
    )

    frames_per = _sample_cfg(cfg)["frames_per_tracklet"]
    preprocess_cfg = _preprocess_cfg(cfg)
    sample_cfg = _sample_cfg(cfg)
    train_ds = _CachedFeatureDataset(
        features, manifests, "train", frames_per, label_map, manifest_hash, preprocess_cfg, sample_cfg
    )
    val_ds = _CachedFeatureDataset(
        features, manifests, "query", frames_per, label_map, manifest_hash, preprocess_cfg, sample_cfg
    )
    if len(train_ds) == 0:
        raise RuntimeError("no cached train tracklets to train on")
    if len(val_ds) == 0:
        print("[warn] no cached query tracklets for validation; falling back to gallery split")
        val_ds = _CachedFeatureDataset(
            features, manifests, "gallery", frames_per, label_map, manifest_hash, preprocess_cfg, sample_cfg
        )

    batch_size = int(_select(training_cfg, "batch_size", 32))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    num_epochs = int(_select(training_cfg, "epochs", 60))
    best_path = trainer.fit(
        train_loader,
        val_loader,
        _val_metric,
        num_epochs=num_epochs,
        checkpoint_dir=run_dir,
    )
    best_probe = run_dir / "best_probe.pt"
    shutil.copy2(best_path, best_probe)
    print(f"best checkpoint -> {best_probe} (from {best_path})")

    final_val = trainer.evaluate(val_loader, _val_metric)
    print("final val metrics:", yaml.dump(final_val, sort_keys=False))


if __name__ == "__main__":
    main()
