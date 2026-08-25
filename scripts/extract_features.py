#!/usr/bin/env python3
"""Extract frozen-backbone tracklet features (spec section 13 / §21).

Hydra entry point::

    python scripts/extract_features.py --config-name experiments/vivreid_cnn

Loads the arm encoder from ``cfg.model.arm`` (only ``cnn_reid`` is
implemented; other arms raise ``NotImplementedError``), samples
``cfg.data.frames_per_tracklet`` frames per tracklet, encodes every tracklet
and caches the features to ``outputs/feature_cache/<key>.pt``. The cache key
covers the encoder checkpoint, preprocessing config, frame sampling config and
the manifest content hash, so re-runs with identical inputs are no-ops. The
fully resolved config is dumped to ``outputs/extract_<exp>_<ts>/``.
"""
from __future__ import annotations

import os
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
from torchvision.transforms import ToTensor

from src.data.manifest import load_manifests
from src.data.sampling import sample_frame_indices
from src.models.cnn_reid import CNNReIDEncoder
from src.training.callbacks import FeatureCache
from src.utils.media import load_frame
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


@hydra.main(config_path="../configs", config_name="experiments/vivreid_cnn", version_base=None)
def main(cfg) -> None:
    orig = _original_cwd()
    os.chdir(orig)
    seed_everything(_seed(cfg))

    arm = OmegaConf.select(cfg, "model.arm", default="cnn_reid")
    if arm != "cnn_reid":
        raise NotImplementedError(
            f"feature extraction for arm={arm!r} is not implemented; only "
            "'cnn_reid' is supported (V-JEPA / OpenVLA arms land in later commits)"
        )

    run_dir = _run_dir(orig, f"extract_{_encoder_name(cfg).replace('/', '_')}")
    _dump_resolved_config(cfg, run_dir)

    backbone = OmegaConf.select(cfg, "model.backbone", default="resnet50")
    pretrained = bool(OmegaConf.select(cfg, "model.pretrained", default=True))
    device = OmegaConf.select(
        cfg, "device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    encoder = CNNReIDEncoder(backbone=backbone, pretrained=pretrained, device=device)
    print(
        f"arm={arm} backbone={encoder.backbone_name} dim={encoder.embedding_dim} "
        f"device={device}"
    )

    dataset = OmegaConf.select(cfg, "data.dataset", default="viv_reid")
    manifest_path = OmegaConf.select(cfg, "data.manifest_path", default=f"data/manifests/{dataset}.jsonl")
    manifests = load_manifests(str(orig / manifest_path))
    manifest_hash = _manifest_hash(manifests)
    print(f"loaded {len(manifests)} tracklets from {orig / manifest_path}")

    cache = FeatureCache(orig / "outputs" / "feature_cache")
    key = _feature_cache_key(cfg, cache, manifest_hash)
    cache_file = cache._path(key)  # noqa: SLF001 - documented helper

    if cache.exists(key):
        print(f"cache hit -> outputs/feature_cache/{key}.pt; skipping extraction")
        features = cache.load(key)
    else:
        frames_per = _sample_cfg(cfg)["frames_per_tracklet"]
        mode = _sample_cfg(cfg)["sample_mode"]
        short_policy = _sample_cfg(cfg)["short_tracklet_policy"]
        to_tensor = ToTensor()
        features: dict[str, torch.Tensor] = {}
        for m in manifests:
            idx = sample_frame_indices(
                len(m.frame_paths),
                frames_per,
                mode=mode,
                short_policy=short_policy,
            )
            frames = []
            for i in idx:
                img = load_frame(m.frame_paths[i], fps=m.fps)
                frames.append(to_tensor(img))
            frames_t = torch.stack(frames).unsqueeze(0).to(device)  # [1, T, C, H, W]
            frames_t = encoder.preprocess(frames_t)
            tokens = encoder.encode_observed(frames_t)  # [1, T, D]
            features[m.tracklet_id] = tokens[0].detach().cpu()
        cache.save(key, features)
        print(f"cached {len(features)} tracklet feature tensors -> {cache_file}")

    first = next(iter(features.values()))
    print(f"feature tensor shape: {tuple(first.shape)}")
    print(f"cache key: {key}")
    print(f"cache file: outputs/feature_cache/{key}.pt")


if __name__ == "__main__":
    main()
