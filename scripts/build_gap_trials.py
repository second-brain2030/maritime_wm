#!/usr/bin/env python3
"""Build disappearance-gap trials (spec section 4.4 / §7.1).

Hydra entry point::

    python scripts/build_gap_trials.py --config-name viv_reid_dgra
    python scripts/build_gap_trials.py --config-name fvessel_blackout

Reads a normalized tracklet manifest, applies the DGRA protocol config, and
writes a deterministic gap-trials JSONL plus a distractor-pool manifest.
Distractor-pool construction needs the neutral-reference feature store and is
a later commit; trials therefore carry ``pool_size 0`` (excluded from
pool-based analyses) and an empty ``DistractorPoolManifest`` records that
limitation. When ``cfg.dataset == "fvessel"`` the script additionally builds
``BlackoutHarness`` episodes and saves them to
``data/gap_trials/fvessel_blackout.jsonl``. The fully resolved config is
dumped to ``outputs/gap_<config_name>_<ts>/resolved_config.yaml``.
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
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.data.distractor_pool import DistractorPoolManifest
from src.data.gap_trials import GapProtocolConfig, GapTrialManifest, build_gap_trials
from src.data.manifest import load_manifests
from src.evaluation.blackout_harness import BlackoutHarness
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


@hydra.main(config_path="../configs", config_name="gap/viv_reid_dgra", version_base=None)
def main(cfg) -> None:
    orig = _original_cwd()
    os.chdir(orig)
    seed_everything(_seed(cfg))

    config_name = (HydraConfig.get().job.config_name or "gap/viv_reid_dgra").split("/")[-1]
    dataset = OmegaConf.select(cfg, "dataset", default="viv_reid")
    manifest_path = _path_opt(cfg, "manifest_path", "data/manifests/viv_reid.jsonl")
    run_dir = _run_dir(orig, f"gap_{config_name}")
    _dump_resolved_config(cfg, run_dir)

    manifests = load_manifests(str(orig / manifest_path))
    print(f"loaded {len(manifests)} tracklets from {orig / manifest_path}")

    gap_cfg_dict = OmegaConf.to_container(OmegaConf.select(cfg, "gap_protocol", default={}), resolve=True)
    gap_config = GapProtocolConfig.from_dict(gap_cfg_dict)
    gap_config.validate()
    print(f"gap protocol: seed={gap_config.seed} bins_seconds={gap_config.gap_bins_seconds}")

    # --- DGRA trials (natural cross-camera + synthetic within-tracklet) ---
    trials = build_gap_trials(manifests, gap_config)
    trial_manifest = GapTrialManifest(trials)
    trials_out = orig / _path_opt(cfg, "trials_out", f"data/gap_trials/{config_name}.jsonl")
    trials_out.parent.mkdir(parents=True, exist_ok=True)
    trial_manifest.save(str(trials_out))
    print(f"wrote {len(trial_manifest)} gap trials -> {trials_out}")
    print(json.dumps(trial_manifest.summary(), indent=2, sort_keys=True, default=str))

    # --- Distractor pools (empty until the neutral-reference feature store lands) ---
    pools = DistractorPoolManifest()
    pools_out = orig / _path_opt(cfg, "pools_out", f"data/gap_trials/{config_name}_pools.jsonl")
    pools_out.parent.mkdir(parents=True, exist_ok=True)
    pools.save(str(pools_out))
    print(
        f"wrote {len(pools)} distractor pools -> {pools_out} "
        "(empty: pool construction requires the neutral-reference feature store; "
        "trials carry pool_size=0 and are excluded from pool-based analyses)"
    )

    # --- Blackout episodes (FVessel only) ---
    if dataset == "fvessel":
        harness = BlackoutHarness(manifests, gap_cfg_dict, seed=_seed(cfg))
        episodes = harness.build()
        episodes_out = orig / _path_opt(cfg, "episodes_out", "data/gap_trials/fvessel_blackout.jsonl")
        harness.save_episodes(episodes, episodes_out)
        from collections import Counter

        print(
            f"wrote {len(episodes)} blackout episodes -> {episodes_out} "
            f"by_duration={dict(Counter(e.blackout_seconds for e in episodes))}"
        )

    # --- Content-addressed hash over trials + pools ---
    hash_parts = {
        "trials": [t.to_dict() for t in trial_manifest],
        "pools": [p.to_dict() for p in pools],
        "manifest_hash": content_addressed_key(
            {"manifests": [m.fingerprint() for m in sorted(manifests, key=lambda m: m.tracklet_id)]}
        ),
    }
    hash_path = orig / "data" / "gap_trials" / "gap_trials_hash.txt"
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(content_addressed_key(hash_parts), encoding="utf-8")
    print(f"wrote gap-trials hash -> {hash_path}")


if __name__ == "__main__":
    main()
