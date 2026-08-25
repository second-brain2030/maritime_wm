#!/usr/bin/env python3
"""Prepare a dataset: build normalized tracklet manifests and validate split
hygiene (spec section 4 / §12).

Hydra entry point::

    python scripts/prepare_dataset.py --config-name viv_reid
    python scripts/prepare_dataset.py --config-name fvessel

Loads the dataset adapter registered under ``cfg.dataset``, runs it (via
``run()`` when the adapter exposes it, else ``build_manifests()`` +
``save_manifests``), validates that train identities do not leak into
query/gallery/test, persists adapter auxiliary manifests (e.g. FVessel AIS
trajectories and camera meta), writes ``data/manifests/dataset_manifest_hash.txt``
and prints per-split tracklet/identity counts. The fully resolved config is
dumped to ``outputs/prepare_<dataset>_<ts>/resolved_config.yaml``.
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
from omegaconf import OmegaConf

from src.data.adapters import get_adapter
from src.data.manifest import load_manifests, save_manifests
from src.data.splits import identity_sets, validate_identity_disjointness
from src.utils.reproducibility import content_addressed_key, seed_everything


def _original_cwd() -> Path:
    """Repo root: Hydra chdirs to the run dir, so resolve everything here."""
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
    """experiment.seed when present (experiment configs), else gap/data seed."""
    seed = OmegaConf.select(cfg, "experiment.seed", default=None)
    if seed is None:
        seed = OmegaConf.select(cfg, "gap_protocol.seed", default=None)
    if seed is None:
        seed = OmegaConf.select(cfg, "layout.split_seed", default=42)
    return int(seed)


def _save_aux(adapter, cfg: dict, base_dir: Path) -> None:
    """Persist adapter auxiliary manifests (FVessel AIS / camera meta)."""
    aux = getattr(adapter, "aux_manifests", None)
    if not aux:
        return
    paths = cfg.get("aux_manifest_paths", {})
    for name, items in aux.items():
        out_rel = paths.get(name)
        if not out_rel:
            continue
        out = base_dir / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), sort_keys=True, default=str) + "\n")
        print(f"wrote {len(items)} {name} entries to {out}")


def _manifest_hash(manifests) -> str:
    """Content-addressed hash over sorted per-tracklet fingerprints."""
    parts = {"manifests": [m.fingerprint() for m in sorted(manifests, key=lambda m: m.tracklet_id)]}
    return content_addressed_key(parts)


@hydra.main(config_path="../configs", config_name="data/viv_reid", version_base=None)
def main(cfg) -> None:
    orig = _original_cwd()
    os.chdir(orig)  # keep relative paths (data/, outputs/) repo-root based
    seed_everything(_seed(cfg))

    dataset = OmegaConf.select(cfg, "dataset", default=None) or OmegaConf.select(
        cfg, "data.dataset", default=None
    )
    if not dataset:
        raise ValueError("config must declare the dataset name (top-level 'dataset')")

    run_dir = _run_dir(orig, f"prepare_{dataset}")
    _dump_resolved_config(cfg, run_dir)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    adapter = get_adapter(dataset, cfg_dict)
    print(f"dataset={dataset} adapter={type(adapter).__name__}")

    out_rel = OmegaConf.select(cfg, "manifest_path", default=f"data/manifests/{dataset}.jsonl")
    out = orig / out_rel
    run = getattr(adapter, "run", None)
    if callable(run):
        # Adapters with a run() contract save the manifest themselves.
        run_out = Path(run())
        run_path = run_out if run_out.is_absolute() else orig / run_out
        manifests = load_manifests(str(run_path))
        if str(run_path.resolve()) != str(out.resolve()):
            print(f"[note] adapter wrote manifests to {run_path}; expected {out}")
        out = Path(run_path)
    else:
        manifests = adapter.build_manifests()
        out.parent.mkdir(parents=True, exist_ok=True)
        save_manifests(str(out), manifests)

    print(f"built {len(manifests)} tracklets -> {out}")

    # Split hygiene: train identities must not leak into query/gallery/test.
    report = validate_identity_disjointness(manifests)
    sets = identity_sets(manifests)
    test_ids = {m.vessel_id for m in manifests if m.split == "test"}
    train_test_overlap = sorted(sets["train"] & test_ids)
    if train_test_overlap:
        raise ValueError(
            "train identities must not appear in test; overlapping identities: "
            + ", ".join(train_test_overlap)
        )
    report["train_test_overlap"] = train_test_overlap
    report["test_identity_count"] = len(test_ids)

    _save_aux(adapter, cfg_dict, orig)

    # Per-split tracklet + identity counts.
    by_split: dict[str, dict[str, int]] = {}
    for split in sorted({m.split for m in manifests}):
        members = [m for m in manifests if m.split == split]
        by_split[split] = {
            "tracklets": len(members),
            "identities": len({m.vessel_id for m in members}),
        }
    print("per-split counts:")
    print(json.dumps(by_split, indent=2, sort_keys=True))
    print("split hygiene:", json.dumps(report, indent=2, sort_keys=True))

    hash_path = orig / "data" / "manifests" / "dataset_manifest_hash.txt"
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(_manifest_hash(manifests), encoding="utf-8")
    print(f"wrote manifest hash -> {hash_path}")


if __name__ == "__main__":
    main()
