#!/usr/bin/env python3
"""Prepare a dataset: validate splits and write normalized manifests (spec §21).

Also persists adapter auxiliary manifests (e.g., FVessel AIS trajectories and
camera meta) via config ``aux_manifest_paths`` when the adapter provides them.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import save_manifests
from data.adapters import get_adapter
from utils.config import load_config


def _save_aux(adapter, cfg: dict, base_dir: Path) -> None:
    aux = getattr(adapter, "aux_manifests", None)
    if not aux:
        return
    paths = cfg.get("aux_manifest_paths", {})
    for name, items in aux.items():
        out = paths.get(name)
        if not out or not items:
            if out:
                print(f"[warn] no {name} items to write to {out}")
            continue
        out = base_dir / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
        print(f"wrote {len(items)} {name} entries to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="viv_reid")
    args = ap.parse_args()

    cfg = load_config("data", args.config_name)
    print("resolved data config:")
    print(json.dumps(cfg, indent=2, default=str))

    adapter = get_adapter(args.config_name, cfg)
    manifests = adapter.build_manifests()
    out = Path(cfg.get("manifest_path", "data/manifests/manifest.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    save_manifests(str(out), manifests)
    print(f"wrote {len(manifests)} tracklets to {out}")
    _save_aux(adapter, cfg, Path("."))


if __name__ == "__main__":
    main()
