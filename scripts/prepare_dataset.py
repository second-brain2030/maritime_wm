#!/usr/bin/env python3
"""Prepare a dataset: validate splits and write the normalized manifest (spec §21)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import save_manifests
from data.adapters import get_adapter
from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="viv_reid")
    args = ap.parse_args()

    cfg = load_config("data", args.config_name)
    print("resolved data config:")
    print(json.dumps(cfg, indent=2, default=str))

    adapter = get_adapter(args.config_name, cfg)  # NotImplementedError until adapters land
    manifests = adapter.build_manifests()
    out = Path(cfg.get("manifest_path", "data/manifests/manifest.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    save_manifests(str(out), manifests)
    print(f"wrote {len(manifests)} tracklets to {out}")


if __name__ == "__main__":
    main()
