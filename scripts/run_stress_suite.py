#!/usr/bin/env python3
"""Run the deterministic diagnostic stress suite (spec §21, §8.4)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="vivreid_vjepa_encoder")
    args = ap.parse_args()

    cfg = load_config("experiments", args.config_name)
    print("resolved experiment config:")
    print(json.dumps(cfg, indent=2, default=str))
    raise NotImplementedError("stress suite lands in a later commit")


if __name__ == "__main__":
    main()
