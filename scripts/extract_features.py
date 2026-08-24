#!/usr/bin/env python3
"""Extract frozen-backbone tracklet features (spec §21)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="vivreid_vjepa_encoder")
    args = ap.parse_args()

    cfg = load_config("experiments", args.config_name)
    print("resolved experiment config:")
    print(json.dumps(cfg, indent=2, default=str))
    raise NotImplementedError(
        "feature extraction lands in a later commit; the resolved config above "
        "is the contract (content-addressed caching per spec §13)"
    )


if __name__ == "__main__":
    main()
