#!/usr/bin/env python3
"""Aggregate experiment outputs into a comparison report (spec §21, §15)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="outputs/<run_id> dirs")
    args = ap.parse_args()

    for run in args.runs:
        path = Path(run)
        if not path.is_dir():
            raise FileNotFoundError(f"run directory not found: {path}")
        print(f"collecting {path}")
    raise NotImplementedError("aggregation and report generation land in a later commit")


if __name__ == "__main__":
    main()
