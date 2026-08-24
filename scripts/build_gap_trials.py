#!/usr/bin/env python3
"""Build disappearance-gap trials (spec §4.4 / §7.1).

Reads a normalized tracklet manifest, applies the DGRA protocol config, and
writes a deterministic gap-trials JSONL. Distractor-pool construction needs
the neutral-reference feature store and is a later commit; trials without a
pool carry pool_size 0 and are excluded from pool-based analyses.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.gap_trials import GapProtocolConfig, GapTrialManifest, build_gap_trials
from data.manifest import load_manifests
from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="viv_reid_dgra")
    ap.add_argument("--manifest", default="data/manifests/viv_reid.jsonl")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config("gap", args.config_name)
    config = GapProtocolConfig.from_dict(cfg.get("gap_protocol", {}))
    if args.seed is not None:
        config.seed = args.seed
    config.validate()

    manifests = load_manifests(args.manifest)
    print(f"loaded {len(manifests)} tracklets from {args.manifest}")

    trials = build_gap_trials(manifests, config)
    manifest = GapTrialManifest(trials)
    out = Path(args.output or f"data/gap_trials/{args.config_name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(str(out))

    summary = manifest.summary()
    print(f"wrote {len(manifest)} trials to {out}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
