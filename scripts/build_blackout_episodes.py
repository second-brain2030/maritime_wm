#!/usr/bin/env python3
"""Build sensor-blackout episodes from tracklet + AIS manifests (brief P1).

Writes <output>.jsonl plus a <output>.config.json sidecar so evaluation and
baseline runs can reproduce the exact AIS jitter/dropout per episode.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.ais import AisTrajectoryManifest
from data.manifest import load_manifests
from evaluation.blackout_harness import (
    BlackoutConfig,
    BlackoutEpisodeManifest,
    build_blackout_episodes,
)
from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="fvessel_blackout")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--ais-manifest", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config("blackout", args.config_name) if args.config_name else {}
    bcfg = BlackoutConfig.from_dict(cfg.get("blackout", {}))
    bcfg.validate()

    manifest_path = args.manifest or cfg.get("manifest_path", "data/manifests/fvessel.jsonl")
    manifests = load_manifests(manifest_path)
    ais_by = {}
    ais_path = args.ais_manifest or cfg.get("ais_manifest_path")
    if ais_path and Path(ais_path).is_file():
        ais_by = {t.vessel_id: t for t in AisTrajectoryManifest.load(ais_path)}
    print(f"loaded {len(manifests)} tracklets, {len(ais_by)} AIS trajectories")

    episodes = build_blackout_episodes(manifests, ais_by, config=bcfg)
    out = Path(args.output or cfg.get("episodes_path", "data/gap_trials/blackout.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    BlackoutEpisodeManifest(episodes).save(str(out))
    (out.with_suffix(out.suffix + ".config.json")).write_text(
        json.dumps(bcfg.to_dict() if hasattr(bcfg, "to_dict") else vars(bcfg), indent=2)
    )
    summary = BlackoutEpisodeManifest(episodes).summary()
    print(f"wrote {len(episodes)} episodes to {out}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
