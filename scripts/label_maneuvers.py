#!/usr/bin/env python3
"""Label blackout episodes with maneuver-during-gap (straight|maneuver|unknown).

Writes a JSONL keyed by episode_id; aggregate_results can then slice arms by
this axis (e.g. --maneuver maneuver) without re-evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import load_manifests
from evaluation.blackout_harness import BlackoutEpisodeManifest
from evaluation.maneuver import label_episodes
from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="fvessel_blackout")
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config("blackout", args.config_name)
    manifests = load_manifests(cfg.get("manifest_path", "data/manifests/fvessel.jsonl"))
    tracklet_map = {(m.camera_id, m.vessel_id): m for m in manifests}
    episodes_path = args.episodes or cfg.get("episodes_path", "data/gap_trials/fvessel_blackout.jsonl")
    episodes = BlackoutEpisodeManifest.load(episodes_path)

    labels = label_episodes(episodes, tracklet_map)
    out = Path(args.output or f"data/gap_trials/{Path(episodes_path).stem}_maneuver.jsonl")
    with open(out, "w") as f:
        for eid, lab in sorted(labels.items()):
            f.write(json.dumps({"episode_id": eid, "maneuver": lab}) + "\n")

    from collections import Counter

    print(f"labeled {len(labels)} episodes -> {out}")
    print("distribution:", dict(Counter(labels.values())))


if __name__ == "__main__":
    main()