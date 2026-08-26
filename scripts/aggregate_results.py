#!/usr/bin/env python3
"""Aggregate re-acquisition results into a per-duration arm comparison table.

Reads outputs/eval/<arm>/reacquisition.json and
outputs/baselines/<baseline>/reacquisition.json (identical schema), prints a
per-duration Top-1 comparison across arms with degradation slopes, and writes
outputs/pilot/comparison.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_results(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _load_episode_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _slice_summary(sliced: list[dict], arm_name: str) -> tuple[dict, float | None, int]:
    """Recompute per-duration Top-1 and degradation slope for a sliced set."""
    from evaluation.degradation import compute_degradation
    from evaluation.tracking_metrics import reacquisition_topk

    by_dur: dict[float, list[dict]] = {}
    for r in sliced:
        by_dur.setdefault(float(r["duration_s"]), []).append(r)
    per_duration: dict[str, float] = {}
    by_bin: dict[str, list[bool]] = {}
    pools: list[int] = []
    for d in sorted(by_dur):
        rs = by_dur[d]
        ranks = [r.get("rank_of_correct") for r in rs]
        per_duration[f"{int(d)}s"] = reacquisition_topk(ranks, 1)
        by_bin[f"{int(d)}s"] = [bool(r.get("rank_of_correct") == 1) for r in rs]
        pools.extend(r.get("n_candidates", 0) for r in rs)
    slope = None
    if by_bin:
        pool_size = max(1, int(round(sum(pools) / len(pools))))
        if pool_size >= 2:
            slope = compute_degradation(by_bin, pool_size=pool_size, arm_name=arm_name).slope
    return per_duration, slope, len(sliced)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="outputs/eval/<arm> or outputs/baselines/<name> dirs "
                         "(default: discover outputs/eval/* and outputs/baselines/*)")
    ap.add_argument("--output", default="outputs/pilot/comparison.json")
    ap.add_argument("--min-pool", type=int, default=None,
                    help="slice: keep only episodes with n_candidates >= N "
                         "(recomputed from per-episode results.jsonl)")
    ap.add_argument("--maneuver", choices=("straight", "maneuver", "unknown"), default=None,
                    help="slice: keep only episodes labeled with this "
                         "maneuver-during-gap value (labels file by dataset)")
    ap.add_argument("--maneuver-labels", default=None,
                    help="episode_id -> maneuver JSONL (default: "
                         "data/gap_trials/<stem>_maneuver.jsonl next to --episodes)")
    args = ap.parse_args()

    maneuver_labels: dict[str, str] | None = None
    if args.maneuver is not None:
        labels_path = Path(args.maneuver_labels or "data/gap_trials/fvessel_blackout_maneuver.jsonl")
        if not labels_path.is_file():
            raise FileNotFoundError(f"maneuver labels not found: {labels_path} "
                                    "(run scripts/label_maneuvers.py first)")
        maneuver_labels = {}
        for line in labels_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    maneuver_labels[d["episode_id"]] = d["maneuver"]
                except json.JSONDecodeError:
                    continue

    dirs: list[Path] = []
    if args.runs:
        dirs = [Path(r) for r in args.runs]
    else:
        for base in (Path("outputs/eval"), Path("outputs/baselines")):
            if base.is_dir():
                dirs.extend(sorted(d for d in base.iterdir() if d.is_dir()))

    rows: list[dict] = []
    for d in dirs:
        res = _load_results(d / "reacquisition.json")
        if res is None:
            continue
        name = res.get("arm") or res.get("baseline") or d.name
        if args.min_pool is not None or args.maneuver is not None:
            # recompute per-duration Top-1 / slope from the sliced episode set
            ep_rows = _load_episode_results(d / "reacquisition.results.jsonl")
            sliced = ep_rows
            if args.min_pool is not None:
                sliced = [r for r in sliced if r.get("n_candidates", 0) >= args.min_pool]
            if args.maneuver is not None:
                sliced = [
                    r for r in sliced
                    if maneuver_labels.get(r.get("episode_id")) == args.maneuver
                ]
            per_duration, slope, n_ep = _slice_summary(sliced, name)
            dataset = res.get("dataset", "unknown")
            if not sliced:
                continue
        else:
            per_duration = {
                f"{int(r['duration_s'])}s": r["top1"]
                for r in res["summary"].get("per_duration", [])
            }
            slope = res.get("degradation_slope")
            n_ep = res.get("n_episodes")
            dataset = res.get("dataset", "unknown")
        rows.append({
            "arm": name,
            "dataset": dataset,
            "top1_by_duration": per_duration,
            "self_cosine_by_duration": res.get("self_cosine_by_duration"),
            "degradation_slope": slope,
            "n_episodes": n_ep,
            "source": str(d),
            "slice": f"pool>={args.min_pool}" if args.min_pool is not None else "all",
        })

    rows.sort(key=lambda r: (r["dataset"], r["arm"]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arms": rows}, indent=2, default=str))

    datasets = sorted({r["dataset"] for r in rows})
    for dataset in datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        durations = sorted(
            {int(d[:-1]) for r in ds_rows for d in r["top1_by_duration"]},
            key=lambda d: d,
        )
        print(f"\n=== dataset: {dataset} ===")
        header = f"{'arm':<22} " + " ".join(f"{d}s".rjust(6) for d in durations) + "  slope"
        print(header)
        for r in ds_rows:
            t = r["top1_by_duration"]
            cells = " ".join(f"{t.get(f'{d}s', float('nan')):>6.2f}" for d in durations)
            slope = f"{r['degradation_slope']:.3f}" if r["degradation_slope"] is not None else "  n/a"
            print(f"{r['arm']:<22} {cells}  {slope}")
    print(f"\n-> {out} ({len(rows)} arms, {len(datasets)} datasets)")


if __name__ == "__main__":
    main()
