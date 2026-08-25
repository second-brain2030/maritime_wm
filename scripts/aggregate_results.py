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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="outputs/eval/<arm> or outputs/baselines/<name> dirs "
                         "(default: discover outputs/eval/* and outputs/baselines/*)")
    ap.add_argument("--output", default="outputs/pilot/comparison.json")
    args = ap.parse_args()

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
        per_duration = {
            f"{int(r['duration_s'])}s": r["top1"]
            for r in res["summary"].get("per_duration", [])
        }
        rows.append({
            "arm": name,
            "top1_by_duration": per_duration,
            "degradation_slope": res.get("degradation_slope"),
            "n_episodes": res.get("n_episodes"),
            "source": str(d),
        })

    rows.sort(key=lambda r: r["arm"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arms": rows}, indent=2, default=str))

    durations = sorted(
        {int(d[:-1]) for r in rows for d in r["top1_by_duration"]},
        key=lambda d: d,
    )
    header = f"{'arm':<22} " + " ".join(f"{d}s".rjust(6) for d in durations) + "  slope"
    print(header)
    for r in rows:
        t = r["top1_by_duration"]
        cells = " ".join(f"{t.get(f'{d}s', float('nan')):>6.2f}" for d in durations)
        slope = f"{r['degradation_slope']:.3f}" if r["degradation_slope"] is not None else "  n/a"
        print(f"{r['arm']:<22} {cells}  {slope}")
    print(f"\n-> {out} ({len(rows)} arms)")


if __name__ == "__main__":
    main()
