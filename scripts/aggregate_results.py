#!/usr/bin/env python3
"""Aggregate experiment outputs into a comparison report (spec section 15 / §21).

Interface (deliberately NOT Hydra — the task defines a ``--runs`` glob
interface, e.g. ``outputs/fvessel_*``; this is the single documented exception
to the Hydra convention used by the other scripts)::

    python scripts/aggregate_results.py --runs "outputs/train_vivreid_cnn_*"

Collects ``metrics.json`` from each run directory, computes cross-run
(per-seed) mean / std / 95% CI per metric, prints a summary table, writes
``outputs/aggregate_<timestamp>.csv``, and — when any run contains a
``degradation_curves.csv`` — renders a matplotlib figure of the degradation
curves to ``outputs/degradation_curves_<timestamp>.png``.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

METRIC_KEYS = ["mAP", 1, 5, 10]


def _collect_runs(patterns: list[str]) -> list[Path]:
    runs: list[Path] = []
    for pattern in patterns:
        for path_str in sorted(glob.glob(pattern)):
            p = Path(path_str)
            if p.is_dir() and (p / "metrics.json").is_file():
                runs.append(p)
    return runs


def _load_metrics(run: Path) -> dict:
    with open(run / "metrics.json") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "mAP" in payload:
        return payload
    # tolerate {"overall": {...}} or {"metrics": {...}} wrappers
    for key in ("overall", "metrics"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return {}


def _aggregate(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    se = std / np.sqrt(n) if n > 1 else 0.0
    return {
        "n_runs": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
    }


def _print_table(rows: list[dict]) -> None:
    header = f"{'metric':<8} {'n':>3} {'mean':>9} {'std':>9} {'ci_low':>9} {'ci_high':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['metric']:<8} {r['n_runs']:>3} {r['mean']:>9.4f} {r['std']:>9.4f} "
            f"{r['ci_low']:>9.4f} {r['ci_high']:>9.4f}"
        )


def _render_degradation_figure(runs: list[Path], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    n_plotted = 0
    for run in runs:
        csv_path = run / "degradation_curves.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        for arm_name, arm_df in df.groupby("arm_name"):
            arm_df = arm_df.sort_values("bin_label")
            ax.plot(
                arm_df["bin_label"].astype(str),
                arm_df["accuracy"],
                marker="o",
                label=f"{arm_name} ({run.name})",
            )
            n_plotted += 1
    if n_plotted == 0:
        print("no degradation_curves.csv found in any run; skipping figure")
        return
    ax.set_xlabel("gap bin")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Degradation curves per arm")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote degradation figure -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="glob pattern(s) matching run directories with metrics.json, e.g. "
        '"outputs/fvessel_*"',
    )
    args = ap.parse_args()

    runs = _collect_runs(args.runs)
    if not runs:
        raise FileNotFoundError(
            f"no run directories with metrics.json matched by {args.runs}"
        )
    print(f"collected {len(runs)} runs:")
    for r in runs:
        print(f"  {r}")

    rows = []
    for key in METRIC_KEYS:
        values = []
        for run in runs:
            metrics = _load_metrics(run)
            v = metrics.get(key, metrics.get(str(key)))  # JSON keys come back as str
            if isinstance(v, (int, float)):
                values.append(v)
        if not values:
            print(f"[warn] metric {key!r} missing from all runs; skipped")
            continue
        rows.append({"metric": str(key), **_aggregate(values)})
    _print_table(rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"aggregate_{ts}.csv"
    df = pd.DataFrame(rows).set_index("metric")
    df.to_csv(csv_path)
    print(f"wrote aggregate table -> {csv_path}")

    png_path = out_dir / f"degradation_curves_{ts}.png"
    _render_degradation_figure(runs, png_path)


if __name__ == "__main__":
    main()
