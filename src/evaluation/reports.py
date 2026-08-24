"""Automated report generation (spec section 15).

report.md must be generated from deterministic rules, not by an LLM, and must
state dataset/split, checkpoints, feature source, metrics with CIs, paired
differences vs Arm A, and limitations.
"""
from __future__ import annotations

from typing import Any, Mapping


def generate_report(results: Mapping[str, Any], out_path: str) -> str:
    """Write report.md from results; NotImplemented until evaluation lands."""
    raise NotImplementedError("report generation lands with the evaluation commit")
