"""Gap baselines (external controls, spec section 6.F/G/H).

``KalmanDeadReckon`` and the new position-based ``GapBaseline`` protocol are
imported eagerly. The legacy controls (``BaseGapBaseline``,
``TrackerReidBaseline``, ``AisUpperBound``) are lazy-loaded on attribute
access so that importing ``models.baselines`` (or ``src.models.baselines``
from the repo root) stays dependency-light: the legacy modules import
``data.gap_trials``, which requires ``src`` on ``sys.path``.
"""
from __future__ import annotations

from .kalman_deadreckon import GapBaseline, KalmanDeadReckon

__all__ = [
    "GapBaseline",
    "KalmanDeadReckon",
    "BaseGapBaseline",
    "TrackerReidBaseline",
    "AisUpperBound",
]


def __getattr__(name: str):
    if name == "BaseGapBaseline":
        from .base import BaseGapBaseline

        return BaseGapBaseline
    if name == "TrackerReidBaseline":
        from .tracker_reid import TrackerReidBaseline

        return TrackerReidBaseline
    if name == "AisUpperBound":
        from .ais_upper_bound import AisUpperBound

        return AisUpperBound
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
