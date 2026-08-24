"""Arm F: motion-only dead-reckoning control (spec section 6.F).

Kalman (or particle) filter over the query tracklet's last observed
positions/velocities; rank gallery candidates by nearest-position matching
with the filter's uncertainty radius. No appearance input. Sanity gate: must
collapse to <= 2x chance at the long-gap bin (spec section 8.2).
"""
from __future__ import annotations

from typing import Mapping

from torch import Tensor

from data.gap_trials import GapTrial
from .base import BaseGapBaseline


class KalmanDeadReckoning(BaseGapBaseline):
    name = "kalman_deadreckon"

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def rank(self, trial: GapTrial, gallery_features: Mapping[str, Tensor]) -> list[str]:
        raise NotImplementedError(
            "Kalman dead-reckoning lands with run_baselines (later commit)"
        )
