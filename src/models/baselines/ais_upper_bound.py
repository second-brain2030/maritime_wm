"""Arm H: AIS-fused association upper bound (spec section 6.H).

Runs ONLY on ``ais_available_at_test: true`` trials, reported in a separate
table. Never merged into the fair vision-only comparison.
"""
from __future__ import annotations

from typing import Mapping

from torch import Tensor

from src.data.gap_trials import GapTrial
from .base import BaseGapBaseline


class AisUpperBound(BaseGapBaseline):
    name = "ais_upper_bound"

    def rank(self, trial: GapTrial, gallery_features: Mapping[str, Tensor]) -> list[str]:
        if not trial.ais_available_at_test:
            raise ValueError(
                "AIS upper bound requires ais_available_at_test=True; "
                "it is not part of the fair vision-only comparison (spec §6.H)"
            )
        raise NotImplementedError("AIS-fused association lands with run_baselines (later commit)")
