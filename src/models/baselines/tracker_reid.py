"""Arm G: conventional tracker with re-ID head (spec section 6.G).

ByteTrack/BoT-SORT-style association WITH ITS DEFAULT appearance embedding
(never V-JEPA features), so this stays an independent practical standard.
"""
from __future__ import annotations

from typing import Mapping

from torch import Tensor

from data.gap_trials import GapTrial
from .base import BaseGapBaseline


class TrackerReidBaseline(BaseGapBaseline):
    name = "tracker_reid"

    def __init__(
        self,
        tracker: str = "bytetrack",
        appearance_embedding: str = "osnet_x1_0",
    ) -> None:
        self.tracker = tracker
        self.appearance_embedding = appearance_embedding  # pin exact version (spec §6.G)

    def rank(self, trial: GapTrial, gallery_features: Mapping[str, Tensor]) -> list[str]:
        raise NotImplementedError(
            "ByteTrack/BoT-SORT wrapper lands with run_baselines (later commit)"
        )
