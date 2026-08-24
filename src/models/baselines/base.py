"""Gap baseline protocol (spec section 11 interfaces).

Baseline controls are NOT representation adapters: they consume whatever
inputs their method defines (positions, track association, AIS) and rank
gallery tracklet ids best-to-worst for a trial.
"""
from __future__ import annotations

from typing import Mapping, Protocol

from torch import Tensor

from data.gap_trials import GapTrial


class GapBaseline(Protocol):
    name: str

    def rank(self, trial: GapTrial, gallery_features: Mapping[str, Tensor]) -> list[str]:
        """Return gallery tracklet ids ranked best-to-worst for the trial."""


class BaseGapBaseline:
    """Convenience base; override ``rank``."""

    name: str = "base"

    def rank(self, trial: GapTrial, gallery_features: Mapping[str, Tensor]) -> list[str]:
        raise NotImplementedError(f"{self.__class__.__name__}.rank is not implemented")
