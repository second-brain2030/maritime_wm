"""Data package: normalized manifests, splits, sampling, gap trials."""
from .manifest import TrackletManifest, load_manifests, save_manifests
from .gap_trials import (
    GapProtocolConfig,
    GapTrial,
    GapTrialManifest,
    assign_gap_bin,
    build_gap_trials,
)
from .distractor_pool import DistractorPool, DistractorPoolManifest, chance_baseline

__all__ = [
    "TrackletManifest",
    "load_manifests",
    "save_manifests",
    "GapProtocolConfig",
    "GapTrial",
    "GapTrialManifest",
    "assign_gap_bin",
    "build_gap_trials",
    "DistractorPool",
    "DistractorPoolManifest",
    "chance_baseline",
]
