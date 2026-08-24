"""Data package: normalized manifests, splits, sampling, gaps, AIS, media."""
from .manifest import TrackletManifest, load_manifests, save_manifests
from .ais import AisPing, AisTrajectory, AisTrajectoryManifest, split_pings_by_window
from .gap_trials import (
    GapProtocolConfig,
    GapTrial,
    GapTrialManifest,
    assign_gap_bin,
    build_gap_trials,
)
from .distractor_pool import DistractorPool, DistractorPoolManifest, chance_baseline
from .intermittent import block_patch_mask, find_gaps, intermittent_observation_mask

__all__ = [
    "TrackletManifest",
    "load_manifests",
    "save_manifests",
    "AisPing",
    "AisTrajectory",
    "AisTrajectoryManifest",
    "split_pings_by_window",
    "GapProtocolConfig",
    "GapTrial",
    "GapTrialManifest",
    "assign_gap_bin",
    "build_gap_trials",
    "DistractorPool",
    "DistractorPoolManifest",
    "chance_baseline",
    "block_patch_mask",
    "find_gaps",
    "intermittent_observation_mask",
]
