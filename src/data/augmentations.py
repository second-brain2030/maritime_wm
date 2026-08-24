"""Evaluation-only deterministic stress transforms (spec section 8.4).

Transforms must be deterministic by seed, applied only at evaluation time,
and must preserve labels. Same-frame occlusion and scattered frame dropout
are DIAGNOSTICS, not the primary hard test (spec section 8.1).
"""
from __future__ import annotations

from utils.registry import Registry

# key: stressor name -> factory(severity: str, seed: int) -> transform
stress_transform_registry = Registry("stress_transforms")


def register_stressor(name: str):
    def deco(fn):
        stress_transform_registry.register(name, fn)
        return fn

    return deco


@register_stressor("block_occlusion")
def block_occlusion(severity: str, seed: int):
    """Deterministic block occlusion (0/15/30/50% masked area)."""
    raise NotImplementedError(
        "block_occlusion transform lands with the stress-suite run (later commit)"
    )


@register_stressor("frame_dropout")
def frame_dropout(severity: str, seed: int):
    """Scattered frame dropout (0/25/50/75%). Diagnostic only."""
    raise NotImplementedError(
        "frame_dropout transform lands with the stress-suite run (later commit)"
    )


@register_stressor("haze")
def haze(severity: str, seed: int):
    """Haze/contrast attenuation. Tests degraded input, NOT fog penetration."""
    raise NotImplementedError("haze transform lands with the stress-suite run (later commit)")


@register_stressor("time_gap")
def time_gap(severity: str, seed: int):
    """Time-gap retrieval; feeds DGRA gap bins."""
    raise NotImplementedError("time_gap transform lands with the stress-suite run (later commit)")
