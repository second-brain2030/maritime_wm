"""Split hygiene (spec section 12: identity split hygiene).

Query and gallery identities may be shared with each other per the official
Re-ID protocol, but neither may leak into training identities unless the
official benchmark explicitly defines a closed-set protocol.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from .manifest import TrackletManifest

SPLIT_NAMES = ("train", "query", "gallery")


def identity_sets(manifests: Iterable[TrackletManifest]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for m in manifests:
        if m.split in out:
            out[m.split].add(m.vessel_id)
    return out


def validate_identity_disjointness(manifests: Sequence[TrackletManifest]) -> dict:
    """Raise if train identities overlap query/gallery; return a report dict."""
    sets = identity_sets(manifests)
    report = {
        "train_identity_count": len(sets["train"]),
        "query_identity_count": len(sets["query"]),
        "gallery_identity_count": len(sets["gallery"]),
        "train_query_overlap": sorted(sets["train"] & sets["query"]),
        "train_gallery_overlap": sorted(sets["train"] & sets["gallery"]),
    }
    overlaps = report["train_query_overlap"] + report["train_gallery_overlap"]
    if overlaps:
        raise ValueError(
            "train identities must not appear in query/gallery; overlapping "
            "identities: " + ", ".join(overlaps)
        )
    return report
