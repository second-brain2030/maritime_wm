"""Distractor pools for DGRA trials (spec section 4.4).

Pool rules: 5-20 visually similar vessels co-present in the test window;
chance baseline per trial is exactly 1/pool_size; membership is fixed once
under a single NEUTRAL reference embedding and reused for every arm (never
chosen per-arm, never using V-JEPA or OpenVLA features).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class DistractorPool:
    pool_id: str
    target_vessel_id: str
    candidate_tracklet_ids: list[str]  # includes the target's gallery tracklet; len == size
    reference_embedding_name: str
    seed: int

    def validate(self) -> None:
        errors: list[str] = []
        if not self.pool_id:
            errors.append("pool_id empty")
        if not self.target_vessel_id:
            errors.append("target_vessel_id empty")
        if not self.candidate_tracklet_ids:
            errors.append("candidate_tracklet_ids empty")
        if len(set(self.candidate_tracklet_ids)) != len(self.candidate_tracklet_ids):
            errors.append("candidate_tracklet_ids contains duplicates")
        if not self.reference_embedding_name:
            errors.append("reference_embedding_name empty")
        if errors:
            raise ValueError(f"DistractorPool {self.pool_id!r}: " + "; ".join(errors))

    @property
    def size(self) -> int:
        return len(self.candidate_tracklet_ids)

    @property
    def chance_top1(self) -> float:
        return 1.0 / self.size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DistractorPool":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def chance_baseline(pool_size: int) -> float:
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")
    return 1.0 / pool_size


class DistractorPoolManifest:
    """JSONL store for distractor pools."""

    def __init__(self, pools: Iterable[DistractorPool] = ()) -> None:
        self._pools: list[DistractorPool] = list(pools)

    def __len__(self) -> int:
        return len(self._pools)

    def __iter__(self):
        return iter(self._pools)

    def __getitem__(self, i: int) -> DistractorPool:
        return self._pools[i]

    def get(self, pool_id: str) -> DistractorPool | None:
        for p in self._pools:
            if p.pool_id == pool_id:
                return p
        return None

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for p in self._pools:
                f.write(json.dumps(p.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str) -> "DistractorPoolManifest":
        pools: list[DistractorPool] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pools.append(DistractorPool.from_dict(json.loads(line)))
        return cls(pools)


def build_distractor_pools(
    reference_embedding_name: str,
    gallery_features: Mapping[str, Any],
    target_vessel_id: str,
    candidates: Sequence[str],
    pool_sizes: Sequence[int],
    seed: int,
) -> list[DistractorPool]:
    """Rank candidates by similarity to the target under a neutral reference
    embedding and build one pool per requested size.

    NotImplemented: requires the neutral-reference feature store (fixed frozen
    DINOv2 or VesselReID-pretrained OSNet, spec section 4.4); implement
    alongside feature extraction.
    """
    raise NotImplementedError(
        "distractor-pool construction needs the neutral-reference feature store; "
        "implement alongside extract_features (spec section 4.4)."
    )
