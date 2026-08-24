"""AIS ping / trajectory manifests (FVessel; spec section 4 + pilot brief P1).

AIS is never model input in the fair vision-only comparison: pings inside a
blackout window are withheld and retained as hidden ground truth for drift
evaluation (brief P1/P4).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class AisPing:
    utc_ms: int
    mmsi: str
    lon: float
    lat: float
    speed_knots: float | None = None
    course_deg: float | None = None
    heading_deg: float | None = None
    vessel_type: str | None = None
    source: str = "ais"

    def validate(self) -> None:
        errors: list[str] = []
        if self.utc_ms < 0:
            errors.append(f"utc_ms < 0 ({self.utc_ms})")
        if not (-180.0 <= self.lon <= 180.0):
            errors.append(f"lon {self.lon} outside [-180, 180]")
        if not (-90.0 <= self.lat <= 90.0):
            errors.append(f"lat {self.lat} outside [-90, 90]")
        if errors:
            raise ValueError(f"AisPing(mmsi={self.mmsi}): " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AisPing":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AisTrajectory:
    trajectory_id: str
    vessel_id: str
    sequence_id: str
    pings: list[AisPing]

    def validate(self) -> None:
        if not self.trajectory_id or not self.vessel_id or not self.sequence_id:
            raise ValueError("AisTrajectory requires trajectory_id, vessel_id, sequence_id")
        for p in self.pings:
            p.validate()
        stamps = [p.utc_ms for p in self.pings]
        if stamps != sorted(stamps):
            raise ValueError(f"AisTrajectory {self.trajectory_id!r}: pings not sorted by utc_ms")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AisTrajectory":
        pings = [AisPing.from_dict(p) for p in d.get("pings", [])]
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "pings"}
        return cls(**kwargs, pings=pings)


class AisTrajectoryManifest:
    """JSONL store for AIS trajectories."""

    def __init__(self, trajectories: Iterable[AisTrajectory] = ()) -> None:
        self._trajectories: list[AisTrajectory] = list(trajectories)

    def __len__(self) -> int:
        return len(self._trajectories)

    def __iter__(self):
        return iter(self._trajectories)

    def get(self, vessel_id: str) -> AisTrajectory | None:
        for t in self._trajectories:
            if t.vessel_id == vessel_id:
                return t
        return None

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for t in self._trajectories:
                f.write(json.dumps(t.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str) -> "AisTrajectoryManifest":
        trajectories: list[AisTrajectory] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(AisTrajectory.from_dict(json.loads(line)))
        return cls(trajectories)


def split_pings_by_window(
    pings: Sequence[AisPing],
    start_utc_ms: int,
    end_utc_ms: int,
    jitter_ms: int = 0,
    dropout_p: float = 0.0,
    seed: int = 42,
) -> tuple[list[AisPing], list[AisPing]]:
    """Split pings into (visible, withheld) around a blackout window.

    Pings with ``start <= utc_ms < end`` are withheld (hidden ground truth).
    Visible pings may receive +/-``jitter_ms`` timestamp noise and a
    ``dropout_p`` fraction removed, simulating asynchronous AIS messaging
    (brief P1). Deterministic given seed.
    """
    import random

    rng = random.Random(seed)
    visible: list[AisPing] = []
    withheld: list[AisPing] = []
    for p in pings:
        if start_utc_ms <= p.utc_ms < end_utc_ms:
            withheld.append(p)
        else:
            if rng.random() < dropout_p:
                continue
            q = AisPing.from_dict(p.to_dict())
            if jitter_ms > 0:
                q.utc_ms += rng.randint(-jitter_ms, jitter_ms)
            visible.append(q)
    visible.sort(key=lambda q: q.utc_ms)
    return visible, withheld
