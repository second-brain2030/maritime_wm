"""Tracklet manifest schema and validation (spec section 4.3).

Rules: never infer protected or missing metadata; store ``unknown`` when the
source annotations do not support a field. ``fingerprint()`` provides the
stable content hash used in feature-cache keys (spec sections 13, 16).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

SPLITS = ("train", "query", "gallery", "test")  # "test" used by MVTD's official split
OCCLUSION_LEVELS = ("none", "partial", "severe", "unknown")
TRUNCATION_LEVELS = ("none", "partial", "severe", "unknown")
KNOWN_SOURCES = ("viv_reid", "vesselreid", "mvtd", "fvessel", "custom")


@dataclass
class TrackletManifest:
    """One normalized tracklet (schema in spec section 4.3)."""

    tracklet_id: str
    vessel_id: str
    camera_id: str
    split: str
    frame_paths: list[str]
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    orientation: str | None = None
    vessel_type: str | None = None
    weather: str | None = None
    occlusion_level: str = "unknown"
    truncation_level: str = "unknown"
    quality_score: float | None = None
    source_dataset: str = "custom"
    # --- optional media / per-frame extensions (FVessel, MVTD; spec §4) ---
    fps: float | None = None
    video_path: str | None = None            # media file when frames come from video
    frame_indices: list[int] | None = None   # indices into the video (or None for image lists)
    frame_timestamps_utc_ms: list[int] | None = None  # aligned to frame_paths
    frame_bboxes: list[list[float] | None] | None = None  # [x, y, w, h] per frame, None = not visible

    def validate(self) -> None:
        errors: list[str] = []
        if not self.tracklet_id:
            errors.append("tracklet_id empty")
        if not self.vessel_id:
            errors.append("vessel_id empty")
        if not self.camera_id:
            errors.append("camera_id empty")
        if self.split not in SPLITS:
            errors.append(f"split {self.split!r} not in {SPLITS}")
        if not self.frame_paths:
            errors.append("frame_paths empty")
        if self.occlusion_level not in OCCLUSION_LEVELS:
            errors.append(
                f"occlusion_level {self.occlusion_level!r} not in {OCCLUSION_LEVELS}"
            )
        if self.truncation_level not in TRUNCATION_LEVELS:
            errors.append(
                f"truncation_level {self.truncation_level!r} not in {TRUNCATION_LEVELS}"
            )
        if self.quality_score is not None and not (0.0 <= self.quality_score <= 1.0):
            errors.append(f"quality_score {self.quality_score!r} outside [0, 1]")
        if self.source_dataset not in KNOWN_SOURCES:
            errors.append(f"source_dataset {self.source_dataset!r} not in {KNOWN_SOURCES}")
        if self.fps is not None and self.fps <= 0:
            errors.append(f"fps must be > 0, got {self.fps}")
        n = len(self.frame_paths)
        if self.frame_indices is not None and len(self.frame_indices) != n:
            errors.append("frame_indices length != frame_paths length")
        if self.frame_timestamps_utc_ms is not None and len(self.frame_timestamps_utc_ms) != n:
            errors.append("frame_timestamps_utc_ms length != frame_paths length")
        if self.frame_bboxes is not None:
            if len(self.frame_bboxes) != n:
                errors.append("frame_bboxes length != frame_paths length")
            else:
                for i, bb in enumerate(self.frame_bboxes):
                    if bb is not None and (len(bb) != 4 or any(v < 0 for v in bb)):
                        errors.append(f"frame_bboxes[{i}] must be [x, y, w, h] >= 0 or null")
        if errors:
            raise ValueError(f"TrackletManifest {self.tracklet_id!r}: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrackletManifest":
        """Build from a dict, ignoring unknown keys (lenient toward extra
        dataset-specific metadata)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def fingerprint(self) -> str:
        """Stable sha256 over canonical JSON (feature-cache keys, spec §13/§16)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_manifests(path: str) -> list[TrackletManifest]:
    with open(path) as f:
        return [
            TrackletManifest.from_dict(json.loads(line))
            for line in f
            if line.strip()
        ]


def save_manifests(path: str, manifests: Sequence[TrackletManifest]) -> None:
    with open(path, "w") as f:
        for m in manifests:
            f.write(json.dumps(m.to_dict(), sort_keys=True) + "\n")
