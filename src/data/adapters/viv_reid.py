"""ViV-ReID dataset adapter (spec section 4.1).

ViV-ReID (480 identities, 20 cameras, 7,165 tracklets, ~1.14M frames) is
distributed under a signed data-usage agreement and its raw folder layout is
not publicly documented. The adapter is therefore CONFIG-DRIVEN: upstream
folder names, the vessel-identity parse pattern, and the camera parse pattern
are mapped through ``configs/data/viv_reid.yaml`` rather than hard-coded
(spec section 4.1). It discovers tracklet directories of frames under
train/query/gallery, validates split hygiene, and fails loudly with a precise
message when the layout does not match.

Convention assumed (overridable): one directory per tracklet, named so the
vessel identity (and optionally camera) can be parsed, e.g.
``v0123_cam5_track2/``. If the distributed layout differs, adjust the config
patterns — no code change required.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..manifest import TrackletManifest
from ..splits import validate_identity_disjointness
from .base import DatasetAdapter

_DEFAULT_IDENTITY_PATTERN = r"^(?P<vessel_id>.*)$"
_DEFAULT_FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class ViVReidAdapter(DatasetAdapter):
    dataset_name = "viv_reid"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.root = Path(config.get("root", "data/raw/viv-reid"))
        layout = config.get("layout", {})
        self.train_dir = layout.get("train_dir", "train")
        self.query_dir = layout.get("query_dir", "query")
        self.gallery_dir = layout.get("gallery_dir", "gallery")
        self.identity_pattern = re.compile(
            layout.get("tracklet_identity_pattern", _DEFAULT_IDENTITY_PATTERN)
        )
        camera_pattern = layout.get("camera_pattern")
        self.camera_pattern = re.compile(camera_pattern) if camera_pattern else None
        self.frame_extensions = tuple(
            layout.get("frame_extensions", list(_DEFAULT_FRAME_EXTENSIONS))
        )

    # ------------------------------------------------------------------ API
    def build_manifests(self) -> list[TrackletManifest]:
        manifests: list[TrackletManifest] = []
        skipped_empty: list[tuple[str, str]] = []
        for split, dirname in (
            ("train", self.train_dir),
            ("query", self.query_dir),
            ("gallery", self.gallery_dir),
        ):
            split_root = self.root / dirname
            if not split_root.is_dir():
                raise FileNotFoundError(
                    f"ViV-ReID split dir not found: {split_root} "
                    f"(config root={self.root}, map upstream names in "
                    "configs/data/viv_reid.yaml -> layout.*)"
                )
            for tracklet_dir in sorted(split_root.iterdir()):
                if not tracklet_dir.is_dir():
                    continue
                frames = self._discover_frames(tracklet_dir)
                if not frames:
                    skipped_empty.append((split, tracklet_dir.name))
                    continue
                vessel_id, camera_id = self._parse_tracklet_name(tracklet_dir.name)
                manifests.append(
                    TrackletManifest(
                        tracklet_id=f"viv_{split}_{tracklet_dir.name}",
                        vessel_id=vessel_id,
                        camera_id=camera_id,
                        split=split,
                        frame_paths=[str(p) for p in frames],
                        source_dataset="viv_reid",
                    )
                )

        if skipped_empty:
            print(
                f"[viv_reid] skipped {len(skipped_empty)} empty tracklet dirs: "
                + ", ".join(f"{s}/{name}" for s, name in skipped_empty[:5])
            )
        if not manifests:
            raise ValueError(
                f"no tracklets discovered under {self.root} "
                f"(train={self.train_dir}, query={self.query_dir}, "
                f"gallery={self.gallery_dir}); check the layout config"
            )

        for m in manifests:
            m.validate()
        validate_identity_disjointness(manifests)  # spec section 12
        return manifests

    # ------------------------------------------------------------- internal
    def _discover_frames(self, tracklet_dir: Path) -> list[Path]:
        return sorted(
            p
            for p in tracklet_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.frame_extensions
        )

    def _parse_tracklet_name(self, name: str) -> tuple[str, str]:
        m = self.identity_pattern.match(name)
        if m is None or not (m.groupdict().get("vessel_id") or "").strip():
            raise ValueError(
                f"cannot parse vessel identity from tracklet folder {name!r} "
                f"with pattern {self.identity_pattern.pattern!r}; adjust "
                "layout.tracklet_identity_pattern in the data config"
            )
        vessel_id = m.group("vessel_id")
        camera_id = "unknown"
        if self.camera_pattern is not None:
            cm = self.camera_pattern.search(name)
            if cm is not None and (cm.groupdict().get("camera_id") or "").strip():
                camera_id = cm.group("camera_id")
        return vessel_id, camera_id
