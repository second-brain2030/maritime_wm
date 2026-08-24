"""MVTD adapter (pilot brief P2; spec section 4).

MVTD (arXiv 2506.02866; HF: AhsanBB/Maritime_Visual_Tracking_Dataset_MVTD)
is a single-object visual tracking benchmark with 182 sequences (~150k
frames, Boat/Ship/SailBoat/USV) in GOT-10k layout:

  <root>/<train|test>/<seq>/frame0001.jpg ... + groundtruth.txt
      groundtruth.txt:  <x1>,<y1>,<x2>,<y2> per frame
      absence.label / cut_by_image.label / cover.label: 0/1 per frame

One sequence = one tracked object (one tracklet). Absence/cover frames map to
per-frame occlusion (bbox None), enabling natural occlusion slices.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..manifest import TrackletManifest
from .base import DatasetAdapter

_SEQ_NAME_RE = re.compile(r"^(?P<seq_id>\d+)-(?P<vessel_type>\w+)$")
_LABELS = ("absence", "cut_by_image", "cover")
_DEFAULT_FPS = 30.0


class MvtdAdapter(DatasetAdapter):
    dataset_name = "mvtd"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.root = Path(config.get("root", "data/raw/mvtd"))
        self.fps = float(config.get("layout", {}).get("fps", _DEFAULT_FPS))
        self.split_dirs = tuple(config.get("layout", {}).get("split_dirs", ["train", "test"]))
        self.frame_extensions = tuple(
            config.get("layout", {}).get("frame_extensions", [".jpg", ".jpeg", ".png"])
        )

    # ------------------------------------------------------------------ API
    def build_manifests(self) -> list[TrackletManifest]:
        manifests: list[TrackletManifest] = []
        for split in self.split_dirs:
            split_root = self.root / split
            if not split_root.is_dir():
                raise FileNotFoundError(f"MVTD split dir not found: {split_root}")
            for seq_dir in sorted(split_root.iterdir()):
                if not seq_dir.is_dir():
                    continue
                manifests.append(self._build_tracklet(seq_dir, split))
        if not manifests:
            raise ValueError(f"no MVTD sequences under {self.root}")
        for m in manifests:
            m.validate()
        return manifests

    # ------------------------------------------------------------- internal
    def _build_tracklet(self, seq_dir: Path, split: str) -> TrackletManifest:
        frames = sorted(
            p for p in seq_dir.iterdir() if p.suffix.lower() in self.frame_extensions
        )
        if not frames:
            raise ValueError(f"MVTD sequence {seq_dir} has no frames")
        gt_path = seq_dir / "groundtruth.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"MVTD sequence {seq_dir} missing groundtruth.txt")
        boxes = self._read_boxes(gt_path, len(frames))
        labels = {
            name: self._read_label(seq_dir / f"{name}.label", len(frames))
            for name in _LABELS
        }
        absence = labels["absence"]
        occlusion = "severe" if any(absence) else (
            "partial" if any(labels["cover"]) else "none"
        )
        truncation = "partial" if any(labels["cut_by_image"]) else "none"
        # zero/absence boxes -> None (vessel not visible)
        frame_bboxes = [
            None if bb is None or absence[i] else bb for i, bb in enumerate(boxes)
        ]
        m = _SEQ_NAME_RE.match(seq_dir.name)
        vessel_type = m.group("vessel_type") if m else None
        return TrackletManifest(
            tracklet_id=f"mvtd_{seq_dir.name}",
            vessel_id=seq_dir.name,
            camera_id=seq_dir.name,
            split=split,
            frame_paths=[str(p) for p in frames],
            fps=self.fps,
            frame_indices=list(range(len(frames))),
            frame_bboxes=frame_bboxes,
            vessel_type=vessel_type,
            occlusion_level=occlusion,
            truncation_level=truncation,
            source_dataset="mvtd",
        )

    def _read_boxes(self, path: Path, n_frames: int) -> list[list[float] | None]:
        """Parse x1,y1,x2,y2 lines -> [x, y, w, h]; all-zero line -> None."""
        boxes: list[list[float] | None] = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    boxes.append(None)
                    continue
                vals = [float(p) for p in parts[:4]]
                if all(v == 0 for v in vals):
                    boxes.append(None)
                else:
                    x1, y1, x2, y2 = vals
                    boxes.append([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)])
        if len(boxes) < n_frames:  # pad missing trailing annotations with None
            boxes += [None] * (n_frames - len(boxes))
        return boxes[:n_frames]

    @staticmethod
    def _read_label(path: Path, n_frames: int) -> list[bool]:
        if not path.is_file():
            return [False] * n_frames
        out: list[bool] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                out.append(line not in ("", "0"))
        if len(out) < n_frames:
            out += [False] * (n_frames - len(out))
        return out[:n_frames]
