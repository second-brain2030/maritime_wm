"""FVessel adapter (pilot brief P1; spec section 4).

FVessel (TITS 2023, github.com/gy65896/FVessel, HF: gy65896/FVessel) contains
26 synchronized video sequences + asynchronous AIS messages captured on the
Yangtze River. Per sequence:

  ais/*.csv                 AIS pings [Number, MMSI, Lon, Lat, Speed, Course,
                            Heading, Type, Timestamp(ms)]
  <start>_<end>_<type>.mp4  video (start/end UTC in the filename)
  camera_para.txt           [Lon, Lat, H-Ori, V-Ori, Height, H-FoV, V-FoV,
                            fx, fy, u0, v0]
  gt/Video-XX_gt_tracking.txt  <second>,<track id>,<bbox xywh>,<conf>,<x,y,z>
  gt/Video-XX_gt_fusion.txt    <second>,<mmsi>,<bbox xywh>,<conf>,<x,y,z>

Outputs: one TrackletManifest per (sequence, tracked vessel) with per-frame
bboxes/timestamps; AIS trajectories; per-sequence camera meta. AIS pings are
kept separate so the blackout harness can withhold them as hidden ground
truth (brief P1).
"""
from __future__ import annotations

import csv
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..ais import AisPing, AisTrajectory
from ..manifest import TrackletManifest
from ..splits import validate_identity_disjointness
from .base import DatasetAdapter

_VIDEO_NAME_RE = re.compile(
    r"^(?P<start>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})_"
    r"(?P<end>\d{2}_\d{2}_\d{2})_(?P<loc>[a-z]+)\."
)


def _parse_utc_ms(stamp: str) -> int:
    dt = datetime.strptime(stamp, "%Y_%m_%d_%H_%M_%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _video_bounds(start: str, end: str) -> tuple[int, int]:
    """start = full UTC datetime; end = time-of-day only (FVessel naming)."""
    start_dt = datetime.strptime(start, "%Y_%m_%d_%H_%M_%S").replace(tzinfo=timezone.utc)
    end_t = datetime.strptime(end, "%H_%M_%S").time()
    end_dt = datetime.combine(start_dt.date(), end_t, tzinfo=timezone.utc)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


@dataclass
class FvesselSequenceMeta:
    sequence_id: str
    video_path: str
    start_utc_ms: int
    end_utc_ms: int
    location_type: str
    camera_lon: float | None = None
    camera_lat: float | None = None
    camera_height_m: float | None = None
    h_fov_deg: float | None = None
    v_fov_deg: float | None = None
    fx: float | None = None
    fy: float | None = None
    u0: float | None = None
    v0: float | None = None
    fps: float = 25.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FvesselSequenceMeta":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class FvesselAdapter(DatasetAdapter):
    dataset_name = "fvessel"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.root = Path(config.get("root", "data/raw/fvessel"))
        layout = config.get("layout", {})
        self.sequences_dir = layout.get("sequences_dir", "01_Video+AIS")
        self.ais_dir = layout.get("ais_dir", "ais")
        self.gt_dir = layout.get("gt_dir", "gt")
        self.camera_para_file = layout.get("camera_para_file", "camera_para.txt")
        self.tracking_suffix = layout.get("tracking_gt_suffix", "_gt_tracking.txt")
        self.fusion_suffix = layout.get("fusion_gt_suffix", "_gt_fusion.txt")
        self.video_extensions = tuple(layout.get("video_extensions", [".mp4"]))
        self.fps = float(layout.get("fps", 25.0))
        self.split_seed = int(layout.get("split_seed", 42))
        self.split_ratios = dict(
            layout.get("split_ratios", {"train": 0.6, "query": 0.2, "gallery": 0.2})
        )
        self.aux_manifests: dict[str, list[Any]] = {}

    # ------------------------------------------------------------------ API
    def build_manifests(self) -> list[TrackletManifest]:
        seq_root = self.root / self.sequences_dir
        if not seq_root.is_dir():
            raise FileNotFoundError(
                f"FVessel sequences dir not found: {seq_root} "
                "(config root + layout.sequences_dir)"
            )
        sequence_ids = sorted(p.name for p in seq_root.iterdir() if p.is_dir())
        if not sequence_ids:
            raise ValueError(f"no sequence directories under {seq_root}")
        split_map = self._assign_splits(sequence_ids)

        manifests: list[TrackletManifest] = []
        ais_trajectories: list[AisTrajectory] = []
        metas: list[FvesselSequenceMeta] = []
        for seq_id in sequence_ids:
            seq_dir = seq_root / seq_id
            meta = self._parse_meta(seq_dir, seq_id)
            metas.append(meta)
            track_rows, fusion_rows = self._parse_gt(seq_dir)
            id_to_mmsi = self._match_tracking_to_mmsi(track_rows, fusion_rows)
            ais_trajectories.extend(self._group_ais(seq_id, self._parse_ais(seq_dir)))
            by_track = self._regroup_by_track(track_rows)
            for track_id in sorted(by_track):
                manifests.append(
                    self._build_tracklet(seq_id, meta, track_id, by_track[track_id], id_to_mmsi, split_map)
                )

        for m in manifests:
            m.validate()
        validate_identity_disjointness(manifests)  # spec section 12
        self.aux_manifests = {"ais": ais_trajectories, "meta": metas}
        return manifests

    # ------------------------------------------------------------- internal
    def _assign_splits(self, sequence_ids: list[str]) -> dict[str, str]:
        rng = random.Random(self.split_seed)
        ordered = sorted(sequence_ids)
        rng.shuffle(ordered)
        n = len(ordered)
        train_n = int(n * self.split_ratios.get("train", 0.6))
        query_n = int(n * self.split_ratios.get("query", 0.2))
        split_map: dict[str, str] = {}
        for i, sid in enumerate(ordered):
            if i < train_n:
                split_map[sid] = "train"
            elif i < train_n + query_n:
                split_map[sid] = "query"
            else:
                split_map[sid] = "gallery"
        return split_map

    def _parse_meta(self, seq_dir: Path, seq_id: str) -> FvesselSequenceMeta:
        video_path = next(
            (p for p in seq_dir.iterdir() if p.suffix.lower() in self.video_extensions),
            None,
        )
        if video_path is None:
            raise FileNotFoundError(f"no video file found in {seq_dir}")
        m = _VIDEO_NAME_RE.match(video_path.name)
        if m is None:
            raise ValueError(
                f"cannot parse start/end/location from video name {video_path.name!r}"
            )
        start_utc_ms, end_utc_ms = _video_bounds(m.group("start"), m.group("end"))
        return FvesselSequenceMeta(
            sequence_id=seq_id,
            video_path=str(video_path.resolve()),
            start_utc_ms=start_utc_ms,
            end_utc_ms=end_utc_ms,
            location_type=m.group("loc"),
            **self._parse_camera_para(seq_dir / self.camera_para_file),
            fps=self.fps,
        )

    def _parse_camera_para(self, path: Path) -> dict[str, float]:
        if not path.is_file():
            return {}
        with open(path) as f:
            tokens = f.read().split()
        if len(tokens) < 11:
            raise ValueError(f"{path}: expected 11 camera parameters, got {len(tokens)}")
        return {
            "camera_lon": float(tokens[0]),
            "camera_lat": float(tokens[1]),
            "camera_height_m": float(tokens[4]),
            "h_fov_deg": float(tokens[5]),
            "v_fov_deg": float(tokens[6]),
            "fx": float(tokens[7]),
            "fy": float(tokens[8]),
            "u0": float(tokens[9]),
            "v0": float(tokens[10]),
        }

    def _parse_gt(self, seq_dir: Path) -> tuple[dict[int, list[list[float]]], dict[int, list[list[float]]]]:
        gt_dir = seq_dir / self.gt_dir
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"FVessel gt dir not found: {gt_dir}")
        tracking = self._read_gt_rows(self._find_gt_file(gt_dir, seq_dir.name, self.tracking_suffix))
        fusion = self._read_gt_rows(self._find_gt_file(gt_dir, seq_dir.name, self.fusion_suffix))
        return tracking, fusion

    @staticmethod
    def _find_gt_file(gt_dir: Path, seq_name: str, suffix: str) -> Path:
        candidates = [
            gt_dir / f"{seq_name}{suffix}",
            gt_dir / f"Video-{seq_name}{suffix}",
        ]
        return next((p for p in candidates if p.is_file()), candidates[0])

    def _read_gt_rows(self, path: Path) -> dict[int, list[list[float]]]:
        """{second: [[ident, bb_left, bb_top, bb_width, bb_height], ...]}."""
        if not path.is_file():
            return {}
        out: dict[int, list[list[float]]] = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                second = int(float(parts[0]))
                ident = float(parts[1])
                bb = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                out.setdefault(second, []).append([ident, *bb])
        return out

    @staticmethod
    def _regroup_by_track(
        track_rows: dict[int, list[list[float]]],
    ) -> dict[int, list[tuple[int, list[float]]]]:
        """track_id -> sorted [(second, bbox), ...]."""
        by_track: dict[int, list[tuple[int, list[float]]]] = {}
        for second, entries in track_rows.items():
            for e in entries:
                by_track.setdefault(int(e[0]), []).append((second, e[1:]))
        for track_id in by_track:
            by_track[track_id].sort(key=lambda o: o[0])
        return by_track

    def _parse_ais(self, seq_dir: Path) -> list[AisPing]:
        ais_root = seq_dir / self.ais_dir
        if not ais_root.is_dir():
            return []
        pings: list[AisPing] = []
        for csv_path in sorted(ais_root.glob("*.csv")):
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            if not rows:
                continue
            start = 1 if not rows[0][0].strip().replace(".", "", 1).isdigit() else 0
            for row in rows[start:]:
                if len(row) < 9:
                    continue
                try:
                    pings.append(
                        AisPing(
                            utc_ms=int(float(row[8])),
                            mmsi=str(int(float(row[1]))),
                            lon=float(row[2]),
                            lat=float(row[3]),
                            speed_knots=_opt_float(row[4]),
                            course_deg=_opt_float(row[5]),
                            heading_deg=_opt_float(row[6]),
                            vessel_type=_opt_str(row[7]),
                        )
                    )
                except (ValueError, IndexError):
                    continue
        return pings

    def _group_ais(self, seq_id: str, pings: list[AisPing]) -> list[AisTrajectory]:
        by_mmsi: dict[str, list[AisPing]] = {}
        for p in pings:
            by_mmsi.setdefault(p.mmsi, []).append(p)
        trajectories: list[AisTrajectory] = []
        for mmsi, ps in sorted(by_mmsi.items()):
            ps.sort(key=lambda q: q.utc_ms)
            trajectories.append(
                AisTrajectory(
                    trajectory_id=f"fv_{seq_id}_{mmsi}",
                    vessel_id=mmsi,
                    sequence_id=seq_id,
                    pings=ps,
                )
            )
        return trajectories

    def _match_tracking_to_mmsi(
        self,
        track_rows: dict[int, list[list[float]]],
        fusion_rows: dict[int, list[list[float]]],
    ) -> dict[int, str | None]:
        """Map per-second track id -> MMSI by best bbox IoU with fusion rows."""
        id_to_mmsi: dict[int, str | None] = {}
        for second, track_entries in track_rows.items():
            fusion_entries = fusion_rows.get(second, [])
            if not fusion_entries:
                continue
            for entry in track_entries:
                track_id, bb = int(entry[0]), entry[1:]
                best, best_iou = None, 0.0
                for fe in fusion_entries:
                    v = _iou_xywh(bb, fe[1:])
                    if v > best_iou:
                        best, best_iou = fe[0], v
                id_to_mmsi[track_id] = str(int(best)) if (best is not None and best_iou >= 0.5) else None
        return id_to_mmsi

    def _build_tracklet(
        self,
        seq_id: str,
        meta: FvesselSequenceMeta,
        track_id: int,
        observations: list[tuple[int, list[float]]],
        id_to_mmsi: dict[int, str | None],
        split_map: dict[str, str],
    ) -> TrackletManifest:
        frame_indices = [round(second * self.fps) for second, _ in observations]
        bboxes = [bb for _, bb in observations]
        timestamps = [int(meta.start_utc_ms + fi / self.fps * 1000) for fi in frame_indices]
        frame_paths = [f"{meta.video_path}#{fi}" for fi in frame_indices]
        mmsi = id_to_mmsi.get(track_id)
        vessel_id = mmsi if mmsi else f"{seq_id}__t{track_id}"
        return TrackletManifest(
            tracklet_id=f"fv_{seq_id}_t{track_id}",
            vessel_id=vessel_id,
            camera_id=seq_id,
            split=split_map[seq_id],
            frame_paths=frame_paths,
            fps=self.fps,
            video_path=meta.video_path,
            frame_indices=frame_indices,
            frame_timestamps_utc_ms=timestamps,
            frame_bboxes=bboxes,
            source_dataset="fvessel",
        )


def _opt_float(v: str) -> float | None:
    return float(v) if v not in ("", "None") else None


def _opt_str(v: str) -> str | None:
    return v if v not in ("", "None") else None
