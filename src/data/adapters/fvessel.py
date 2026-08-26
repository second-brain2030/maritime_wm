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

Two adapter classes coexist here:

* ``FvesselAdapter`` — legacy pilot adapter for the raw ``01_Video+AIS``
  layout (``build_manifests()``, per-sequence dirs under root/layout).
* ``FVesselAdapter`` — task-spec adapter (phase 3b) for the public
  ``data/raw/fvessel/{videos,annotations,ais}`` layout: per-sequence
  ``videos/seq_001/`` frame dirs + ``annotations/seq_001.csv|.json`` +
  ``ais/seq_001.csv``; exposes ``build_manifest()``/``run()`` and stores the
  matched AIS trajectory on the manifest as ``ais_trajectory``.
"""
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from ..ais import AisPing, AisTrajectory
from ..manifest import TrackletManifest, save_manifests
from ..splits import validate_identity_disjointness
from .base import DatasetAdapter

# FVessel V1.0 ships two video-naming conventions:
#   A: 2022_05_10_19_21_05_19_31_10_b.mp4   (start = full datetime, end = time)
#   B: 2022-06-04_11.59.22-12.19.12_b.mp4  or  Video_2022-11-10_17.42.23-..._r.mp4
_REV1 = re.compile(
    r"^(?P<start>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})[_-](?P<end>\d{2}_\d{2}_\d{2})_(?P<loc>[a-z]+)\.",
    re.I,
)
_REV2 = re.compile(
    r"^(?:Video_)?(?P<date>\d{4}-\d{2}-\d{2})_(?P<start>\d{2}\.\d{2}\.\d{2})"
    r"-(?P<end>\d{2}\.\d{2}\.\d{2})_(?P<loc>[a-z]+)\.",
    re.I,
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


def _video_bounds_v2(date: str, start: str, end: str) -> tuple[int, int]:
    """date + start/end times in HH.MM.SS form (convention B)."""
    start_dt = datetime.strptime(f"{date} {start}", "%Y-%m-%d %H.%M.%S").replace(
        tzinfo=timezone.utc
    )
    end_dt = datetime.strptime(f"{date} {end}", "%Y-%m-%d %H.%M.%S").replace(
        tzinfo=timezone.utc
    )
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
                    self._build_tracklet(seq_id, meta, track_id, by_track[track_id], id_to_mmsi)
                )

        # identity-disjoint splits: the same vessel (MMSI) may appear in
        # several sequences, so assign splits per VESSEL, not per sequence
        split_map = self._assign_vessel_splits(manifests)
        for m in manifests:
            m.split = split_map[m.vessel_id]

        for m in manifests:
            m.validate()
        validate_identity_disjointness(manifests)  # spec section 12
        self.aux_manifests = {"ais": ais_trajectories, "meta": metas}
        return manifests

    # ------------------------------------------------------------- internal
    def _assign_vessel_splits(self, manifests: list[TrackletManifest]) -> dict[str, str]:
        rng = random.Random(self.split_seed)
        vessels = sorted({m.vessel_id for m in manifests})
        rng.shuffle(vessels)
        n = len(vessels)
        train_n = int(n * self.split_ratios.get("train", 0.6))
        query_n = int(n * self.split_ratios.get("query", 0.2))
        split_map: dict[str, str] = {}
        for i, vid in enumerate(vessels):
            if i < train_n:
                split_map[vid] = "train"
            elif i < train_n + query_n:
                split_map[vid] = "query"
            else:
                split_map[vid] = "gallery"
        return split_map

    def _parse_meta(self, seq_dir: Path, seq_id: str) -> FvesselSequenceMeta:
        video_path = next(
            (p for p in seq_dir.iterdir() if p.suffix.lower() in self.video_extensions),
            None,
        )
        if video_path is None:
            raise FileNotFoundError(f"no video file found in {seq_dir}")
        m = _REV1.match(video_path.name)
        if m is not None:
            start_utc_ms, end_utc_ms = _video_bounds(m.group("start"), m.group("end"))
            loc = m.group("loc")
        else:
            m2 = _REV2.match(video_path.name)
            if m2 is None:
                raise ValueError(
                    f"cannot parse start/end/location from video name {video_path.name!r}"
                )
            start_utc_ms, end_utc_ms = _video_bounds_v2(
                m2.group("date"), m2.group("start"), m2.group("end")
            )
            loc = m2.group("loc")
        return FvesselSequenceMeta(
            sequence_id=seq_id,
            video_path=str(video_path.resolve()),
            start_utc_ms=start_utc_ms,
            end_utc_ms=end_utc_ms,
            location_type=loc,
            **self._parse_camera_para(seq_dir / self.camera_para_file),
            fps=self.fps,
        )

    def _parse_camera_para(self, path: Path) -> dict[str, float]:
        if not path.is_file():
            return {}
        with open(path) as f:
            raw = f.read()
        # the real files are list literals e.g. "[114.32, 30.60, 7, -4, ...]";
        # extract all numbers regardless of separator (space, comma, brackets)
        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
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
            split="train",  # provisional; overwritten by vessel-based assignment
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


# ---------------------------------------------------------------------------
# Task-spec adapter (phase 3b). Public repo layout assumed (documented in the
# module docstring):
#
#   data/raw/fvessel/
#     videos/          seq_001/ ... seq_026/  frame dirs, <frame_id:06d><frame_ext>
#     annotations/     per-sequence bbox-track CSV or JSON
#     ais/             per-sequence AIS CSV: timestamp,mmsi,lat,lon,sog,cog
# ---------------------------------------------------------------------------
_ANNO_COLUMN_ALIASES = {
    "frame_id": ("frame_id", "frame", "Frame"),
    "track_id": ("track_id", "track", "id", "trackid", "Track ID"),
    "x1": ("x1", "xmin", "left", "x_left"),
    "y1": ("y1", "ymin", "top", "y_top"),
    "x2": ("x2", "xmax", "right", "x_right"),
    "y2": ("y2", "ymax", "bottom", "y_bottom"),
    "vessel_class": ("vessel_class", "class", "cls", "type", "vessel_type"),
}


def _normalize_anno_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Map raw annotation columns/keys to the canonical field names."""
    lowered = {str(k).strip().lower(): v for k, v in raw.items()}
    out: dict[str, Any] = {}
    for canon, aliases in _ANNO_COLUMN_ALIASES.items():
        key = next((a for a in aliases if a in raw), None)
        if key is None:
            key = next((a.lower() for a in aliases if a.lower() in lowered), None)
        if key is None:
            continue
        out[canon] = raw[key]
    try:
        frame_id = int(out.get("frame_id"))
        track_id = str(out.get("track_id"))
    except (TypeError, ValueError):
        return None
    if out.get("x1") is None or out.get("y1") is None or out.get("x2") is None or out.get("y2") is None:
        return None
    return {
        "frame_id": frame_id,
        "track_id": track_id,
        "x1": float(out["x1"]),
        "y1": float(out["y1"]),
        "x2": float(out["x2"]),
        "y2": float(out["y2"]),
        "vessel_class": str(out["vessel_class"]) if out.get("vessel_class") not in (None, "") else None,
    }


def _parse_ais_timestamp(value: Any) -> datetime | None:
    """Parse an AIS CSV timestamp cell (ISO-8601 with T or space) to aware UTC."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


class FVesselAdapter:
    """Task-spec FVessel adapter (phase 3b): public ``data/raw/fvessel`` layout.

    Config keys: ``raw_root``, ``manifest_out``, ``fps_fallback`` (default
    25.0), ``ais_dir`` (relative to raw_root, default "ais"), ``anno_dir``
    (default "annotations"), ``frame_ext`` (default ".jpg").
    """

    dataset_name = "fvessel"
    SEQUENCE_IDS = [f"seq_{i:03d}" for i in range(1, 27)]  # seq_001..seq_026

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.raw_root = Path(config.get("raw_root", "data/raw/fvessel"))
        self.manifest_out = Path(config.get("manifest_out", "data/manifests/fvessel.jsonl"))
        self.fps_fallback = float(config.get("fps_fallback", 25.0))
        self.ais_dir = config.get("ais_dir", "ais")
        self.anno_dir = config.get("anno_dir", "annotations")
        self.frame_ext = config.get("frame_ext", ".jpg")
        if not self.frame_ext.startswith("."):
            self.frame_ext = "." + self.frame_ext

    # ------------------------------------------------------------------ API
    def build_manifest(self) -> list[TrackletManifest]:
        """Build normalized tracklets for seq_001..seq_026.

        Splits by sequence index: seq_001-018 -> train, seq_019-022 -> query,
        seq_023-026 -> gallery. Track identity IS vessel identity (track_id).
        ``ais_trajectory`` carries the AIS pings within ±30s of the tracklet
        window; timestamps fall back to frame_id / fps_fallback when no AIS
        anchor exists.
        """
        manifests: list[TrackletManifest] = []
        for seq_id in self.SEQUENCE_IDS:
            annotations = self._parse_annotations(seq_id)
            ais_pings = self._parse_ais(seq_id)
            for track_id in sorted(annotations):
                anns = annotations[track_id]
                frame_ids = [a["frame_id"] for a in anns]
                frame_paths = self._frames_for_track(seq_id, track_id, frame_ids)
                if not frame_paths:
                    continue
                rel_min = frame_ids[0] / self.fps_fallback
                rel_max = frame_ids[-1] / self.fps_fallback
                anchor = self._ais_anchor(ais_pings)
                timestamp_start = self._ts_iso(anchor, rel_min)
                timestamp_end = self._ts_iso(anchor, rel_max)
                traj = self._match_ais_window(ais_pings, anchor, rel_min, rel_max)
                manifests.append(
                    TrackletManifest(
                        tracklet_id=f"fvessel_{seq_id}_{track_id}",
                        vessel_id=track_id,
                        camera_id=seq_id,
                        split=self._split_for_seq(seq_id),
                        frame_paths=frame_paths,
                        timestamp_start=timestamp_start,
                        timestamp_end=timestamp_end,
                        fps=self.fps_fallback,
                        source_dataset="fvessel",
                        vessel_type=(anns[0]["vessel_class"] or None),
                        ais_trajectory=traj,
                    )
                )
        for m in manifests:
            m.validate()
        return manifests

    def run(self) -> Path:
        """build_manifest(), save JSONL, print count, return the output path."""
        manifests = self.build_manifest()
        self.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        save_manifests(str(self.manifest_out), manifests)
        print(f"FVessel: built {len(manifests)} tracklets -> {self.manifest_out}")
        return self.manifest_out

    # ------------------------------------------------------------- internal
    @staticmethod
    def _split_for_seq(seq_id: str) -> str:
        """train for seq_001-018, query for seq_019-022, gallery for seq_023-026."""
        m = re.search(r"(\d+)$", seq_id)
        idx = int(m.group(1)) if m else 0
        if idx <= 18:
            return "train"
        if idx <= 22:
            return "query"
        return "gallery"

    def _parse_ais(self, seq_id: str) -> list[dict[str, Any]]:
        """Read ``ais/<seq_id>.csv`` (columns timestamp,mmsi,lat,lon,sog,cog).

        Returns list of dicts keyed timestamp (datetime), mmsi (str), lat,
        lon, sog, cog (float); ``[]`` when the file is absent.
        """
        path = self.raw_root / self.ais_dir / f"{seq_id}.csv"
        if not path.is_file():
            return []
        pings: list[dict[str, Any]] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_ais_timestamp(row.get("timestamp") or row.get("Timestamp"))
                if ts is None:
                    continue
                try:
                    pings.append(
                        {
                            "timestamp": ts,
                            "mmsi": str(row.get("mmsi") or row.get("MMSI")).strip(),
                            "lat": float(row.get("lat") or row.get("Lat")),
                            "lon": float(row.get("lon") or row.get("Lon")),
                            "sog": float(row.get("sog") or row.get("Speed")),
                            "cog": float(row.get("cog") or row.get("Course")),
                        }
                    )
                except (TypeError, ValueError):
                    continue
        return pings

    def _parse_annotations(self, seq_id: str) -> dict[str, list[dict[str, Any]]]:
        """Read ``annotations/<seq_id>.csv`` or ``.json``.

        Expected columns/keys: frame_id (int), track_id (str), x1,y1,x2,y2
        (bbox), vessel_class (str|None). Returns track_id -> list of frame
        annotation dicts, each sorted by frame_id.
        """
        csv_path = self.raw_root / self.anno_dir / f"{seq_id}.csv"
        json_path = self.raw_root / self.anno_dir / f"{seq_id}.json"
        rows: list[dict[str, Any]] = []
        if csv_path.is_file():
            with open(csv_path, newline="") as f:
                for raw in csv.DictReader(f):
                    ann = _normalize_anno_row(raw)
                    if ann is not None:
                        rows.append(ann)
        elif json_path.is_file():
            with open(json_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                for raw in data.values():
                    if isinstance(raw, list):
                        rows.extend(raw)
            elif isinstance(data, list):
                rows.extend(data)
            rows = [
                ann
                for raw in rows
                if (ann := _normalize_anno_row(raw if isinstance(raw, dict) else {})) is not None
            ]
        by_track: dict[str, list[dict[str, Any]]] = {}
        for ann in rows:
            by_track.setdefault(ann["track_id"], []).append(ann)
        for track_id in by_track:
            by_track[track_id].sort(key=lambda a: a["frame_id"])
        return by_track

    def _frames_for_track(
        self, seq_id: str, track_id: str, frame_ids: list[int]
    ) -> list[str]:
        """Sorted absolute paths to existing frame images for ``frame_ids``.

        Looks in ``raw_root/videos/<seq_id>/`` for files named
        ``<frame_id:06d><frame_ext>``; missing frames are skipped.
        """
        seq_dir = self.raw_root / "videos" / seq_id
        paths: list[str] = []
        for fid in sorted(frame_ids):
            p = seq_dir / f"{fid:06d}{self.frame_ext}"
            if p.is_file():
                paths.append(str(p.resolve()))
        return paths

    @staticmethod
    def _ais_anchor(pings: list[dict[str, Any]]) -> datetime | None:
        """Earliest AIS ping timestamp, assumed aligned to video frame 0."""
        stamps = [p["timestamp"] for p in pings if p.get("timestamp") is not None]
        return min(stamps) if stamps else None

    @staticmethod
    def _ts_iso(anchor: datetime | None, rel_s: float) -> str:
        if anchor is not None:
            return (anchor + timedelta(seconds=rel_s)).isoformat()
        return datetime.fromtimestamp(rel_s, tz=timezone.utc).isoformat()

    def _match_ais_window(
        self,
        pings: list[dict[str, Any]],
        anchor: datetime | None,
        rel_min: float,
        rel_max: float,
    ) -> list[dict[str, Any]] | None:
        """AIS pings within ±30s of the tracklet window (JSON-serializable)."""
        if anchor is None:
            return None
        lo = anchor + timedelta(seconds=rel_min - 30.0)
        hi = anchor + timedelta(seconds=rel_max + 30.0)
        matched = [
            {
                "timestamp": p["timestamp"].isoformat(),
                "mmsi": p["mmsi"],
                "lat": p["lat"],
                "lon": p["lon"],
                "sog": p["sog"],
                "cog": p["cog"],
            }
            for p in pings
            if lo <= p["timestamp"] <= hi
        ]
        return matched or None
