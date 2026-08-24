from datetime import datetime, timezone

import pytest

from data.adapters import adapter_registry, get_adapter
from data.adapters.fvessel import FvesselAdapter, FvesselSequenceMeta

VIDEO_NAME = "2022_05_10_19_21_05_19_31_10_b.mp4"
CAMERA_PARA = "114.32583 30.60139 7 -1 20 55 30.94 2391.26 2446.89 1305.04 855.214"
TRACKING = [
    "0,0,558,720,388,71,1,1,1,1",
    "0,1,297,729,196,46,1,1,1,1",
    "1,0,566,720,386,71,1,1,1,1",
    "1,1,294,729,196,46,1,1,1,1",
    "2,0,575,719,385,71,1,1,1,1",
    "2,1,292,729,197,46,1,1,1,1",
    "3,0,583,719,383,70,1,1,1,1",
    "3,1,289,729,197,46,1,1,1,1",
]
FUSION = [
    "2,250000000,575,719,385,71,1,1,1,1",
    "2,190000000,292,729,197,46,1,1,1,1",
    "3,250000000,583,719,383,70,1,1,1,1",
    "3,190000000,289,729,197,46,1,1,1,1",
]
AIS_CSV = [
    "Number,MMSI,Lon,Lat,Speed,Course,Heading,Type,Timestamp",
    "0,250000000,114.325327,30.60166,0,293.6,511,18,1652181559844",
    "1,190000000,114.302683,30.58059,6.8,33.6,33,18,1652181659157",
]

START_UTC_MS = int(
    datetime(2022, 5, 10, 19, 21, 5, tzinfo=timezone.utc).timestamp() * 1000
)


def _make_layout(tmp_path):
    root = tmp_path / "fvessel"
    seq = root / "01_Video+AIS" / "Video-01"
    (seq / "ais").mkdir(parents=True)
    (seq / "gt").mkdir(parents=True)
    (seq / VIDEO_NAME).touch()
    (seq / "camera_para.txt").write_text(CAMERA_PARA)
    (seq / "gt" / "Video-01_gt_tracking.txt").write_text("\n".join(TRACKING) + "\n")
    (seq / "gt" / "Video-01_gt_fusion.txt").write_text("\n".join(FUSION) + "\n")
    (seq / "ais" / "2022_05_10_19_21_04.csv").write_text("\n".join(AIS_CSV) + "\n")
    return root


def _cfg(root):
    return {"root": str(root)}


def test_adapter_registered():
    assert "fvessel" in adapter_registry


def test_build_manifests(tmp_path):
    root = _make_layout(tmp_path)
    adapter = get_adapter("fvessel", _cfg(root))
    ms = adapter.build_manifests()
    assert len(ms) == 2
    for m in ms:
        m.validate()
    by_vessel = {m.vessel_id: m for m in ms}
    # track id 0 matched to mmsi 250000000 via fusion IoU
    m0 = by_vessel["250000000"]
    assert m0.frame_indices == [0, 25, 50, 75]  # seconds * fps(25)
    assert len(m0.frame_bboxes) == 4
    assert m0.frame_bboxes[0] == [558.0, 720.0, 388.0, 71.0]
    assert m0.frame_timestamps_utc_ms == [START_UTC_MS + i * 1000 for i in range(4)]
    assert m0.fps == 25.0
    assert m0.video_path.endswith(".mp4")
    assert m0.split == "gallery"  # single sequence -> gallery under 0.6/0.2/0.2
    assert m0.source_dataset == "fvessel"
    assert m0.camera_id == "Video-01"
    assert m0.frame_paths[0] == f"{m0.video_path}#0"


def test_aux_manifests(tmp_path):
    root = _make_layout(tmp_path)
    adapter = get_adapter("fvessel", _cfg(root))
    adapter.build_manifests()
    aux = adapter.aux_manifests
    assert set(aux) == {"ais", "meta"}
    assert len(aux["ais"]) == 2
    assert len(aux["meta"]) == 1
    meta: FvesselSequenceMeta = aux["meta"][0]
    assert meta.start_utc_ms == START_UTC_MS
    assert meta.camera_lon == pytest.approx(114.32583)
    assert meta.location_type == "b"


def test_missing_sequences_dir_raises(tmp_path):
    root = tmp_path / "fvessel"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        FvesselAdapter({"root": str(root)}).build_manifests()


def test_bad_video_name_raises(tmp_path):
    root = _make_layout(tmp_path)
    (root / "01_Video+AIS" / "Video-01" / "not_a_video.mp4").touch()
    # two mp4s -> the first in sorted order may be the bad one; force by removing
    (root / "01_Video+AIS" / "Video-01" / VIDEO_NAME).unlink()
    with pytest.raises(ValueError, match="cannot parse"):
        FvesselAdapter(_cfg(root)).build_manifests()
