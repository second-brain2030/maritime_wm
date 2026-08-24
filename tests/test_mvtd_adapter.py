import pytest

from data.adapters import adapter_registry, get_adapter
from data.adapters.mvtd import MvtdAdapter

GROUNDTRUTH = ["10,20,110,120", "0,0,0,0", "30,40,130,140"]
ABSENCE = ["0", "1", "0"]
CUT = ["0", "0", "1"]
COVER = ["0", "1", "0"]


def _make_layout(tmp_path):
    root = tmp_path / "mvtd"
    train = root / "train" / "1-Ship"
    train.mkdir(parents=True)
    for i in range(1, 4):
        (train / f"frame{i:08d}.jpg").touch()
    (train / "groundtruth.txt").write_text("\n".join(GROUNDTRUTH) + "\n")
    (train / "absence.label").write_text("\n".join(ABSENCE) + "\n")
    (train / "cut_by_image.label").write_text("\n".join(CUT) + "\n")
    (train / "cover.label").write_text("\n".join(COVER) + "\n")

    test = root / "test" / "2-Boat"
    test.mkdir(parents=True)
    (test / "frame00000001.jpg").touch()
    (test / "groundtruth.txt").write_text("0,0,0,0\n")
    return root


def test_adapter_registered():
    assert "mvtd" in adapter_registry


def test_build_manifests(tmp_path):
    root = _make_layout(tmp_path)
    ms = get_adapter("mvtd", {"root": str(root)}).build_manifests()
    assert len(ms) == 2
    for m in ms:
        m.validate()
    ship = next(m for m in ms if m.split == "train")
    assert ship.vessel_id == "1-Ship"
    assert ship.vessel_type == "Ship"
    assert ship.camera_id == "1-Ship"
    assert ship.fps == 30.0
    assert ship.frame_indices == [0, 1, 2]
    assert len(ship.frame_paths) == 3
    # frame 1 absent (zero box + absence label) -> None
    assert ship.frame_bboxes == [[10.0, 20.0, 100.0, 100.0], None, [30.0, 40.0, 100.0, 100.0]]
    assert ship.occlusion_level == "severe"  # absence present
    assert ship.truncation_level == "partial"  # cut_by_image present
    boat = next(m for m in ms if m.split == "test")
    assert boat.vessel_type == "Boat"
    assert boat.frame_bboxes == [None]


def test_missing_split_raises(tmp_path):
    root = tmp_path / "mvtd"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        MvtdAdapter({"root": str(root)}).build_manifests()


def test_missing_groundtruth_raises(tmp_path):
    root = tmp_path / "mvtd"
    seq = root / "train" / "3-USV"
    seq.mkdir(parents=True)
    (seq / "frame00000001.jpg").touch()
    with pytest.raises(FileNotFoundError):
        MvtdAdapter({"root": str(root)}).build_manifests()


def test_missing_labels_default_false(tmp_path):
    root = tmp_path / "mvtd"
    seq = root / "train" / "4-SailBoat"
    seq.mkdir(parents=True)
    (seq / "frame00000001.jpg").touch()
    (seq / "groundtruth.txt").write_text("1,2,101,102\n")
    m = MvtdAdapter({"root": str(root), "layout": {"split_dirs": ["train"]}}).build_manifests()[0]
    assert m.occlusion_level == "none"
    assert m.frame_bboxes == [[1.0, 2.0, 100.0, 100.0]]
