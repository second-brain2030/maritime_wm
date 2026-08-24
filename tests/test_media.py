from PIL import Image

import pytest

from utils.media import (
    is_video_frame_path,
    load_frame,
    split_video_frame_path,
)


def test_video_frame_path_detection():
    assert is_video_frame_path("v.mp4#123")
    assert not is_video_frame_path("frame0001.jpg")
    assert split_video_frame_path("v.mp4#123") == ("v.mp4", 123)


def test_load_image_frame(tmp_path):
    p = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(p)
    img = load_frame(str(p))
    assert img.mode == "RGB"
    assert img.size == (8, 6)


def test_load_video_frame_requires_fps():
    with pytest.raises(ValueError):
        load_frame("v.mp4#0", fps=None)
