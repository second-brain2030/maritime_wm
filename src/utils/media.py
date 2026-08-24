"""Frame loading for image lists and video-with-frame-index paths.

FVessel tracklets reference ``<video.mp4>#<frame_idx>`` (see fvessel adapter);
MVTD / ViV-ReID use plain image paths. This module resolves both.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

VIDEO_FRAME_SEP = "#"


def is_video_frame_path(path: str) -> bool:
    return VIDEO_FRAME_SEP in path


def split_video_frame_path(path: str) -> tuple[str, int]:
    video, idx = path.rsplit(VIDEO_FRAME_SEP, 1)
    return video, int(idx)


def load_frame(path: str, fps: float | None = None) -> Image.Image:
    """Load a single RGB frame as PIL Image.

    For ``video#idx`` paths the frame is decoded from the video via
    torchvision.io (whole-video read per call; fine for pilot scale).
    """
    if is_video_frame_path(path):
        video, frame_idx = split_video_frame_path(path)
        if fps is None or fps <= 0:
            raise ValueError(f"fps required to decode video frames, got {fps!r}")
        import torchvision.io

        video_tensor, _, _ = torchvision.io.read_video(
            video, pts_unit="sec", output_format="TCHW"
        )
        frame_tensor = video_tensor[frame_idx]
        return Image.fromarray(frame_tensor.permute(1, 2, 0).numpy())
    return Image.open(path).convert("RGB")
