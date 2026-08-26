"""Frame loading for image lists and video-with-frame-index paths.

FVessel tracklets reference ``<video.mp4>#<frame_idx>`` (see fvessel adapter);
MVTD / ViV-ReID use plain image paths. This module resolves both.

Video frames are decoded with OpenCV ``VideoCapture`` + ``CAP_PROP_POS_FRAMES``
seeking: frame-accurate and memory-safe (decodes ONE frame per call, unlike
``torchvision.io.read_video`` which materializes the whole video — a 10-minute
1080p clip would be ~90 GB in RAM).
"""
from __future__ import annotations

VIDEO_FRAME_SEP = "#"


def is_video_frame_path(path: str) -> bool:
    return VIDEO_FRAME_SEP in path


def split_video_frame_path(path: str) -> tuple[str, int]:
    video, idx = path.rsplit(VIDEO_FRAME_SEP, 1)
    return video, int(idx)


def load_frame(path: str, fps: float | None = None) -> Image.Image:
    """Load a single RGB frame as PIL Image.

    For ``video#idx`` paths the frame is decoded by seeking OpenCV to the
    exact frame index (``CAP_PROP_POS_FRAMES``), decoding only that frame.
    ``fps`` is accepted for interface compatibility (image paths).
    """
    if is_video_frame_path(path):
        video, frame_idx = split_video_frame_path(path)
        return _load_video_frame(video, frame_idx)
    return _load_image(path)


def _load_image(path: str) -> Image.Image:
    from PIL import Image

    return Image.open(path).convert("RGB")


def _load_video_frame(video: str, frame_idx: int) -> Image.Image:
    from PIL import Image

    import cv2

    cap = cv2.VideoCapture(video)
    try:
        if not cap.isOpened():
            raise OSError(f"cv2 cannot open video: {video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode frame {frame_idx} from {video}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()