"""On-demand frame embedding for evaluation (brief P3).

Extracts per-frame embeddings for specific frame indices (query-before-blackout
frames, candidate reappearance frames) with an arm encoder and optionally the
trained probe head. This avoids the leakage that a whole-tracklet feature
cache would introduce into re-acquisition evaluation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torchvision.transforms import ToTensor

from data.manifest import TrackletManifest
from utils.media import load_frame
from utils.tokens import pool_tokens_to


def embed_frames(
    encoder,
    head,
    paths: Sequence[str],
    fps: float | None,
    max_frames: int = 16,
    device: str = "cpu",
    load_resolution: int = 256,
    pool_tokens: int | None = None,
) -> np.ndarray:
    """Mean embedding of up to ``max_frames`` frames as a [D] numpy vector.

    With ``head``: pooled probe embedding; without: mean-pooled raw encoder
    tokens (used by the appearance baseline, Arm G). Frames are resized to
    ``load_resolution`` at load time to bound memory on large maritime
    imagery; ``pool_tokens`` chunk-mean-pools long token sequences
    (V-JEPA) before the head, keeping transformer pooling RAM-bounded.
    """
    paths = list(paths)[:max_frames]
    if not paths:
        raise ValueError("embed_frames requires at least one frame path")
    tensors = []
    for p in paths:
        try:
            tensors.append(ToTensor()(load_frame(p, fps=fps).resize((load_resolution, load_resolution))))
        except (OSError, ValueError):
            # frame beyond the recorded video / missing media: skip it
            continue
    if not tensors:
        raise ValueError(f"no decodable frames among {len(paths)} paths (first: {paths[0]})")
    frames = torch.stack(tensors)
    frames = frames.unsqueeze(0).to(device)  # [1, T, 3, H, W]
    with torch.no_grad():
        tokens = encoder.encode_observed(frames, None)  # [1, T, D]
        if head is not None:
            if pool_tokens is not None:
                tokens = pool_tokens_to(tokens, pool_tokens)
            emb = head(tokens.to(device), None)["embedding"][0]
        else:
            emb = tokens[0].mean(dim=0)
    return emb.detach().cpu().numpy()


def tracklet_frame_paths(
    tracklet: TrackletManifest, frame_indices: Sequence[int]
) -> list[str]:
    """Resolve frame indices to frame paths for one tracklet."""
    path_map = dict(zip(tracklet.frame_indices or [], tracklet.frame_paths))
    return [path_map[fi] for fi in frame_indices if fi in path_map]


def tracklet_visible_bboxes(
    tracklet: TrackletManifest,
) -> list[tuple[int, list[float]]]:
    """[(frame_index, [x, y, w, h]), ...] for frames with a visible bbox."""
    out: list[tuple[int, list[float]]] = []
    if tracklet.frame_indices is None or tracklet.frame_bboxes is None:
        return out
    for fi, bb in zip(tracklet.frame_indices, tracklet.frame_bboxes):
        if bb is not None:
            out.append((fi, bb))
    return out
