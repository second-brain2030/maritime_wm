"""Token-sequence utilities for the probe pipeline."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def pool_tokens_to(tokens: Tensor, max_tokens: int = 64) -> Tensor:
    """Mean-pool ``[..., T, D]`` tokens into <= ``max_tokens`` representatives.

    Long token sequences (e.g. V-JEPA video tokens: 1152-4608 per clip) make
    transformer pooling on CPU RAM-heavy; this deterministic chunk-mean
    aggregation bounds the head's attention cost while retaining the temporal
    sequence structure. T <= max_tokens passes through unchanged.
    """
    t = tokens.shape[-2]
    if t <= max_tokens:
        return tokens
    stride = math.ceil(t / max_tokens)
    pad = (-t) % stride
    if pad:
        tokens = nn.functional.pad(tokens, (0, 0, 0, pad))
    b, tt, d = tokens.shape
    pooled = tokens.view(b, tt // stride, stride, d).mean(dim=2)
    return pooled