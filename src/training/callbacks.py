"""Training callbacks."""
from __future__ import annotations

from pathlib import Path

import torch


class EarlyStopping:
    """Stop training when a monitored metric stops improving (mode min/max)."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode {mode!r} not in (min, max)")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.best_epoch: int | None = None

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        delta = value - self.best
        return delta < -self.min_delta if self.mode == "min" else delta > self.min_delta

    def __call__(self, metric: float, epoch: int | None = None) -> bool:
        """Register a metric; return True when training should stop."""
        if self._is_better(metric):
            self.best = metric
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


class ModelCheckpoint:
    def __init__(self, dirpath: str | Path, filename: str = "best.pt") -> None:
        self.dirpath = Path(dirpath)
        self.filename = filename

    def save(self, state: dict) -> Path:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / self.filename
        torch.save(state, path)
        return path
