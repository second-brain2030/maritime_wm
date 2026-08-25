"""Training callbacks: checkpointing, early stopping, feature caching, metrics."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


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


class FeatureCache:
    """Cache frozen backbone features to disk keyed by content hash (spec §13).

    The cache key covers the encoder checkpoint, preprocessing config, frame
    sampling config and dataset manifest hash, so features are reused only when
    all of those are unchanged. Stored payloads are ``dict[str, Tensor]`` keyed
    by tracklet id; files live at ``cache_dir/<key>.pt``.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        checkpoint_name: str,
        preprocess_cfg: dict,
        sample_cfg: dict,
        manifest_hash: str,
    ) -> str:
        """sha256 over canonical JSON of all args (sorted); first 16 hex chars."""
        payload = {
            "checkpoint": checkpoint_name,
            "preprocess": preprocess_cfg,
            "sample": sample_cfg,
            "manifest_hash": manifest_hash,
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def save(self, key: str, features: dict[str, Tensor]) -> None:
        torch.save(features, self._path(key))

    def load(self, key: str) -> dict[str, Tensor] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return torch.load(path, weights_only=False)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class MetricLogger:
    """Append per-epoch metrics as CSV rows: epoch, split, metric_name, value, timestamp."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)

    def log(self, epoch: int, split: str, metrics: dict[str, float]) -> None:
        new_file = not self.log_path.exists()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.time()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "split", "metric_name", "value", "timestamp"]
            )
            if new_file:
                writer.writeheader()
            for name, value in metrics.items():
                writer.writerow(
                    {
                        "epoch": epoch,
                        "split": split,
                        "metric_name": name,
                        "value": float(value),
                        "timestamp": timestamp,
                    }
                )

    def read(self) -> Any:
        import pandas as pd

        return pd.read_csv(self.log_path)
