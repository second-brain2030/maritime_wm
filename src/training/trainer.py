"""Probe trainer (spec section 12).

Trains only the shared Re-ID head on features produced by a frozen
``TrackletEncoder``. Frozen backbone features may be cached to disk via
``FeatureCache`` so repeated epochs skip encoder forward passes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from training.callbacks import MetricLogger, ModelCheckpoint
from utils.reproducibility import seed_everything


class ProbeTrainer:
    """Trains a ``SharedReIDHead`` probe on frozen encoder features.

    Batch contract (dict or tuple, in this order):
      ``frames``      [B, T, C, H, W] tensor
      ``frame_mask``  [B, T] bool tensor (True = valid token; may be None)
      ``labels``      [B] integer tensor of vessel ids
      ``tracklet_ids``[B] (optional, required for feature caching)

    When ``cache`` is provided, each batch must also carry ``tracklet_ids``;
    features are then read from / written to the cache keyed by
    ``(encoder.name, dataset.preprocess_cfg, dataset.sample_cfg,
    dataset.manifest_hash)`` — attributes are read off ``dataloader.dataset``
    when present and default to empty otherwise. Without ``tracklet_ids`` the
    cache is skipped and features are computed on the fly.
    """

    def __init__(
        self,
        head,
        encoder,
        loss_fn,
        optimizer_cfg: dict[str, Any],
        scheduler_cfg: dict[str, Any],
        cache=None,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.head = head.to(device)
        self.encoder = encoder
        self.loss_fn = loss_fn
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.cache = cache
        self.device = device
        self.seed = seed
        seed_everything(seed)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

    # ------------------------------------------------------------------ #
    # Optimizer / scheduler
    # ------------------------------------------------------------------ #
    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = str(self.optimizer_cfg.get("name", "adamw")).lower()
        if name != "adamw":
            raise ValueError(f"unsupported optimizer {name!r}; only 'adamw' supported")
        return torch.optim.AdamW(
            self.head.parameters(),
            lr=float(self.optimizer_cfg["lr"]),
            weight_decay=float(self.optimizer_cfg.get("weight_decay", 1e-4)),
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, num_epochs: int):
        warmup = int(self.scheduler_cfg.get("warmup_epochs", 5))
        if warmup <= 0:
            return CosineAnnealingLR(optimizer, T_max=num_epochs)
        linear = LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
        )
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup))
        return SequentialLR(optimizer, [linear, cosine], milestones=[warmup])

    # ------------------------------------------------------------------ #
    # Data plumbing
    # ------------------------------------------------------------------ #
    def _unpack(self, batch) -> tuple[Tensor, Tensor | None, Tensor | None, list | None]:
        if isinstance(batch, dict):
            frames = batch["frames"]
            frame_mask = batch.get("frame_mask")
            labels = batch.get("labels")
            tracklet_ids = batch.get("tracklet_ids")
        else:
            frames = batch[0]
            frame_mask = batch[1] if len(batch) > 1 else None
            labels = batch[2] if len(batch) > 2 else None
            tracklet_ids = batch[3] if len(batch) > 3 else None
        frames = frames.to(self.device)
        if frame_mask is not None:
            frame_mask = frame_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        return frames, frame_mask, labels, tracklet_ids

    def _feature_key(self, dataloader) -> str:
        dataset = getattr(dataloader, "dataset", None)
        preprocess_cfg = getattr(dataset, "preprocess_cfg", None) or {}
        sample_cfg = getattr(dataset, "sample_cfg", None) or {}
        manifest_hash = getattr(dataset, "manifest_hash", None) or ""
        return self.cache._cache_key(  # noqa: SLF001 - internal helper of FeatureCache
            self.encoder.name, preprocess_cfg, sample_cfg, manifest_hash
        )

    def _get_tokens(
        self,
        frames: Tensor,
        frame_mask: Tensor | None,
        tracklet_ids,
        cache_dict: dict | None,
        new_features: dict,
    ) -> Tensor:
        """Encoder features [B, T, D] for this batch, cache-aware."""
        if cache_dict is not None and tracklet_ids is not None:
            cached = [t for t in tracklet_ids if t in cache_dict]
            if len(cached) == len(tracklet_ids):
                return torch.stack([cache_dict[t] for t in tracklet_ids]).to(
                    self.device
                )
        tokens = self.encoder.encode_observed(frames, frame_mask)
        tokens = tokens.to(self.device)
        if cache_dict is not None and tracklet_ids is not None:
            for tid, tok in zip(tracklet_ids, tokens):
                new_features[tid] = tok.detach().cpu()
        return tokens

    def _init_cache(self, dataloader) -> tuple[str | None, dict | None, dict]:
        if self.cache is None:
            return None, None, {}
        key = self._feature_key(dataloader)
        return key, self.cache.load(key) or {}, {}

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train_epoch(self, dataloader, epoch: int) -> dict[str, float]:
        """One training epoch over the probe head.

        Features come from ``encoder.encode_observed`` unless a complete
        FeatureCache entry exists (then they are loaded instead). Returns
        ``{"loss", "ce", "triplet"}`` (epoch means).
        """
        self.head.train()
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        cache_key, cache_dict, new_features = self._init_cache(dataloader)

        total = ce = triplet = 0.0
        n_batches = 0
        for batch in dataloader:
            frames, frame_mask, labels, tracklet_ids = self._unpack(batch)
            tokens = self._get_tokens(
                frames, frame_mask, tracklet_ids, cache_dict, new_features
            )
            out = self.head(tokens, token_mask=frame_mask, return_logits=True)
            losses = self.loss_fn(out["embedding"], out["logits"], labels)

            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            self.optimizer.step()

            total += losses["total"].item()
            ce += losses["ce"].item()
            triplet += losses["triplet"].item()
            n_batches += 1

        if self.cache is not None and new_features and cache_key is not None:
            cache_dict.update(new_features)
            self.cache.save(cache_key, cache_dict)

        if n_batches == 0:
            return {"loss": 0.0, "ce": 0.0, "triplet": 0.0}
        return {
            "loss": total / n_batches,
            "ce": ce / n_batches,
            "triplet": triplet / n_batches,
        }

    @torch.no_grad()
    def evaluate(self, dataloader, metric_fn) -> dict[str, float]:
        """Collect embeddings + labels, then ``metric_fn(embeddings, labels)``.

        Features are read from / written to the FeatureCache the same way as
        in ``train_epoch``, so validation never re-runs the frozen encoder
        once its features are cached.
        """
        self.head.eval()
        cache_key, cache_dict, new_features = self._init_cache(dataloader)
        embeddings: list[Tensor] = []
        labels: list[Tensor] = []
        for batch in dataloader:
            frames, frame_mask, labels_b, tracklet_ids = self._unpack(batch)
            tokens = self._get_tokens(
                frames, frame_mask, tracklet_ids, cache_dict, new_features
            )
            emb = self.head.get_embedding(tokens, frame_mask)
            embeddings.append(emb.cpu())
            if labels_b is not None:
                labels.append(labels_b.cpu())
        if self.cache is not None and new_features and cache_key is not None:
            cache_dict.update(new_features)
            self.cache.save(cache_key, cache_dict)
        if not embeddings:
            return {}
        emb_all = torch.cat(embeddings)
        lab_all = torch.cat(labels) if labels else None
        return dict(metric_fn(emb_all, lab_all))

    def fit(
        self,
        train_loader,
        val_loader,
        metric_fn,
        num_epochs: int,
        checkpoint_dir: Path,
        early_stop_patience: int = 10,
    ) -> Path:
        """Run the probe training loop.

        Saves the best checkpoint (by validation mAP) to
        ``checkpoint_dir/best.pt``, early-stops when val mAP does not improve
        for ``early_stop_patience`` epochs, logs every epoch to
        ``checkpoint_dir/metrics.csv`` and returns the best checkpoint path.
        """
        seed_everything(self.seed)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(self.optimizer, num_epochs)

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger = MetricLogger(checkpoint_dir / "metrics.csv")
        saver = ModelCheckpoint(checkpoint_dir)

        best_map = -float("inf")
        best_path: Path | None = None
        patience = 0
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            val_metrics = self.evaluate(val_loader, metric_fn)
            val_map = float(val_metrics.get("mAP", 0.0))

            logger.log(epoch, "train", train_metrics)
            logger.log(epoch, "val", val_metrics)

            if val_map > best_map:
                best_map = val_map
                patience = 0
                best_path = saver.save(
                    {
                        "epoch": epoch,
                        "head_state_dict": self.head.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_mAP": val_map,
                        "seed": self.seed,
                    }
                )
            else:
                patience += 1
                if patience >= early_stop_patience:
                    break

        if best_path is None:
            raise RuntimeError("training produced no checkpoint (empty val_loader?)")
        return best_path
