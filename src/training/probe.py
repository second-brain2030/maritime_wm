"""Probe training on cached frozen features (brief P3; spec section 12).

Trains only the shared Re-ID head (LayerNorm -> temporal attention pool ->
projection -> L2 norm -> classifier) on content-cached backbone features.
Loss: ID cross-entropy + batch-hard triplet. Deterministic per seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from data.manifest import TrackletManifest
from models.common_head import SharedReIDHead
from training.callbacks import EarlyStopping
from training.losses import BatchHardTripletLoss, IDCrossEntropyLoss
from utils.reproducibility import seed_everything


@dataclass
class ProbeArtifacts:
    state_dict: dict[str, Tensor]
    class_map: dict[str, int]
    token_dim: int
    embed_dim: int
    config: dict[str, Any]

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict,
                "class_map": self.class_map,
                "token_dim": self.token_dim,
                "embed_dim": self.embed_dim,
                "config": self.config,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProbeArtifacts":
        d = torch.load(str(path), weights_only=False)
        return cls(
            state_dict=d["state_dict"],
            class_map=d["class_map"],
            token_dim=d["token_dim"],
            embed_dim=d["embed_dim"],
            config=d.get("config", {}),
        )


def build_head(artifacts: ProbeArtifacts) -> SharedReIDHead:
    head = SharedReIDHead(
        token_dim=artifacts.token_dim,
        embed_dim=artifacts.embed_dim,
        num_classes=len(artifacts.class_map),
    )
    head.load_state_dict(artifacts.state_dict)
    head.eval()
    return head


def load_cached_features(features_dir: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in Path(features_dir).glob("*.pt"):
        d = torch.load(str(p), weights_only=False)
        out[d["tracklet_id"]] = d
    return out


def collate_tokens(token_list: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad a batch of [T_i, D] token sequences to [B, T_max, D] + bool mask."""
    max_t = max(t.shape[0] for t in token_list)
    d = token_list[0].shape[-1]
    tokens = torch.zeros(len(token_list), max_t, d)
    mask = torch.zeros(len(token_list), max_t, dtype=torch.bool)
    for i, t in enumerate(token_list):
        n = t.shape[0]
        tokens[i, :n] = t
        mask[i, :n] = True
    return tokens, mask


def train_probe(
    features_dir: str | Path,
    manifests: Sequence[TrackletManifest],
    token_dim: int,
    embed_dim: int = 512,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4,
    id_ce_weight: float = 1.0,
    triplet_weight: float = 1.0,
    patience: int = 5,
    seed: int = 42,
    max_tracklets: int | None = None,
    device: str = "cpu",
) -> ProbeArtifacts:
    """Train the shared head on cached features of TRAIN-split tracklets.

    Features are keyed by ``tracklet_id`` in the cache (spec section 13).
    Only training identities are used for the classifier (spec section 12).
    """
    seed_everything(seed)
    feats = load_cached_features(features_dir)
    train_manifests = [
        m for m in manifests
        if m.split == "train" and m.tracklet_id in feats
    ]
    if max_tracklets is not None:
        train_manifests = train_manifests[:max_tracklets]
    if not train_manifests:
        raise ValueError(
            f"no train-split tracklets with cached features under {features_dir}"
        )
    class_map = {
        mid: i for i, mid in enumerate(sorted({m.vessel_id for m in train_manifests}))
    }
    labels = torch.tensor([class_map[m.vessel_id] for m in train_manifests])

    head = SharedReIDHead(
        token_dim=token_dim, embed_dim=embed_dim, num_classes=len(class_map)
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    ce_loss = IDCrossEntropyLoss()
    triplet_loss = BatchHardTripletLoss(margin=0.3)
    early_stop = EarlyStopping(patience=patience, mode="min")
    history: list[float] = []

    for epoch in range(epochs):
        rng = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(len(train_manifests), generator=rng)
        head.train()
        total = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idxs = perm[i : i + batch_size]
            tokens, mask = collate_tokens(
                [feats[train_manifests[j].tracklet_id]["features"] for j in idxs]
            )
            tokens = tokens.to(device)
            mask = mask.to(device)
            out = head(tokens, mask)
            loss = (
                id_ce_weight * ce_loss(out["logits"], labels[idxs].to(device))
                + triplet_weight * triplet_loss(out["embedding"], labels[idxs].to(device))
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_batches += 1
        mean_loss = total / max(1, n_batches)
        history.append(mean_loss)
        if early_stop(mean_loss, epoch):
            break

    return ProbeArtifacts(
        state_dict=head.state_dict(),
        class_map=class_map,
        token_dim=token_dim,
        embed_dim=embed_dim,
        config={
            "epochs_run": len(history),
            "loss_history": history,
            "seed": seed,
            "train_tracklets": len(train_manifests),
            "num_classes": len(class_map),
        },
    )
