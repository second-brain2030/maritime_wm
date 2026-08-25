"""Re-ID losses (spec section 12).

L_total = L_cross_entropy + lambda_triplet * L_batch_hard_triplet.
Do not change loss functions per model arm in the headline comparison.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class IDCrossEntropyLoss(nn.Module):
    """Cross-entropy identity classification (legacy signature)."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.ce(logits, targets)


class IDCELoss(nn.Module):
    """Identity cross-entropy loss with optional label smoothing.

    Wraps ``nn.CrossEntropyLoss``; the number of classes is needed at
    construction time so the smoothed label distribution is well defined.
    """

    def __init__(self, num_classes: int, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.ce(logits, labels)


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss over squared Euclidean distances (spec §12).

    For each anchor: hardest positive = max distance to a same-label sample
    (excluding itself); hardest negative = min distance to a different-label
    sample. Hinge: ``max(0, d_pos - d_neg + margin)``. Anchors without any
    positive in the batch are excluded from the mean.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be [B, D], got shape {tuple(embeddings.shape)}"
            )
        # Squared Euclidean distance matrix [B, B].
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        dist = (diff * diff).sum(dim=-1)

        eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
        same = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
        has_pos = same.any(dim=1)

        # Hardest positive: max distance among same-label, non-self pairs.
        d_pos = dist.masked_fill(~same, -float("inf")).max(dim=1).values
        # Hardest negative: min distance among different-label pairs.
        d_neg = dist.masked_fill(
            labels.unsqueeze(1) == labels.unsqueeze(0), float("inf")
        ).min(dim=1).values

        losses = F.relu(d_pos - d_neg + self.margin)
        if has_pos.any():
            return losses[has_pos].mean()
        return torch.zeros((), device=embeddings.device, requires_grad=True)


class CombinedLoss(nn.Module):
    """Weighted combination of identity cross-entropy and batch-hard triplet."""

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        triplet_weight: float = 1.0,
        label_smoothing: float = 0.1,
        margin: float = 0.3,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        self.ce = IDCELoss(num_classes=num_classes, label_smoothing=label_smoothing)
        self.triplet = BatchHardTripletLoss(margin=margin)

    def forward(
        self, embeddings: Tensor, logits: Tensor, labels: Tensor
    ) -> dict[str, Tensor]:
        ce_loss = self.ce(logits, labels)
        triplet_loss = self.triplet(embeddings, labels)
        total = self.ce_weight * ce_loss + self.triplet_weight * triplet_loss
        return {"total": total, "ce": ce_loss, "triplet": triplet_loss}


class CircleLoss(nn.Module):
    """Optional ablation (spec section 12)."""

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError("Circle loss lands with the training commit")


class SupervisedContrastiveLoss(nn.Module):
    """Optional ablation (spec section 12)."""

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError("SupCon loss lands with the training commit")
