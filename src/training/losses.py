"""Re-ID losses (spec section 12).

L_total = L_cross_entropy + lambda_triplet * L_batch_hard_triplet.
Do not change loss functions per model arm in the headline comparison.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class IDCrossEntropyLoss(nn.Module):
    """Cross-entropy identity classification."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.ce(logits, targets)


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss over L2-normalized embeddings (spec §12)."""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        sim = embeddings @ embeddings.t()  # cosine similarity (L2-normalized input)
        eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
        same = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye

        has_pos = same.any(dim=1)
        pos_sim = sim.masked_fill(~same, float("inf")).min(dim=1).values  # hardest positive
        pos_sim = torch.where(has_pos, pos_sim, torch.zeros_like(pos_sim))
        # hardest negative: exclude same-label pairs AND the anchor's own diagonal
        neg_sim = sim.masked_fill(same | eye, -float("inf")).max(dim=1).values

        losses = F.relu(self.margin - pos_sim + neg_sim)
        if has_pos.any():
            return losses[has_pos].mean()
        return torch.zeros((), device=embeddings.device, requires_grad=True)


class CircleLoss(nn.Module):
    """Optional ablation (spec section 12)."""

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError("Circle loss lands with the training commit")


class SupervisedContrastiveLoss(nn.Module):
    """Optional ablation (spec section 12)."""

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError("SupCon loss lands with the training commit")
