import pytest
import torch
from torch.nn import functional as F

from training.losses import BatchHardTripletLoss, IDCrossEntropyLoss


def test_triplet_identity_no_loss():
    loss_fn = BatchHardTripletLoss(margin=0.3)
    emb = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    assert loss_fn(emb, labels).item() == pytest.approx(0.0, abs=1e-6)


def test_triplet_margin_violation():
    loss_fn = BatchHardTripletLoss(margin=0.3)
    emb = F.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]), dim=-1)
    labels = torch.tensor([0, 0, 1])
    # anchor 0: hardest pos ~0.994, hardest neg 0.0 -> relu(0.3 - 0.994 + 0) = 0
    # anchor 1: hardest pos ~0.994, hardest neg ~0.1 -> relu(0.3 - 0.994 + 0.1) = 0
    # anchor 2: hardest pos 0.0 (none? no: label 1 only itself -> has_pos False -> 0)
    assert loss_fn(emb, labels).item() == pytest.approx(0.0, abs=1e-6)


def test_triplet_all_same_class_zero():
    loss_fn = BatchHardTripletLoss(margin=0.3)
    emb = F.normalize(torch.ones(3, 4), dim=-1)
    labels = torch.tensor([0, 0, 0])
    assert loss_fn(emb, labels).item() == pytest.approx(0.0, abs=1e-6)


def test_triplet_violation_positive():
    loss_fn = BatchHardTripletLoss(margin=0.3)
    emb = F.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), dim=-1
    )  # class 0 anchor vs class 1 pair
    labels = torch.tensor([0, 1, 1])
    loss = loss_fn(emb, labels)
    # anchor 0 has NO positive -> excluded
    # anchors 1,2: pos sim 1.0, hardest neg 0.0 -> 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_id_cross_entropy():
    loss_fn = IDCrossEntropyLoss()
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    assert loss_fn(logits, labels).item() == pytest.approx(0.0, abs=1e-3)
