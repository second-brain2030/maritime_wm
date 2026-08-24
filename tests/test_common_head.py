import torch

from models.common_head import SharedReIDHead


def test_head_shapes_and_l2_norm():
    head = SharedReIDHead(token_dim=64, embed_dim=512, num_heads=4, num_layers=2, num_classes=10)
    tokens = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8, dtype=torch.bool)
    out = head(tokens, mask)
    assert out["embedding"].shape == (2, 512)
    assert torch.allclose(out["embedding"].norm(dim=-1), torch.ones(2), atol=1e-5)
    assert out["logits"].shape == (2, 10)


def test_head_with_padding_mask():
    head = SharedReIDHead(token_dim=64, embed_dim=128, num_heads=4, num_layers=1)
    tokens = torch.randn(2, 8, 64)
    mask = torch.tensor([[True] * 8, [True] * 4 + [False] * 4])
    out = head(tokens, mask)
    assert out["embedding"].shape == (2, 128)
    assert torch.allclose(out["embedding"].norm(dim=-1), torch.ones(2), atol=1e-5)


def test_head_no_classifier():
    head = SharedReIDHead(token_dim=64, embed_dim=128, num_heads=4, num_layers=1)
    out = head(torch.randn(1, 8, 64))
    assert "logits" not in out
