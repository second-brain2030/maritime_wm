import torch

from models.common_head import SharedReIDHead


def test_head_shapes_and_l2_norm():
    head = SharedReIDHead(input_dim=64, num_classes=10, embed_dim=512, num_heads=4)
    tokens = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8, dtype=torch.bool)
    out = head(tokens, mask)
    assert out["embedding"].shape == (2, 512)
    assert torch.allclose(out["embedding"].norm(dim=-1), torch.ones(2), atol=1e-5)
    assert out["logits"].shape == (2, 10)


def test_head_with_padding_mask():
    head = SharedReIDHead(input_dim=64, num_classes=5, embed_dim=128, num_heads=4)
    tokens = torch.randn(2, 8, 64)
    mask = torch.tensor([[True] * 8, [True] * 4 + [False] * 4])
    out = head(tokens, mask)
    assert out["embedding"].shape == (2, 128)
    assert torch.allclose(out["embedding"].norm(dim=-1), torch.ones(2), atol=1e-5)


def test_head_logits_optional_and_get_embedding():
    head = SharedReIDHead(input_dim=64, num_classes=10, embed_dim=128, num_heads=4)
    out = head(torch.randn(2, 8, 64))
    assert out["logits"].shape == (2, 10)
    assert head(torch.randn(2, 8, 64), return_logits=False)["logits"] is None
    emb = head.get_embedding(torch.randn(2, 8, 64))
    assert emb.shape == (2, 128)
    assert not emb.requires_grad


def test_head_mean_pooler_variant():
    head = SharedReIDHead(input_dim=64, num_classes=10, embed_dim=128, pooler="mean")
    out = head(torch.randn(2, 8, 64))
    assert out["embedding"].shape == (2, 128)
