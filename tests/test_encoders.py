import pytest
import torch

from models import encoder_registry
from models.cnn_reid import CnnReidEncoder
from models.transformers_encoders import DinoV2Encoder, SigLIPEncoder


def test_registry_has_all_arms():
    names = encoder_registry.names()
    assert "cnn_reid" in names
    assert "dinov2" in names
    assert "siglip" in names
    assert "vjepa_encoder" in names
    assert "openvla_vision" in names


def test_cnn_encoder_shapes_no_download():
    enc = encoder_registry.create("cnn_reid", pretrained=False)
    assert enc.name == "cnn_reid_resnet50"
    assert enc.embedding_dim == 2048
    frames = torch.rand(1, 2, 3, 64, 64)  # [B, T, C, H, W]
    x = enc.preprocess(frames)
    assert x.shape == (1, 2, 3, 224, 224)
    tokens = enc.encode_observed(frames, None)
    assert tokens.shape == (1, 2, 2048)
    assert not torch.isnan(tokens).any()


def test_cnn_backbone_validation():
    with pytest.raises(NotImplementedError):
        encoder_registry.create("cnn_reid", backbone="resnet50_ibn_a")


def test_dinov2_constructor_no_download():
    enc = DinoV2Encoder()
    assert enc.name == "dinov2"
    assert enc.embedding_dim == 768
    assert enc.checkpoint == "facebook/dinov2-base"


def test_siglip_constructor_no_download():
    enc = SigLIPEncoder()
    assert enc.name == "siglip"
    assert enc.embedding_dim == 768


def test_vjepa_blocked_by_api():
    enc = encoder_registry.create("vjepa_encoder")
    with pytest.raises(NotImplementedError):
        enc.encode_observed(torch.rand(1, 2, 3, 64, 64), None)


def test_openvla_feature_source_validated():
    with pytest.raises(ValueError):
        encoder_registry.create("openvla_vision", feature_source="nope")
