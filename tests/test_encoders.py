import pytest
import torch

from models import encoder_registry
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
    assert enc.name == "cnn_reid"
    assert enc.embedding_dim == 2048
    frames = torch.rand(1, 2, 3, 64, 64)  # [B, T, C, H, W]
    x = enc.preprocess(frames)
    assert x.shape == (1, 2, 3, 224, 224)
    tokens = enc.encode_observed(frames, None)
    assert tokens.shape == (1, 2, 2048)
    assert not torch.isnan(tokens).any()


def test_cnn_backbone_fallback_without_torchreid():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        enc = encoder_registry.create("cnn_reid", backbone="osnet_x1_0", pretrained=False)
    assert enc.name == "cnn_reid"
    assert enc.embedding_dim == 2048
    assert any("torchreid" in str(x.message) for x in w)


def test_dinov2_constructor_no_download():
    enc = DinoV2Encoder()
    assert enc.name == "dinov2"
    assert enc.embedding_dim == 768
    assert enc.checkpoint == "facebook/dinov2-base"


def test_siglip_constructor_no_download():
    enc = SigLIPEncoder()
    assert enc.name == "siglip"
    assert enc.embedding_dim == 768


def test_vjepa_class_attributes_no_download():
    # Attribute-level checks only: instantiating the adapter loads 1.6 GB of
    # weights (submodule models/vjepa2 + timm), which belongs in the run
    # environment, not CI.
    from models.vjepa_adapter import VJEPAEncoderAdapter
    from models.vjepa_predictor_adapter import VJEPAPredictorAdapter

    assert VJEPAEncoderAdapter.name == "vjepa_encoder_vitb384"
    assert VJEPAEncoderAdapter.embedding_dim == 768
    assert VJEPAEncoderAdapter.CHECKPOINT == "vjepa2_1_vitb_dist_vitG_384"
    assert VJEPAPredictorAdapter.embedding_dim == 1664  # teacher embedding space
    assert VJEPAPredictorAdapter.CHECKPOINT == "vjepa2_1_vitb_dist_vitG_384"


def test_vjepa_registry_names():
    names = encoder_registry.names()
    assert "vjepa_encoder" in names
    assert "vjepa_predictor" in names


def test_openvla_feature_source_validated():
    with pytest.raises(ValueError):
        encoder_registry.create("openvla_vision", feature_source="nope")
