"""Model package: encoders, shared head, baselines, registry."""
from utils.registry import Registry
from .interfaces import TrackletEncoder
from .common_head import SharedReIDHead
from .cnn_reid import CnnReidEncoder
from .vjepa_adapter import VJEPAEncoderAdapter
from .vjepa_predictor_adapter import VJEPAPredictorAdapter
from .openvla_adapter import OpenVLAVisionAdapter

encoder_registry = Registry("tracklet_encoders")
encoder_registry.register("cnn_reid", CnnReidEncoder)
encoder_registry.register("vjepa_encoder", VJEPAEncoderAdapter)
encoder_registry.register("vjepa_predictor", VJEPAPredictorAdapter)
encoder_registry.register("openvla_vision", OpenVLAVisionAdapter)

__all__ = [
    "TrackletEncoder",
    "SharedReIDHead",
    "CnnReidEncoder",
    "VJEPAEncoderAdapter",
    "VJEPAPredictorAdapter",
    "OpenVLAVisionAdapter",
    "encoder_registry",
]
