"""Model package: encoders, shared head, baselines, registry."""
try:  # installed package, or src on sys.path (pytest via conftest)
    from utils.registry import Registry
except ModuleNotFoundError as exc:
    # repo-root namespace usage: `from src.models...`; only fall back when the
    # failure is about the top-level `utils` package itself, not a genuine
    # error inside it (e.g. a missing transitive dependency).
    if exc.name is None or not exc.name.startswith("utils"):
        raise
    from ..utils.registry import Registry

from .interfaces import TrackletEncoder
from .common_head import SharedReIDHead
from .cnn_reid import CNNReIDEncoder
from .vjepa_adapter import VJEPAEncoderAdapter
from .vjepa_predictor_adapter import VJEPAPredictorAdapter
from .openvla_adapter import OpenVLAVisionAdapter
from .transformers_encoders import DinoV2Encoder, SigLIPEncoder
from .kinematic import (
    ConstantVelocityKalman,
    KinematicState,
    predict_across_gap,
    propose_search_window,
)

encoder_registry = Registry("tracklet_encoders")
encoder_registry.register("cnn_reid", CNNReIDEncoder)
encoder_registry.register("vjepa_encoder", VJEPAEncoderAdapter)
encoder_registry.register("vjepa_predictor", VJEPAPredictorAdapter)
encoder_registry.register("openvla_vision", OpenVLAVisionAdapter)
encoder_registry.register("dinov2", DinoV2Encoder)
encoder_registry.register("siglip", SigLIPEncoder)

__all__ = [
    "TrackletEncoder",
    "SharedReIDHead",
    "CNNReIDEncoder",
    "VJEPAEncoderAdapter",
    "VJEPAPredictorAdapter",
    "OpenVLAVisionAdapter",
    "DinoV2Encoder",
    "SigLIPEncoder",
    "ConstantVelocityKalman",
    "KinematicState",
    "predict_across_gap",
    "propose_search_window",
    "encoder_registry",
]
