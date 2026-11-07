"""Multimodal vision infrastructure — detector, feature extractor, and renderer."""

from src.domain.multimodal import (
    MULTIMODAL_SCHEMA_VERSION,
    ModalityConfig,
    ModalityMode,
    MultimodalFeatureSchema,
    MultimodalFrame,
)

__all__ = [
    "MULTIMODAL_SCHEMA_VERSION",
    "ModalityConfig",
    "ModalityMode",
    "MultimodalFeatureSchema",
    "MultimodalFrame",
]
