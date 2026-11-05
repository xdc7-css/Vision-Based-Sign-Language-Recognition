"""BaseInferenceEngine — abstract interface for all inference backends.

Supports:
    - ONNX Runtime (CPU/CUDA)
    - TensorRT (planned)
    - Raw sklearn/PyTorch fallback

All recognizers delegate to an InferenceEngine for production inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class InferenceEngineConfig:
    provider: str = "cpu"
    device_id: int = 0
    intra_op_threads: int = 4
    inter_op_threads: int = 2
    enable_profiling: bool = False


@dataclass
class InferenceResult:
    label_id: int
    label_name: str
    confidence: float
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    extra: dict = field(default_factory=dict)


class BaseInferenceEngine(ABC):
    """Abstract base for all inference backends.

    Subclasses must implement predict(), predict_batch(),
    load(), and unload().
    """

    @abstractmethod
    def predict(self, features: np.ndarray, threshold: float = 0.0) -> InferenceResult:
        """Run inference on a single sample.

        Args:
            features: Input features (n_features,).
            threshold: Minimum confidence to accept.

        Returns:
            InferenceResult with label_id, label_name, confidence.
        """

    @abstractmethod
    def predict_batch(
        self, features: np.ndarray, threshold: float = 0.0
    ) -> list[InferenceResult]:
        """Run inference on a batch of samples.

        Args:
            features: (n_samples, n_features).
            threshold: Minimum confidence to accept.

        Returns:
            List of InferenceResult, one per sample.
        """

    @abstractmethod
    def load(self, path: str) -> bool:
        """Load a model from disk.

        Args:
            path: Path to model file (.onnx, .pkl, .pt).

        Returns:
            True if loaded successfully.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Model metadata (input shape, output names, version, etc.)."""

    @property
    @abstractmethod
    def config(self) -> InferenceEngineConfig:
        """Current engine configuration."""
