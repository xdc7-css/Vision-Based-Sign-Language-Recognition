"""ONNXInferenceEngine — production inference via ONNX Runtime.

Supports CPU and CUDA providers. Loads .onnx models exported by
ModelExporter. Provides predict() and predict_batch() via the
BaseInferenceEngine ABC.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.infrastructure.inference.base import (
    BaseInferenceEngine,
    InferenceEngineConfig,
    InferenceResult,
)


class ONNXInferenceEngine(BaseInferenceEngine):
    """Inference engine wrapping onnxruntime.InferenceSession.

    Usage:
        engine = ONNXInferenceEngine()
        engine.load("model.onnx")
        result = engine.predict(features)
    """

    def __init__(self, config: Optional[InferenceEngineConfig] = None) -> None:
        self._config = config or InferenceEngineConfig()
        self._session: Any = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._input_shape: tuple[int, ...] = ()
        self._output_shape: tuple[int, ...] = ()
        self._class_names: list[str] = []
        self._model_path: str = ""
        self._logger = logging.getLogger(__name__)

    # ── BaseInferenceEngine ──────────────────────────────────────────

    def predict(self, features: np.ndarray, threshold: float = 0.0) -> InferenceResult:
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if features.ndim == 1:
            features = features[np.newaxis, :]

        ort_inputs = {self._input_name: features.astype(np.float32)}
        ort_outputs = self._session.run([self._output_name], ort_inputs)[0]

        if ort_outputs.ndim == 2:
            probs = self._softmax(ort_outputs[0])
        else:
            probs = np.array([float(ort_outputs)])

        confidence = float(np.max(probs))
        label_id = int(np.argmax(probs))
        label_name = self._class_names[label_id] if label_id < len(self._class_names) else str(label_id)

        return InferenceResult(
            label_id=label_id,
            label_name=label_name,
            confidence=confidence,
            probabilities=probs,
        )

    def predict_batch(self, features: np.ndarray, threshold: float = 0.0) -> list[InferenceResult]:
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if features.ndim == 1:
            features = features[np.newaxis, :]

        ort_inputs = {self._input_name: features.astype(np.float32)}
        ort_outputs = self._session.run([self._output_name], ort_inputs)[0]

        results: list[InferenceResult] = []
        for i in range(ort_outputs.shape[0]):
            if ort_outputs.ndim == 2:
                probs = self._softmax(ort_outputs[i])
            else:
                probs = np.array([float(ort_outputs[i])])
            confidence = float(np.max(probs))
            label_id = int(np.argmax(probs))
            label_name = self._class_names[label_id] if label_id < len(self._class_names) else str(label_id)
            results.append(InferenceResult(
                label_id=label_id, label_name=label_name,
                confidence=confidence, probabilities=probs,
            ))

        return results

    def load(self, path: str) -> bool:
        try:
            import onnxruntime as ort

            model_path = Path(path)
            if not model_path.exists():
                self._logger.error(f"Model not found: {path}")
                return False

            # Configure session options
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = self._config.intra_op_threads
            sess_opts.inter_op_num_threads = self._config.inter_op_threads
            sess_opts.enable_profiling = self._config.enable_profiling
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Select provider
            if self._config.provider == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            self._session = ort.InferenceSession(str(path), sess_opts, providers=providers)
            self._model_path = str(path)

            # Introspect input/output
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._input_shape = self._session.get_inputs()[0].shape
            self._output_shape = self._session.get_outputs()[0].shape

            # Try to load class names from metadata
            self._class_names = self._load_class_names()

            self._logger.info(
                f"ONNX model loaded: {path} "
                f"(input: {self._input_name}{self._input_shape}, "
                f"output: {self._output_name}{self._output_shape}, "
                f"provider: {providers[0]})"
            )
            return True

        except ImportError:
            self._logger.error("onnxruntime not installed")
            return False
        except Exception as e:
            self._logger.error(f"Failed to load ONNX model: {e}")
            return False

    def unload(self) -> None:
        self._session = None
        self._model_path = ""
        self._input_name = ""
        self._output_name = ""
        self._input_shape = ()
        self._output_shape = ()
        self._class_names = []

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def metadata(self) -> dict:
        return {
            "model_path": self._model_path,
            "input_name": self._input_name,
            "input_shape": self._input_shape,
            "output_name": self._output_name,
            "output_shape": self._output_shape,
            "num_classes": len(self._class_names),
            "class_names": self._class_names,
            "provider": self._config.provider,
            "engine": "onnxruntime",
        }

    @property
    def config(self) -> InferenceEngineConfig:
        return self._config

    # ── Private ─────────────────────────────────────────────────────

    def _load_class_names(self) -> list[str]:
        try:
            meta = self._session.get_modelmeta()
            custom_meta = getattr(meta, "custom_metadata_map", {}) or {}
            names_str = custom_meta.get("class_names", "")
            if names_str:
                return names_str.split(",")
        except Exception:
            pass
        return []

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
