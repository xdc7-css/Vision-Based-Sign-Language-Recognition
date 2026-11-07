"""MultimodalSignRecognizer — multimodal dual-hand and face gesture classification engine."""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.domain.multimodal import (
    FEATURE_DIM_HANDS_FACE,
    ModalityMode,
    MultimodalFeatureSchema,
    MultimodalFrame,
)
from src.domain.recognition import GestureCategory, RecognitionResult
from src.infrastructure.vision.feature_extractor import MultimodalFeatureExtractor


@dataclass
class MultimodalModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_time_ms: float
    inference_time_ms: float
    model_type: str
    num_classes: int
    num_samples: int


class MultimodalSignRecognizer:
    """Classifies Arabic sign language gestures from synchronized hands and face features.

    Completely isolated from the legacy single-hand Iraqi classifier, operating with
    its own versioned checkpoint and feature schema.
    """

    def __init__(
        self,
        model_dir: str | Path = "data/models/multimodal",
        confidence_threshold: float = 0.5,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._confidence_threshold = confidence_threshold
        self._logger = logging.getLogger(__name__)

        self._model: Any = None
        self._label_encoder: Any = None
        self._scaler: Any = None
        self._schema: Optional[MultimodalFeatureSchema] = None
        self._metrics: Optional[dict[str, Any]] = None
        self._classes: list[str] = []
        self._feature_dim: int = FEATURE_DIM_HANDS_FACE

        self._extractor = MultimodalFeatureExtractor(ModalityMode.TWO_HANDS_FACE)
        self.load_model()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._label_encoder is not None

    @property
    def is_trained(self) -> bool:
        return self.is_loaded

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def metrics(self) -> Optional[dict[str, Any]]:
        return self._metrics

    def load_model(self) -> bool:
        """Load trained multimodal checkpoint from disk."""
        model_path = self._model_dir / "static_model.pkl"
        if not model_path.exists():
            self._logger.debug(f"Multimodal checkpoint not found at {model_path}")
            return False

        try:
            with open(model_path, "rb") as f:
                checkpoint = pickle.load(f)

            self._model = checkpoint.get("model")
            self._label_encoder = checkpoint.get("label_encoder")
            self._scaler = checkpoint.get("scaler")
            self._schema = checkpoint.get("schema")
            self._metrics = checkpoint.get("metrics")
            self._feature_dim = checkpoint.get("feature_dim", FEATURE_DIM_HANDS_FACE)

            if self._label_encoder is not None and hasattr(self._label_encoder, "classes_"):
                self._classes = [str(c) for c in self._label_encoder.classes_]

            # Sync extractor mode
            if self._schema and hasattr(self._schema, "mode"):
                try:
                    mode_enum = ModalityMode(self._schema.mode)
                    self._extractor = MultimodalFeatureExtractor(mode_enum)
                except Exception:
                    pass

            self._logger.info(
                f"Multimodal classifier loaded successfully ({len(self._classes)} classes, {self._feature_dim} dims)"
            )
            return True
        except Exception as e:
            self._logger.error(f"Error loading multimodal model from {model_path}: {e}")
            self._model = None
            self._label_encoder = None
            return False

    @staticmethod
    def _make_invalid(reason: str = "") -> RecognitionResult:
        return RecognitionResult(
            gesture_id="",
            arabic_text="",
            category=GestureCategory.STATIC_WORD,
            confidence=0.0,
            is_rejected=True,
            rejection_reason=reason,
        )

    def predict(
        self,
        frame: MultimodalFrame,
        confidence_threshold: float = 0.0,
    ) -> RecognitionResult:
        """Run multimodal inference on a single synchronized observation.

        Args:
            frame: Synchronized MultimodalFrame.
            confidence_threshold: Minimum confidence threshold.

        Returns:
            RecognitionResult with recognized Arabic word or invalid flag.
        """
        if not self.is_loaded:
            return self._make_invalid("Multimodal Model Not Trained")

        # In multimodal recognition, at least one hand must be present
        if not frame.has_left_hand and not frame.has_right_hand:
            return self._make_invalid("No hands detected")

        start_time = time.perf_counter()

        # 1. Extract normalized feature vector
        features = self._extractor.extract(frame)
        if features.shape[0] != self._feature_dim:
            self._logger.warning(
                f"Feature dimension mismatch: expected {self._feature_dim}, got {features.shape[0]}"
            )
            return self._make_invalid("Feature dimension mismatch")

        # 2. Scale features if scaler exists
        X = features.reshape(1, -1)
        if self._scaler is not None:
            try:
                X = self._scaler.transform(X)
            except Exception:
                pass

        # 3. Model Prediction
        try:
            pred_idx = self._model.predict(X)[0]
            if self._label_encoder is not None:
                class_name = str(self._label_encoder.inverse_transform([pred_idx])[0])
            else:
                class_name = str(pred_idx)

            # Probabilities
            if hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(X)[0]
                confidence = float(np.max(probs))
            else:
                confidence = 1.0

        except Exception as e:
            self._logger.error(f"Inference prediction error: {e}")
            return self._make_invalid(f"Prediction error: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        thresh = confidence_threshold if confidence_threshold > 0.0 else self._confidence_threshold
        if confidence < thresh:
            return self._make_invalid("Below confidence threshold")

        display_text = class_name.replace("_", " ")

        return RecognitionResult(
            gesture_id=f"multimodal_{class_name}",
            category=GestureCategory.STATIC_WORD,
            arabic_text=display_text,
            confidence=confidence,
            is_rejected=False,
            processing_time_ms=elapsed_ms,
            extra={
                "class_name": class_name,
                "decision_path": "multimodal_iraqi",
                "inference_time_ms": elapsed_ms,
                "has_left_hand": frame.has_left_hand,
                "has_right_hand": frame.has_right_hand,
                "has_face": frame.has_face,
                "has_pose": frame.has_pose,
            },
        )
