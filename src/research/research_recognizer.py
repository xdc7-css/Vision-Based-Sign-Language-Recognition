"""Real-Time Webcam Recognition Adapter for Research Models (SVM & 1D CNN).

Provides live inference on webcam frames using:
- Rolling buffer of T=30 frames
- 184 MediaPipe landmark extraction (736 dims)
- Adapter for both Model A (SVM) and Model B (1D Temporal CNN)
- Seamless integration with PredictionStabilizer and ArabicTextBuilder
"""

from __future__ import annotations

import collections
import logging
from typing import Optional

import numpy as np

from src.domain.recognition import GestureCategory, RecognitionResult
from src.research.cnn_pipeline import ResearchCNNPipeline
from src.research.karsl_dataset import SELECTED_RESEARCH_CLASSES
from src.research.preprocessing import ResearchFeatureExtractor
from src.research.svm_pipeline import ResearchSVMPipeline

logger = logging.getLogger("arsl.research.recognizer")


class ResearchSignRecognizer:
    """Live inference recognizer running either Research SVM or Research CNN."""

    def __init__(
        self,
        model_type: str = "svm",  # "svm" or "cnn"
        confidence_threshold: float = 0.60,
        seq_len: int = 30,
        feat_dim: int = 736,
    ) -> None:
        self.model_type = model_type.lower()
        self.confidence_threshold = confidence_threshold
        self.seq_len = seq_len
        self.feat_dim = feat_dim

        self.svm_pipeline = ResearchSVMPipeline(feature_schema="hands_only")
        self.cnn_pipeline = ResearchCNNPipeline(feature_schema="hands_only")
        self.extractor = ResearchFeatureExtractor(seq_len=seq_len, feat_dim=feat_dim, feature_schema="hands_only")

        self.frame_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=seq_len)
        self.classes = SELECTED_RESEARCH_CLASSES
        self._is_ready = False
        self._processor = None
        self.load_active_model()

    @property
    def is_loaded(self) -> bool:
        return self._is_ready

    @property
    def _buffer(self) -> collections.deque[np.ndarray]:
        return self.frame_buffer

    def set_model_type(self, model_type: str) -> bool:
        """Switch active research algorithm ('svm' or 'cnn')."""
        self.model_type = model_type.lower()
        self.frame_buffer.clear()
        return self.load_active_model()

    def load_active_model(self) -> bool:
        """Load weights for the chosen research algorithm."""
        if "cnn" in self.model_type:
            ok = self.cnn_pipeline.load_model()
            if ok:
                self.classes = self.cnn_pipeline.classes
            self._is_ready = ok
            return ok
        else:
            ok = self.svm_pipeline.load_model()
            if ok:
                self.classes = self.svm_pipeline.classes
            self._is_ready = ok
            return ok

    def is_trained(self) -> bool:
        """Check if active research model checkpoint exists on disk."""
        if "cnn" in self.model_type:
            return self.cnn_pipeline.is_trained()
        return self.svm_pipeline.is_trained()

    def get_classes(self) -> list[str]:
        return [c["name_ar"] for c in self.classes]

    def add_frame_features(self, frame_features: np.ndarray) -> None:
        """Append a single frame's 736-dim features to the temporal rolling buffer."""
        if len(frame_features) != self.feat_dim:
            # Handle padding / slicing to exact feature dimension
            padded = np.zeros(self.feat_dim, dtype=np.float32)
            valid_len = min(len(frame_features), self.feat_dim)
            padded[:valid_len] = frame_features[:valid_len]
            frame_features = padded

        self.frame_buffer.append(frame_features.astype(np.float32))

    def predict_window(self) -> Optional[RecognitionResult]:
        """Perform classification on current temporal sliding window."""
        if not self._is_ready or len(self.frame_buffer) < 10:
            return None

        # Resample buffer to target seq_len (30 frames)
        seq = self.extractor.resample_sequence(list(self.frame_buffer), target_len=self.seq_len)

        try:
            if "cnn" in self.model_type:
                class_idx, conf, name_ar, name_en = self.cnn_pipeline.predict_sequence(seq)
            else:
                class_idx, conf, name_ar, name_en = self.svm_pipeline.predict_sequence(seq)

            if conf < self.confidence_threshold:
                return None

            return RecognitionResult(
                gesture_id=f"karsl_{class_idx:04d}",
                arabic_text=name_ar,
                category=GestureCategory.DYNAMIC_GESTURE,
                confidence=conf,
            )
        except Exception as e:
            logger.warning(f"Error during research model inference: {e}")
            return None

    def reset_buffer(self) -> None:
        """Clear temporal frame buffer."""
        self.frame_buffer.clear()

    def process_frame(
        self,
        frame: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[RecognitionResult]:
        """Convenience method to extract MediaPipe landmarks and perform rolling inference."""
        if threshold is not None:
            self.confidence_threshold = threshold

        if self._processor is None:
            from src.infrastructure.recognition.karsl_recognizer import LandmarkerProcessor
            try:
                self._processor = LandmarkerProcessor("data/models/karsl")
            except Exception as e:
                logger.warning(f"Failed to initialize LandmarkerProcessor in recognizer: {e}")

        if self._processor is not None:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
            kps, _ = self._processor.extract_frame_keypoints(rgb)
            self.add_frame_features(kps.flatten())
            return self.predict_window()
        return None

