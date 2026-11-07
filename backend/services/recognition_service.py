"""Recognition and inference service for ArSL Studio."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional
import uuid

import numpy as np

from src.domain.multimodal import (
    FEATURE_DIM_HANDS_FACE,
    ModalityConfig,
    ModalityMode,
    MULTIMODAL_SCHEMA_VERSION,
)
from src.domain.text import GestureInputType, TextCommand
from src.infrastructure.dataset.gesture_registry import sanitize_folder_name
from src.infrastructure.recognition.multimodal_recognizer import MultimodalSignRecognizer
from src.infrastructure.text_builder import ArabicTextBuilder
from src.infrastructure.vision.feature_extractor import MultimodalFeatureExtractor
from src.infrastructure.vision.multimodal_detector import MultimodalDetector

logger = logging.getLogger(__name__)


class RecognitionService:
    """Core real-time inference, landmark extraction, and sentence building engine."""

    def __init__(
        self,
        model_dir: str = "data/models/multimodal",
        confidence_threshold: float = 0.65,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold

        # Multimodal Detector
        config = ModalityConfig(
            mode=ModalityMode.TWO_HANDS_FACE,
            enable_left_hand=True,
            enable_right_hand=True,
            enable_face=True,
            enable_pose=False,
            mirror_camera=False,  # Frontend handles mirroring or we do it explicitly
        )
        self.detector = MultimodalDetector(config=config)
        self.recognizer = MultimodalSignRecognizer(
            model_dir=model_dir,
            confidence_threshold=confidence_threshold,
        )
        self.extractor = MultimodalFeatureExtractor(ModalityMode.TWO_HANDS_FACE)
        self.text_builder = ArabicTextBuilder()

        # Telemetry & Smoothing
        self._last_time = time.time()
        self._fps = 0.0
        self._recent_predictions: list[str] = []
        self._prediction_hold_count = 0
        self._current_stable_sign = ""
        self._current_stable_conf = 0.0
        self._last_added_sign = ""

    def reload_model(self) -> bool:
        """Reload active model from disk."""
        ok = self.recognizer.load_model()
        logger.info(f"Model reloaded: ok={ok}, classes={self.recognizer.classes}")
        return ok

    def get_active_model_info(self) -> dict[str, Any]:
        """Return metadata for the currently active model."""
        metrics = self.recognizer.metrics or {}
        model_type = metrics.get("model_type", "MLP Classifier")
        acc = metrics.get("accuracy")

        updated_at = None
        model_path = self.model_dir / "static_model.pkl"
        if model_path.exists():
            updated_at = model_path.stat().st_mtime

        return {
            "name": "Iraqi Multimodal Classifier",
            "model_type": model_type,
            "classes": self.recognizer.classes,
            "num_classes": len(self.recognizer.classes),
            "accuracy": acc,
            "is_trained": self.recognizer.is_trained,
            "updated_at": updated_at,
        }

    # ── Frame Processing ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Process incoming video frame from browser, extract landmarks, and run inference."""
        t0 = time.time()
        
        # Calculate FPS
        dt = t0 - self._last_time
        self._last_time = t0
        if dt > 0:
            current_fps = 1.0 / dt
            self._fps = 0.85 * self._fps + 0.15 * current_fps if self._fps > 0 else current_fps

        # Run multimodal detection
        mm_frame = self.detector.detect(frame)

        # Build landmark serialization for canvas overlay
        left_lms = None
        if mm_frame.left_hand is not None:
            left_lms = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in mm_frame.left_hand.landmarks
            ]

        right_lms = None
        if mm_frame.right_hand is not None:
            right_lms = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in mm_frame.right_hand.landmarks
            ]

        face_lms = None
        if mm_frame.face_landmarks:
            face_lms = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in mm_frame.face_landmarks
            ]

        pose_lms = None
        if mm_frame.pose_landmarks:
            pose_lms = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in mm_frame.pose_landmarks
            ]

        # Run classifier inference if at least one hand is detected
        has_hands = mm_frame.has_left or mm_frame.has_right
        rec_sign = ""
        rec_conf = 0.0
        is_valid = False

        if has_hands and self.recognizer.is_trained:
            res = self.recognizer.predict(mm_frame)
            if res is not None and res.is_valid and res.arabic_text:
                rec_sign = res.arabic_text
                rec_conf = float(res.confidence)
                is_valid = True

        # Temporal smoothing: require sign to persist for stability
        if is_valid and rec_conf >= self.confidence_threshold:
            if rec_sign == self._current_stable_sign:
                self._prediction_hold_count += 1
            else:
                self._current_stable_sign = rec_sign
                self._current_stable_conf = rec_conf
                self._prediction_hold_count = 1

            # Auto-accumulate word to sentence buffer after 8 consecutive stable frames
            if self._prediction_hold_count >= 8 and self._current_stable_sign != self._last_added_sign:
                self.text_builder.process(TextCommand(GestureInputType.WORD, self._current_stable_sign))
                self._last_added_sign = self._current_stable_sign
        else:
            if self._prediction_hold_count > 0:
                self._prediction_hold_count -= 1
            if self._prediction_hold_count == 0:
                self._current_stable_sign = ""
                self._current_stable_conf = 0.0

        latency_ms = (time.time() - t0) * 1000.0

        return {
            "recognized_sign": self._current_stable_sign or (rec_sign if rec_conf >= 0.5 else ""),
            "confidence": round(self._current_stable_conf or rec_conf, 3),
            "is_valid": is_valid and (rec_conf >= self.confidence_threshold),
            "active_model": f"Multimodal Iraqi Classifier ({len(self.recognizer.classes)} signs)",
            "num_classes": len(self.recognizer.classes),
            "latency_ms": round(latency_ms, 1),
            "fps": round(self._fps, 1),
            "sentence": self.text_builder.full_text,
            "landmarks": {
                "left_hand": left_lms,
                "right_hand": right_lms,
                "face": face_lms,
                "pose": pose_lms,
            },
            "indicators": {
                "has_left": mm_frame.has_left,
                "has_right": mm_frame.has_right,
                "has_face": mm_frame.has_face,
                "has_pose": mm_frame.has_pose,
                "mouth_openness": round(float(mm_frame.mouth_openness), 3),
                "eyebrow_elevation": round(float(mm_frame.eyebrow_elevation), 3),
            },
        }

    # ── Sample Recording Support ─────────────────────────────────────────

    def record_multimodal_sample(self, class_name: str, frame: np.ndarray, category: str = "static") -> Optional[dict[str, Any]]:
        """Extract multimodal features from frame and persist directly to dataset directory."""
        mm_frame = self.detector.detect(frame)
        if not (mm_frame.has_left or mm_frame.has_right):
            return None  # Sample must contain at least one hand

        feature_vector = self.extractor.extract(mm_frame)
        if feature_vector is None or feature_vector.shape[0] != FEATURE_DIM_HANDS_FACE:
            return None

        # Build paths
        folder_name = sanitize_folder_name(class_name)
        target_sub = "dynamic" if category == "dynamic" else "words"
        sample_id = str(uuid.uuid4())[:8]
        sample_dir = Path("data/datasets") / target_sub / folder_name / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save landmarks array (1D static or 2D single frame)
        arr_2d = feature_vector.reshape(1, -1).astype(np.float32)
        np.save(sample_dir / "landmarks.npy", arr_2d)

        # Save metadata.json
        meta = {
            "sample_id": sample_id,
            "gesture_type": category,
            "gesture_label": class_name,
            "dialect": "iraqi",
            "timestamp": time.time(),
            "num_frames": 1,
            "num_hands": (1 if mm_frame.has_left else 0) + (1 if mm_frame.has_right else 0),
            "handedness": "both" if (mm_frame.has_left and mm_frame.has_right) else ("left" if mm_frame.has_left else "right"),
            "duration_seconds": 0.05,
            "fps": 20.0,
            "has_motion": False,
            "total_distance": 0.0,
            "motion_energy": 0.0,
            "schema_version": MULTIMODAL_SCHEMA_VERSION,
            "modality_mode": "two_hands_face",
            "has_face": mm_frame.has_face,
            "has_pose": mm_frame.has_pose,
            "has_left_hand": mm_frame.has_left,
            "has_right_hand": mm_frame.has_right,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return {
            "sample_id": sample_id,
            "class_name": class_name,
            "has_left": mm_frame.has_left,
            "has_right": mm_frame.has_right,
            "has_face": mm_frame.has_face,
            "feature_dim": FEATURE_DIM_HANDS_FACE,
        }

    # ── Sentence Builder Commands ────────────────────────────────────────

    def add_sentence_word(self, word: str) -> str:
        self.text_builder.process(TextCommand(GestureInputType.WORD, word))
        return self.text_builder.full_text

    def delete_last_word(self) -> str:
        self.text_builder.process(TextCommand(GestureInputType.DELETE, ""))
        return self.text_builder.full_text

    def clear_sentence(self) -> str:
        self.text_builder.process(TextCommand(GestureInputType.CLEAR, ""))
        self._last_added_sign = ""
        return self.text_builder.full_text

    def get_sentence(self) -> str:
        return self.text_builder.full_text

