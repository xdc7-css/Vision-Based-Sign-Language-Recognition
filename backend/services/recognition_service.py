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
from src.domain.stabilizer import PredictionStabilizer
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

        # Telemetry & Prediction Stabilization
        self._last_time = time.perf_counter()
        self._fps = 0.0
        self.stabilizer = PredictionStabilizer(
            min_confidence=confidence_threshold,
            min_stable_frames=5,
            agreement_threshold=0.80,
            release_frames=3,
            neutral_y_threshold=0.84,
        )

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold across recognizer and stabilizer."""
        self.confidence_threshold = threshold
        self.recognizer.confidence_threshold = threshold
        self.stabilizer.min_confidence = threshold

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

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        client_time_ms: float = 0.0,
        decode_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Process incoming video frame, extract multimodal landmarks, stabilize predictions, and run inference."""
        t0 = time.perf_counter()

        # Calculate FPS with EWMA smoothing
        dt = t0 - self._last_time
        self._last_time = t0
        if dt > 0:
            current_fps = 1.0 / dt
            self._fps = 0.85 * self._fps + 0.15 * current_fps if self._fps > 0 else current_fps

        # Run multimodal detection
        mm_frame = self.detector.detect(frame)
        det_timings = getattr(mm_frame, "detection_timings", {})
        hands_ms = det_timings.get("hands_ms", 0.0)
        face_ms = det_timings.get("face_ms", 0.0)
        pose_ms = det_timings.get("pose_ms", 0.0)

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
        infer_ms = 0.0

        if has_hands and self.recognizer.is_trained:
            t_infer = time.perf_counter()
            res = self.recognizer.predict(mm_frame)
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

            if res is not None and not res.is_rejected and res.arabic_text:
                rec_sign = res.arabic_text
                rec_conf = float(res.confidence)

        # Obtain hand vertical anchor for resting detection
        hand_y: Optional[float] = None
        if mm_frame.right_hand is not None and mm_frame.right_hand.landmarks:
            hand_y = float(mm_frame.right_hand.landmarks[0].y)
        elif mm_frame.left_hand is not None and mm_frame.left_hand.landmarks:
            hand_y = float(mm_frame.left_hand.landmarks[0].y)

        # Update prediction stabilizer state machine
        stab_res = self.stabilizer.update(
            raw_sign=rec_sign,
            confidence=rec_conf,
            has_hands=has_hands,
            hand_y=hand_y,
        )

        # Only emit word into ArabicTextBuilder when confirmed by stabilizer
        if stab_res.emitted_sign:
            self.text_builder.process(TextCommand(GestureInputType.WORD, stab_res.emitted_sign))

        backend_total_ms = (time.perf_counter() - t0) * 1000.0 + decode_ms

        return {
            "frame_id": frame_id,
            "client_time_ms": client_time_ms,
            "recognized_sign": stab_res.candidate_sign,
            "confidence": round(stab_res.confidence, 3),
            "is_valid": bool(stab_res.candidate_sign and stab_res.confidence >= self.confidence_threshold),
            "state": stab_res.state.value,
            "state_progress": round(stab_res.progress, 2),
            "status_label": stab_res.status_label,
            "is_holding": stab_res.is_holding,
            "is_neutral": stab_res.is_neutral,
            "emitted_word": stab_res.emitted_sign,
            "active_model": f"Multimodal Iraqi Classifier ({len(self.recognizer.classes)} signs)",
            "num_classes": len(self.recognizer.classes),
            "latency_ms": round(backend_total_ms, 1),
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
            "perf": {
                "decode_ms": round(decode_ms, 2),
                "hands_ms": round(hands_ms, 2),
                "face_ms": round(face_ms, 2),
                "pose_ms": round(pose_ms, 2),
                "infer_ms": round(infer_ms, 2),
                "backend_total_ms": round(backend_total_ms, 1),
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

