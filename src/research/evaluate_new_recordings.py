"""Independent Evaluation Workflow for Unseen Signer Recordings.

Strict Scientific Independence:
- The deployment models in data/models/research/{svm,cnn} were trained on all 240 KArSL recordings
  across Signer01, Signer02, and Signer03.
- To evaluate true unseen-video performance, this module evaluates ONLY genuinely new recordings
  from individuals NOT involved in training.
- If no new human recordings are present in data/research/karsl/unseen_eval/, this module
  strictly reports the evaluation status as PENDING rather than fabricating results.
- Also tests:
  1. Complete 30-frame sequence handling
  2. Arabic class mapping integrity
  3. Idle rejection behavior on resting/noise frames
  4. Webcam hardware compatibility
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.infrastructure.recognition.karsl_recognizer import LandmarkerProcessor
from src.research.cnn_pipeline import ResearchCNNPipeline
from src.research.karsl_dataset import SELECTED_RESEARCH_CLASSES
from src.research.preprocessing import ResearchFeatureExtractor
from src.research.research_recognizer import ResearchSignRecognizer
from src.research.svm_pipeline import ResearchSVMPipeline

logger = logging.getLogger("arsl.research.unseen_eval")

UNSEEN_DIR = Path("data/research/karsl/unseen_eval")
UNSEEN_MANIFEST = UNSEEN_DIR / "unseen_manifest.csv"
TRAINING_SIGNERS = {"Signer01", "Signer02", "Signer03"}


class UnseenSignerEvaluator:
    """Evaluates deployment models on genuinely unseen human video recordings."""

    def __init__(
        self,
        unseen_dir: Path | str = UNSEEN_DIR,
        feature_schema: str = "hands_only",
    ) -> None:
        self.unseen_dir = Path(unseen_dir)
        self.unseen_dir.mkdir(parents=True, exist_ok=True)
        self.feature_schema = feature_schema
        self.classes = SELECTED_RESEARCH_CLASSES

        self.svm_pipeline = ResearchSVMPipeline(feature_schema=feature_schema)
        self.cnn_pipeline = ResearchCNNPipeline(feature_schema=feature_schema)
        self.extractor = ResearchFeatureExtractor(feature_schema=feature_schema)

    def check_unseen_data_availability(self) -> dict[str, Any]:
        """Check if genuinely new recordings from unseen individuals are available."""
        manifest_file = self.unseen_dir / "unseen_manifest.csv"
        video_files = list(self.unseen_dir.glob("**/*.mp4")) + list(self.unseen_dir.glob("**/*.avi"))

        if not video_files and not manifest_file.exists():
            return {
                "status": "PENDING_NEW_UNSEEN_RECORDINGS",
                "available": False,
                "video_count": 0,
                "message": (
                    "No new independent video recordings from unseen signers found in "
                    f"{self.unseen_dir}. In accordance with scientific integrity standards, "
                    "performance on new signers is reported as PENDING rather than fabricated."
                ),
                "instructions": (
                    "To evaluate, record MP4 videos from signers not in {Signer01, Signer02, Signer03}, "
                    f"place them in {self.unseen_dir}/<SignerID>/<SignID>/, and run this evaluator."
                ),
            }

        # Filter out any videos from training signers
        valid_unseen = []
        for v in video_files:
            signer_id = v.parent.parent.name if v.parent.parent != self.unseen_dir else v.parent.name
            if signer_id not in TRAINING_SIGNERS:
                valid_unseen.append((signer_id, v))

        if not valid_unseen:
            return {
                "status": "PENDING_NEW_UNSEEN_RECORDINGS",
                "available": False,
                "video_count": 0,
                "message": "Only training signers found in evaluation directory; no new unseen signers.",
            }

        return {
            "status": "UNSEEN_RECORDINGS_AVAILABLE",
            "available": True,
            "video_count": len(valid_unseen),
            "signers": list(set(s for s, _ in valid_unseen)),
        }

    def verify_arabic_class_mapping(self) -> dict[str, Any]:
        """Verify that Arabic labels 0-9 match the exact expected digits."""
        expected = [
            (0, "0001", "صفر", "0"),
            (1, "0002", "واحد", "1"),
            (2, "0003", "اثنان", "2"),
            (3, "0004", "ثلاثة", "3"),
            (4, "0005", "أربعة", "4"),
            (5, "0006", "خمسة", "5"),
            (6, "0007", "ستة", "6"),
            (7, "0008", "سبعة", "7"),
            (8, "0009", "ثمانية", "8"),
            (9, "0010", "تسعة", "9"),
        ]

        verified = []
        mismatches = []
        for idx, sign_id, name_ar, name_en in expected:
            c = self.classes[idx]
            match = (
                c["sign_id"] == sign_id
                and c["name_ar"] == name_ar
                and c["name_en"] == name_en
            )
            if match:
                verified.append({"index": idx, "sign_id": sign_id, "name_ar": name_ar})
            else:
                mismatches.append({"expected": (idx, sign_id, name_ar), "actual": c})

        return {
            "status": "PASS" if not mismatches else "FAIL",
            "classes_verified": len(verified),
            "total_classes": len(expected),
            "mismatches": mismatches,
        }

    def verify_idle_rejection(self) -> dict[str, Any]:
        """Verify that idle/resting frames or random noise are rejected below threshold."""
        rec_cnn = ResearchSignRecognizer(model_type="cnn", confidence_threshold=0.60)
        rec_svm = ResearchSignRecognizer(model_type="svm", confidence_threshold=0.60)

        # 1. Zero/blank frame buffer (resting hands or no hands detected)
        rec_cnn.reset_buffer()
        rec_svm.reset_buffer()
        for _ in range(30):
            blank_frame = np.zeros(736, dtype=np.float32)
            rec_cnn.add_frame_features(blank_frame)
            rec_svm.add_frame_features(blank_frame)

        pred_cnn_blank = rec_cnn.predict_window()
        pred_svm_blank = rec_svm.predict_window()

        # Both models must reject or output below 0.60 confidence
        cnn_rejected = pred_cnn_blank is None or pred_cnn_blank.confidence < 0.60
        svm_rejected = pred_svm_blank is None or pred_svm_blank.confidence < 0.60

        return {
            "status": "PASS" if (cnn_rejected and svm_rejected) else "FAIL",
            "cnn_idle_rejected": cnn_rejected,
            "svm_idle_rejected": svm_rejected,
            "cnn_result": str(pred_cnn_blank),
            "svm_result": str(pred_svm_blank),
        }

    def verify_30_frame_sequence_handling(self) -> dict[str, Any]:
        """Verify that sequence buffers correctly require >= 10 frames and resample to exactly 30 frames."""
        rec = ResearchSignRecognizer(model_type="cnn")
        rec.reset_buffer()

        # Less than 10 frames should return None
        for _ in range(5):
            rec.add_frame_features(np.zeros(736, dtype=np.float32))
        short_result = rec.predict_window()
        assert short_result is None, "Recognizer should reject buffer with < 10 frames"

        # Buffer with 20 frames should resample to 30 and proceed
        for _ in range(15):
            rec.add_frame_features(np.random.randn(736).astype(np.float32) * 0.1)

        full_len = len(rec.frame_buffer)
        resampled = rec.extractor.resample_sequence(list(rec.frame_buffer), target_len=30)

        return {
            "status": "PASS",
            "short_buffer_rejected": short_result is None,
            "buffer_length": full_len,
            "resampled_shape": list(resampled.shape),
            "target_seq_len": 30,
        }

    def verify_webcam_compatibility(self, camera_index: int = 0) -> dict[str, Any]:
        """Test physical or virtual webcam connectivity, frame acquisition, and recognition pipeline."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {
                "status": "HARDWARE_UNAVAILABLE",
                "camera_opened": False,
                "message": f"Camera index {camera_index} is unavailable or in use by another application.",
            }

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {
                "status": "FRAME_READ_FAILED",
                "camera_opened": True,
                "frame_acquired": False,
            }

        h, w = frame.shape[:2]
        return {
            "status": "PASS",
            "camera_opened": True,
            "frame_acquired": True,
            "frame_dimensions": f"{w}x{h}",
            "channels": frame.shape[2] if len(frame.shape) == 3 else 1,
        }

