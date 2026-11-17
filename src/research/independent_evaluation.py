"""Independent Human Evaluation System for Vision-Based Arabic Sign Language Recognition.

Scientific Protocol:
1. Genuine Independent Evaluation: Tests frozen SVM and 1D Temporal CNN models on newly
   recruited human volunteers who were NOT part of KArSL-502 training (Signer01, Signer02, Signer03).
2. Protocol: 10 predefined Arabic digit classes (0001-0010: صفر to تسعة), 5 repetitions per volunteer.
3. Review Step: Shows reference sign name and allows reviewing recording before accepting.
4. Storage Isolation: All evaluation videos and labels are saved strictly in
   data/research/independent_evaluation/ and are NEVER mixed into training sets or retrained on.
5. Error Disentanglement: Explicitly separates MediaPipe tracking failures (no hands detected)
   from model classification errors.
6. Idle Monitoring: Tests and reports false-positive prediction rates during idle/resting movement.
7. Absolute Honesty: Displays PENDING until actual volunteer recordings exist; never fabricates results.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch

from src.infrastructure.recognition.karsl_recognizer import LandmarkerProcessor
from src.research.cnn_pipeline import ResearchCNNPipeline
from src.research.karsl_dataset import SELECTED_RESEARCH_CLASSES
from src.research.preprocessing import ResearchFeatureExtractor
from src.research.svm_pipeline import ResearchSVMPipeline

logger = logging.getLogger("arsl.research.independent_eval")

BASE_DIR = Path("data/research/independent_evaluation")
RECORDINGS_DIR = BASE_DIR / "recordings"
MANIFEST_FILE = BASE_DIR / "manifest.json"
RESULTS_FILE = BASE_DIR / "evaluation_results.json"
VOLUNTEERS_FILE = BASE_DIR / "volunteers.json"

TARGET_CLASSES = SELECTED_RESEARCH_CLASSES  # 10 classes: 0001 to 0010
TARGET_REPETITIONS = 5


class IndependentEvaluationManager:
    """Manages recording, storage, review, and frozen evaluation on new human volunteers."""

    def __init__(self, base_dir: Path | str = BASE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.recordings_dir = self.base_dir / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.base_dir / "manifest.json"
        self.results_file = self.base_dir / "evaluation_results.json"
        self.volunteers_file = self.base_dir / "volunteers.json"

        self.classes = TARGET_CLASSES
        self.target_reps = TARGET_REPETITIONS

        # Frozen deployment models
        self.svm_pipeline = ResearchSVMPipeline(output_dir="data/models/research/svm", feature_schema="hands_only")
        self.cnn_pipeline = ResearchCNNPipeline(output_dir="data/models/research/cnn", feature_schema="hands_only")
        self.extractor = ResearchFeatureExtractor(feature_schema="hands_only")
        self._processor: Optional[LandmarkerProcessor] = None

    def _get_processor(self) -> LandmarkerProcessor:
        if self._processor is None:
            self._processor = LandmarkerProcessor("data/models/karsl")
        return self._processor

    # ─── Volunteer Management ──────────────────────────────────────────────

    def list_volunteers(self) -> list[dict[str, Any]]:
        """List registered volunteer participants."""
        if not self.volunteers_file.exists():
            return []
        try:
            with open(self.volunteers_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def register_volunteer(self, name: str, notes: str = "") -> dict[str, Any]:
        """Register a new human volunteer for independent evaluation."""
        volunteers = self.list_volunteers()
        volunteer_id = f"vol_{len(volunteers) + 1:03d}_{int(time.time()) % 10000:04d}"
        entry = {
            "volunteer_id": volunteer_id,
            "name": name.strip() or f"Volunteer {len(volunteers) + 1}",
            "notes": notes.strip(),
            "registered_at": time.time(),
            "completed_signs": 0,
            "total_recordings": 0,
        }
        volunteers.append(entry)
        with open(self.volunteers_file, "w", encoding="utf-8") as f:
            json.dump(volunteers, f, ensure_ascii=False, indent=2)
        logger.info(f"Registered new volunteer: {entry['volunteer_id']} ({entry['name']})")
        return entry

    # ─── Manifest & Progress ───────────────────────────────────────────────

    def load_manifest(self) -> list[dict[str, Any]]:
        """Load evaluation recordings manifest."""
        if not self.manifest_file.exists():
            return []
        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save_manifest(self, manifest: list[dict[str, Any]]) -> None:
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def get_volunteer_progress(self, volunteer_id: str) -> dict[str, Any]:
        """Return recording progress for a given volunteer (10 classes x 5 repetitions)."""
        manifest = self.load_manifest()
        v_records = [r for r in manifest if r["volunteer_id"] == volunteer_id and r.get("is_accepted", True)]

        progress_by_sign = {}
        for c in self.classes:
            sign_id = c["sign_id"]
            reps_done = sum(1 for r in v_records if r["sign_id"] == sign_id and not r.get("is_idle", False))
            progress_by_sign[sign_id] = {
                "name_ar": c["name_ar"],
                "name_en": c["name_en"],
                "reps_completed": reps_done,
                "target_reps": self.target_reps,
                "is_complete": reps_done >= self.target_reps,
            }

        idle_reps = sum(1 for r in v_records if r.get("is_idle", False))
        total_recorded = sum(p["reps_completed"] for p in progress_by_sign.values())
        total_target = len(self.classes) * self.target_reps  # 50

        return {
            "volunteer_id": volunteer_id,
            "total_recorded": total_recorded,
            "total_target": total_target,
            "is_complete": total_recorded >= total_target,
            "idle_samples_count": idle_reps,
            "progress_by_sign": progress_by_sign,
        }

    # ─── Recording & Storage ───────────────────────────────────────────────

    def save_recording(
        self,
        volunteer_id: str,
        sign_id: str,
        rep_idx: int,
        frames: list[np.ndarray],
        fps: float = 30.0,
        is_idle: bool = False,
    ) -> dict[str, Any]:
        """Save a reviewed and accepted video recording separately from training data."""
        if len(frames) == 0:
            raise ValueError("Cannot save empty recording.")

        h, w = frames[0].shape[:2]
        dest_dir = self.recordings_dir / volunteer_id / sign_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        tag = "idle" if is_idle else f"rep_{rep_idx:02d}"
        filename = f"{volunteer_id}_{sign_id}_{tag}_{int(time.time())}.mp4"
        file_path = dest_dir / filename

        # Write MP4 video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(file_path), fourcc, fps, (w, h))
        for f in frames:
            # Ensure BGR for VideoWriter
            bgr = f if len(f.shape) == 3 and f.shape[2] == 3 else cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            out.write(bgr)
        out.release()

        # Class metadata
        matched_class = next((c for c in self.classes if c["sign_id"] == sign_id), None)
        class_idx = matched_class["index"] if matched_class else -1
        name_ar = matched_class["name_ar"] if matched_class else "غير معروف"

        record_id = f"ind_{volunteer_id}_{sign_id}_{tag}_{int(time.time())}"
        record_entry = {
            "record_id": record_id,
            "volunteer_id": volunteer_id,
            "sign_id": sign_id,
            "class_idx": class_idx,
            "name_ar": name_ar,
            "rep_idx": rep_idx,
            "file_path": str(file_path.relative_to(self.base_dir.parent.parent)),
            "frame_count": len(frames),
            "width": w,
            "height": h,
            "fps": fps,
            "duration_sec": round(len(frames) / max(fps, 1.0), 2),
            "is_idle": is_idle,
            "is_accepted": True,
            "recorded_at": time.time(),
        }

        manifest = self.load_manifest()
        manifest.append(record_entry)
        self.save_manifest(manifest)

        # Update volunteer stats
        self._update_volunteer_stats(volunteer_id)

        logger.info(f"Saved independent evaluation recording: {file_path} ({len(frames)} frames)")
        return record_entry

    def _update_volunteer_stats(self, volunteer_id: str) -> None:
        volunteers = self.list_volunteers()
        manifest = self.load_manifest()
        v_recs = [r for r in manifest if r["volunteer_id"] == volunteer_id and not r.get("is_idle", False)]
        completed_signs = len(set(r["sign_id"] for r in v_recs))

        for v in volunteers:
            if v["volunteer_id"] == volunteer_id:
                v["total_recordings"] = len(v_recs)
                v["completed_signs"] = completed_signs

        with open(self.volunteers_file, "w", encoding="utf-8") as f:
            json.dump(volunteers, f, ensure_ascii=False, indent=2)

    # ─── Multi-Stage Evaluation Engine ─────────────────────────────────────

    def evaluate_independent_dataset(self) -> dict[str, Any]:
        """Execute genuine evaluation of frozen SVM & CNN on independent volunteer recordings.

        Never retrains models. Disentangles tracking failures from classification errors.
        """
        manifest = self.load_manifest()
        accepted_records = [r for r in manifest if r.get("is_accepted", True)]
        sign_records = [r for r in accepted_records if not r.get("is_idle", False)]
        idle_records = [r for r in accepted_records if r.get("is_idle", False)]

        # Requirement 11: Show PENDING until actual recordings exist. Never generate fake results.
        if len(sign_records) == 0:
            pending_report = {
                "status": "PENDING_VOLUNTEER_RECORDINGS",
                "is_evaluated": False,
                "message": (
                    "No independent volunteer recordings found. In accordance with university research "
                    "integrity standards, performance on unseen individuals remains strictly unmeasured "
                    "until genuine volunteers complete the test protocol."
                ),
                "volunteers_count": len(self.list_volunteers()),
                "accepted_recordings_count": 0,
                "evaluation_date": time.time(),
            }
            with open(self.results_file, "w", encoding="utf-8") as f:
                json.dump(pending_report, f, ensure_ascii=False, indent=2)
            return pending_report

        logger.info(f"Starting Independent Evaluation on {len(sign_records)} volunteer recordings...")
        processor = self._get_processor()

        ground_truth = []
        svm_preds = []
        cnn_preds = []
        tracking_statuses = []  # True = success, False = failed tracking
        sample_details = []

        svm_times = []
        cnn_times = []

        for r in sign_records:
            v_path = Path(r["file_path"])
            if not v_path.exists():
                # Try relative to repo root
                v_path = Path("data/research/independent_evaluation") / Path(r["file_path"]).name
            if not v_path.exists():
                logger.warning(f"Video file missing on disk: {v_path}")
                continue

            cap = cv2.VideoCapture(str(v_path))
            raw_kps = []
            valid_hand_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (640, int(h * (640.0 / w))))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                kps, vis_info = processor.extract_frame_keypoints(rgb)
                raw_kps.append(kps.flatten())
                if vis_info.get("has_hand", False):
                    valid_hand_frames += 1
            cap.release()

            if len(raw_kps) == 0:
                tracking_statuses.append(False)
                continue

            # Stage 1: Tracking Disentanglement
            # Hand tracking is deemed successful if hands are present in >= 5 frames or >= 20% of the video
            hand_tracked = (valid_hand_frames >= 5) or (valid_hand_frames / max(len(raw_kps), 1) >= 0.20)
            tracking_statuses.append(hand_tracked)

            gt_idx = r["class_idx"]
            ground_truth.append(gt_idx)

            # Resample sequence to 30 frames
            seq_arr = np.array(raw_kps, dtype=np.float32)
            seq_30 = self.extractor.resample_sequence(list(seq_arr), target_len=30)

            # Stage 2: Model Inference
            # SVM
            t0 = time.perf_counter()
            s_class, s_conf, s_ar, _ = self.svm_pipeline.predict_sequence(seq_30)
            svm_times.append((time.perf_counter() - t0) * 1000.0)
            svm_preds.append(s_class)

            # CNN
            t0 = time.perf_counter()
            c_class, c_conf, c_ar, _ = self.cnn_pipeline.predict_sequence(seq_30)
            cnn_times.append((time.perf_counter() - t0) * 1000.0)
            cnn_preds.append(c_class)

            sample_details.append({
                "record_id": r["record_id"],
                "volunteer_id": r["volunteer_id"],
                "sign_id": r["sign_id"],
                "ground_truth_class": gt_idx,
                "ground_truth_ar": r["name_ar"],
                "hand_tracked": hand_tracked,
                "hand_frames_ratio": round(valid_hand_frames / max(len(raw_kps), 1), 2),
                "svm_pred_class": s_class,
                "svm_pred_ar": s_ar,
                "svm_conf": round(s_conf, 3),
                "svm_correct": (s_class == gt_idx),
                "cnn_pred_class": c_class,
                "cnn_pred_ar": c_ar,
                "cnn_conf": round(c_conf, 3),
                "cnn_correct": (c_class == gt_idx),
            })

        # Stage 3: Idle False-Positive Testing
        idle_results = self._evaluate_idle_recordings(idle_records, processor)

        # Stage 4: Compile Genuine Metrics
        total_evaluated = len(ground_truth)
        tracked_count = sum(1 for t in tracking_statuses if t)
        tracking_rate = round(tracked_count / max(total_evaluated, 1), 4)

        # Conditioned subset (where MediaPipe successfully tracked hands)
        tracked_indices = [i for i, t in enumerate(tracking_statuses) if t]
        gt_arr = np.array(ground_truth)
        s_pred_arr = np.array(svm_preds)
        c_pred_arr = np.array(cnn_preds)

        # Raw accuracy across ALL recordings (penalizes tracking drops)
        raw_svm_acc = round(float(accuracy_score(gt_arr, s_pred_arr)), 4) if total_evaluated > 0 else 0.0
        raw_cnn_acc = round(float(accuracy_score(gt_arr, c_pred_arr)), 4) if total_evaluated > 0 else 0.0

        # Conditioned accuracy (evaluates classifier when hand was properly visible)
        if len(tracked_indices) > 0:
            cond_gt = gt_arr[tracked_indices]
            cond_s_pred = s_pred_arr[tracked_indices]
            cond_c_pred = c_pred_arr[tracked_indices]

            cond_svm_acc = round(float(accuracy_score(cond_gt, cond_s_pred)), 4)
            cond_svm_f1 = round(float(f1_score(cond_gt, cond_s_pred, average="macro", zero_division=0)), 4)
            cond_cnn_acc = round(float(accuracy_score(cond_gt, cond_c_pred)), 4)
            cond_cnn_f1 = round(float(f1_score(cond_gt, cond_c_pred, average="macro", zero_division=0)), 4)
        else:
            cond_svm_acc = cond_svm_f1 = cond_cnn_acc = cond_cnn_f1 = 0.0

        # Confusion Matrices
        num_classes = len(self.classes)
        svm_cm = confusion_matrix(gt_arr, s_pred_arr, labels=list(range(num_classes))).tolist()
        cnn_cm = confusion_matrix(gt_arr, c_pred_arr, labels=list(range(num_classes))).tolist()

        # Compile specific mistakes
        mistakes = [
            d for d in sample_details if not (d["svm_correct"] and d["cnn_correct"])
        ]

        volunteers_evaluated = list(set(r["volunteer_id"] for r in sign_records))

        report = {
            "status": "EVALUATED_ON_AUTHENTIC_VOLUNTEERS",
            "is_evaluated": True,
            "evaluation_date": time.time(),
            "volunteers_count": len(volunteers_evaluated),
            "volunteers": volunteers_evaluated,
            "total_recordings_evaluated": total_evaluated,
            "mediapipe_tracking": {
                "tracking_success_count": tracked_count,
                "tracking_failure_count": total_evaluated - tracked_count,
                "tracking_success_rate": tracking_rate,
                "explanation": (
                    "Distinguishes MediaPipe landmark loss (hand occlusion or camera angle) "
                    "from algorithm classification errors."
                ),
            },
            "svm_metrics": {
                "raw_overall_accuracy": raw_svm_acc,
                "conditioned_accuracy": cond_svm_acc,
                "conditioned_macro_f1": cond_svm_f1,
                "mean_latency_ms": round(float(np.mean(svm_times)), 2) if svm_times else 0.0,
            },
            "cnn_metrics": {
                "raw_overall_accuracy": raw_cnn_acc,
                "conditioned_accuracy": cond_cnn_acc,
                "conditioned_macro_f1": cond_cnn_f1,
                "mean_latency_ms": round(float(np.mean(cnn_times)), 2) if cnn_times else 0.0,
            },
            "idle_false_positive_monitoring": idle_results,
            "confusion_matrices": {
                "svm": svm_cm,
                "cnn": cnn_cm,
            },
            "mistakes_count": len(mistakes),
            "sample_details": sample_details,
        }

        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Independent Evaluation completed: Tracking={tracking_rate*100:.1f}%, SVM={cond_svm_acc*100:.1f}%, CNN={cond_cnn_acc*100:.1f}%")
        return report

    def _evaluate_idle_recordings(
        self,
        idle_records: list[dict[str, Any]],
        processor: LandmarkerProcessor,
    ) -> dict[str, Any]:
        """Test whether idle/resting movements falsely trigger sign classifications."""
        if len(idle_records) == 0:
            return {
                "idle_recordings_tested": 0,
                "false_positive_count_svm": 0,
                "false_positive_count_cnn": 0,
                "false_positive_rate": 0.0,
                "note": "No idle baseline recordings currently stored. Recommended to capture 5 resting baseline videos.",
            }

        fp_svm = 0
        fp_cnn = 0

        for r in idle_records:
            v_path = Path(r["file_path"])
            if not v_path.exists():
                v_path = Path("data/research/independent_evaluation") / Path(r["file_path"]).name
            if not v_path.exists():
                continue

            cap = cv2.VideoCapture(str(v_path))
            raw_kps = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (640, int(h * (640.0 / w))))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                kps, _ = processor.extract_frame_keypoints(rgb)
                raw_kps.append(kps.flatten())
            cap.release()

            if len(raw_kps) == 0:
                continue

            seq_30 = self.extractor.resample_sequence(list(np.array(raw_kps, dtype=np.float32)), target_len=30)
            # Threshold 0.60
            _, s_conf, _, _ = self.svm_pipeline.predict_sequence(seq_30)
            _, c_conf, _, _ = self.cnn_pipeline.predict_sequence(seq_30)

            if s_conf >= 0.60:
                fp_svm += 1
            if c_conf >= 0.60:
                fp_cnn += 1

        total = len(idle_records)
        return {
            "idle_recordings_tested": total,
            "false_positive_count_svm": fp_svm,
            "false_positive_count_cnn": fp_cnn,
            "svm_false_positive_rate": round(fp_svm / total, 3),
            "cnn_false_positive_rate": round(fp_cnn / total, 3),
        }

    def get_status(self) -> dict[str, Any]:
        """Return current status: PENDING or EVALUATED with full metrics."""
        manifest = self.load_manifest()
        volunteers = self.list_volunteers()

        if self.results_file.exists():
            try:
                with open(self.results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    return {
                        **results,
                        "volunteers_count": len(volunteers),
                        "total_manifest_recordings": len(manifest),
                    }
            except Exception:
                pass

        # Default pending state
        return {
            "status": "PENDING_VOLUNTEER_RECORDINGS",
            "is_evaluated": False,
            "message": (
                "Independent human evaluation remains strictly unmeasured until new volunteer "
                "recordings are collected. No synthetic or training data is substituted."
            ),
            "volunteers_count": len(volunteers),
            "total_manifest_recordings": len(manifest),
            "target_classes": [c["name_ar"] for c in self.classes],
            "target_reps_per_volunteer": self.target_reps,
        }

