"""Independent Human Evaluation System for Vision-Based Arabic Sign Language Recognition.

Scientific Protocol & Integrity Guidelines:
1. Genuine Independent Evaluation: Tests frozen SVM and 1D Temporal CNN models on newly
   recruited human volunteers who were NOT part of KArSL-502 training (Signer01, Signer02, Signer03).
2. Protocol: 10 predefined Arabic digit classes (0001-0010: صفر to تسعة), 5 repetitions per volunteer (50 recordings/person).
3. Review & Confirmation: Live preview during recording, video playback during review, and explicit accept/retry/discard.
4. Storage Isolation: All evaluation videos and labels are saved strictly in
   data/research/independent_evaluation/ and are NEVER mixed into training sets or retrained on.
5. Error Disentanglement: Explicitly separates MediaPipe tracking failures (no hands detected)
   from model classification errors. Failed tracking counts as an end-to-end system failure in overall accuracy.
6. Idle Monitoring: Tests and reports false-positive prediction rates during idle/resting movement.
   Reports "Not Evaluated" when no idle recordings exist.
7. Absolute Honesty: Displays PENDING until actual volunteer recordings exist; never fabricates results.
8. Privacy & Consent: Anonymous volunteer IDs by default, local data storage only, git-ignored video files,
   and instant volunteer data deletion.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.infrastructure.recognition.karsl_recognizer import LandmarkerProcessor
from src.research.cnn_pipeline import ResearchCNNPipeline
from src.research.karsl_dataset import SELECTED_RESEARCH_CLASSES
from src.research.preprocessing import ResearchFeatureExtractor
from src.research.svm_pipeline import ResearchSVMPipeline

logger = logging.getLogger("arsl.research.independent_eval")

# Repository-root-aware paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = REPO_ROOT / "data" / "research" / "independent_evaluation"
RECORDINGS_DIR = BASE_DIR / "recordings"
MANIFEST_FILE = BASE_DIR / "manifest.json"
RESULTS_FILE = BASE_DIR / "evaluation_results.json"
VOLUNTEERS_FILE = BASE_DIR / "volunteers.json"

TARGET_CLASSES = SELECTED_RESEARCH_CLASSES  # 10 classes: 0001 to 0010
TARGET_REPETITIONS = 5

SIGN_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "0001": {
        "name_ar": "صفر",
        "name_en": "Zero",
        "tip_ar": "إشارة الصفر: ضم أصابع اليد وتشكيل قبضة مغلقة أو دائرة بالإبهام والسبابة.",
        "tip_en": "Zero gesture: Closed fist or circular pinch using thumb and index finger.",
    },
    "0002": {
        "name_ar": "واحد",
        "name_en": "One",
        "tip_ar": "إشارة الواحد: رفع إصبع السبابة عمودياً لأعلى مع ثني باقي الأصابع بإحكام.",
        "tip_en": "One gesture: Index finger extended vertically upright with other fingers folded.",
    },
    "0003": {
        "name_ar": "اثنان",
        "name_en": "Two",
        "tip_ar": "إشارة الاثنين: رفع إصبعي السبابة والوسطى مشكلين حرف V مع ثني الباقي.",
        "tip_en": "Two gesture: Index and middle fingers extended upright forming a V-shape.",
    },
    "0004": {
        "name_ar": "ثلاثة",
        "name_en": "Three",
        "tip_ar": "إشارة الثلاثة: رفع الإبهام والسبابة والوسطى مع ثني البنصر والخنصر.",
        "tip_en": "Three gesture: Thumb, index, and middle fingers extended with ring and pinky folded.",
    },
    "0005": {
        "name_ar": "أربعة",
        "name_en": "Four",
        "tip_ar": "إشارة الأربعة: رفع الأصابع الأربعة مفرودة للأعلى مع ثني الإبهام على راحة اليد.",
        "tip_en": "Four gesture: Four fingers (index, middle, ring, pinky) extended with thumb folded.",
    },
    "0006": {
        "name_ar": "خمسة",
        "name_en": "Five",
        "tip_ar": "إشارة الخمسة: فتح كف اليد بالكامل مع تباعد الأصابع الخمسة المفرودة للأعلى.",
        "tip_en": "Five gesture: Open palm facing forward with all five fingers spread upright.",
    },
    "0007": {
        "name_ar": "ستة",
        "name_en": "Six",
        "tip_ar": "إشارة الستة: ثني السبابة والوسطى والبنصر مع فرد الإبهام والخنصر أو خفض السبابة أفقياً.",
        "tip_en": "Six gesture: Thumb and pinky extended or index pointed downward.",
    },
    "0008": {
        "name_ar": "سبعة",
        "name_en": "Seven",
        "tip_ar": "إشارة السبعة: فتح السبابة والإبهام متوجهين لأعلى بشكل زاوية تشبه رقم 7 بالعربية.",
        "tip_en": "Seven gesture: Index finger and thumb pointing upward forming a sharp 7 angle.",
    },
    "0009": {
        "name_ar": "ثمانية",
        "name_en": "Eight",
        "tip_ar": "إشارة الثمانية: توجيه السبابة والوسطى للأسفل باتجاه الأرض لتشكيل رقم 8 بالعربية.",
        "tip_en": "Eight gesture: Index and middle fingers pointing downward forming an inverted V (Arabic 8).",
    },
    "0010": {
        "name_ar": "تسعة",
        "name_en": "Nine",
        "tip_ar": "إشارة التسعة: تشكيل دائرة بإغلاق طرف السبابة مع الإبهام وفرد باقي الأصابع الثلاثة للأعلى.",
        "tip_en": "Nine gesture: Index finger and thumb forming an O circle with 3 fingers upright.",
    },
    "idle": {
        "name_ar": "استراحة / خمول",
        "name_en": "Idle / Resting Baseline",
        "tip_ar": "حالة الاستراحة: الجلوس باسترخاء أمام الكاميرا دون أداء أي إشارة لقياس معدل عدم التنبؤ الكاذب.",
        "tip_en": "Idle Baseline: Sit relaxed facing camera without signing, to verify false-positive rejection.",
    },
}

CONSENT_NOTICE = {
    "title_ar": "إشعار الخصوصية والموافقة المستنيرة",
    "title_en": "Informed Consent & Privacy Notice",
    "body_ar": (
        "تُجمع تسجيلات الفيديو هذه حصرياً لأغراض التقييم العلمي لمشروع التخرج في قسم هندسة الحاسوب - "
        "جامعة البصرة تحت إشراف الدكتورة هبة حاكم عبد الزهرة. يتم تخزين جميع الفيديوهات محلياً على هذا الجهاز فقط "
        "ولا تُرفع أو تُنشر إلى أي مستودع عام (GitHub). يحق للمشارك طلب حذف تسجيلاته ومعلوماته في أي وقت بنقرة واحدة."
    ),
    "body_en": (
        "These video recordings are captured strictly for academic research evaluation at the University of Basrah, "
        "Department of Computer Engineering, supervised by Dr. Hiba Hakim Abdulzahra. All recordings remain local to this machine "
        "and are NEVER committed or published to any public repository. Participants may permanently delete their data at any time."
    ),
    "privacy_principles": [
        "Local-Only Storage: Recordings are saved strictly on this local disk and never pushed to GitHub.",
        "Zero Retraining: Evaluation recordings are never used to retrain or adapt models.",
        "One-Click Right to Erasure: Volunteers can delete all their data and video recordings anytime.",
        "Anonymized Identifiers: Real names are never committed or exposed in benchmarks.",
    ],
}


class IndependentEvaluationManager:
    """Manages volunteer recording, storage isolation, review, and frozen dual-model evaluation."""

    def __init__(self, base_dir: Path | str = BASE_DIR) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.recordings_dir = self.base_dir / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.base_dir / "manifest.json"
        self.results_file = self.base_dir / "evaluation_results.json"
        self.volunteers_file = self.base_dir / "volunteers.json"

        self.classes = TARGET_CLASSES
        self.target_reps = TARGET_REPETITIONS

        # Frozen deployment checkpoints (strictly read-only)
        self.svm_pipeline = ResearchSVMPipeline(output_dir="data/models/research/svm", feature_schema="hands_only")
        self.cnn_pipeline = ResearchCNNPipeline(output_dir="data/models/research/cnn", feature_schema="hands_only")
        self.extractor = ResearchFeatureExtractor(feature_schema="hands_only")
        self._processor: Optional[LandmarkerProcessor] = None

    def _get_processor(self) -> LandmarkerProcessor:
        if self._processor is None:
            self._processor = LandmarkerProcessor("data/models/karsl")
        return self._processor

    # ─── Path Resolution & Traversal Security ──────────────────────────────

    def resolve_recording_path(self, raw_path: str | Path, check_exists: bool = False) -> Optional[Path]:
        """Safely resolve an evaluation recording path against the repository and base directory.

        Protects against path traversal attacks and resolves relative paths cleanly.
        Raises ValueError if path traversal is detected.
        """
        if not raw_path:
            return None

        p_str = str(raw_path).replace("\\", "/").strip()
        rel_p = Path(p_str)

        base_res = self.base_dir.resolve()
        repo_res = REPO_ROOT.resolve()

        # Check traversal on relative path
        test_candidate = (self.base_dir / rel_p).resolve()
        is_safe = False
        try:
            if test_candidate.is_relative_to(base_res) or test_candidate.is_relative_to(repo_res):
                is_safe = True
        except (ValueError, AttributeError):
            pass

        if not is_safe:
            raise ValueError(f"Path traversal detected: {raw_path}")

        candidates = [
            test_candidate,
            (REPO_ROOT / rel_p).resolve(),
            (self.recordings_dir / rel_p).resolve(),
        ]

        if "recordings" in rel_p.parts:
            idx = rel_p.parts.index("recordings")
            sub_parts = Path(*rel_p.parts[idx:])
            candidates.append((self.base_dir / sub_parts).resolve())

        for c in candidates:
            try:
                if (c.is_relative_to(base_res) or c.is_relative_to(repo_res)) and c.exists() and c.is_file():
                    return c
            except (ValueError, RuntimeError):
                continue

        if check_exists:
            logger.debug(f"Could not find existing recording file on disk: {raw_path}")
            return None

        return test_candidate

    # ─── Reference Signs & Consent ─────────────────────────────────────────

    def get_consent_notice(self) -> dict[str, str]:
        """Return the bilingual research consent notice."""
        return CONSENT_NOTICE

    def list_sign_references(self) -> list[dict[str, Any]]:
        """List the 10 Arabic digit reference signs with descriptions and pose instructions."""
        result = []
        for c in self.classes:
            s_id = c["sign_id"]
            info = SIGN_INSTRUCTIONS.get(s_id, {})
            result.append({
                "sign_id": s_id,
                "class_idx": c["index"],
                "name_ar": c["name_ar"],
                "name_en": c["name_en"],
                "tip_ar": info.get("tip_ar", ""),
                "tip_en": info.get("tip_en", ""),
                "target_reps": self.target_reps,
            })
        return result

    def get_sign_reference_info(self, sign_id: str) -> Optional[dict[str, Any]]:
        """Get instructions for a specific sign ID or 'idle'."""
        if sign_id == "idle":
            info = SIGN_INSTRUCTIONS["idle"]
            return {
                "sign_id": "idle",
                "class_idx": -1,
                "name_ar": info["name_ar"],
                "name_en": info["name_en"],
                "tip_ar": info["tip_ar"],
                "tip_en": info["tip_en"],
                "target_reps": 5,
            }
        matched = next((c for c in self.classes if c["sign_id"] == sign_id), None)
        if not matched:
            return None
        info = SIGN_INSTRUCTIONS.get(sign_id, {})
        return {
            "sign_id": sign_id,
            "class_idx": matched["index"],
            "name_ar": matched["name_ar"],
            "name_en": matched["name_en"],
            "tip_ar": info.get("tip_ar", ""),
            "tip_en": info.get("tip_en", ""),
            "target_reps": self.target_reps,
        }

    # ─── Volunteer Management (Privacy Protected) ──────────────────────────

    def list_volunteers(self) -> list[dict[str, Any]]:
        """List registered volunteer participants."""
        if not self.volunteers_file.exists():
            return []
        try:
            with open(self.volunteers_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def register_volunteer(self, name: str = "", notes: str = "") -> dict[str, Any]:
        """Register a new human volunteer for independent testing with anonymous default ID."""
        volunteers = self.list_volunteers()
        vol_num = len(volunteers) + 1
        volunteer_id = f"vol_{vol_num:03d}_{int(time.time()) % 10000:04d}"
        display_name = name.strip() or f"Volunteer {vol_num:02d}"

        entry = {
            "volunteer_id": volunteer_id,
            "name": display_name,
            "notes": notes.strip(),
            "registered_at": time.time(),
            "completed_signs": 0,
            "total_recordings": 0,
        }
        volunteers.append(entry)
        with open(self.volunteers_file, "w", encoding="utf-8") as f:
            json.dump(volunteers, f, ensure_ascii=False, indent=2)

        # Invalidate stale results when volunteer roster changes
        self.invalidate_results()
        logger.info(f"Registered volunteer: {entry['volunteer_id']} ({entry['name']})")
        return entry

    def delete_volunteer(self, volunteer_id: str) -> dict[str, Any]:
        """Permanently delete a volunteer and all their local video recordings for privacy."""
        volunteers = self.list_volunteers()
        volunteers = [v for v in volunteers if v["volunteer_id"] != volunteer_id]
        with open(self.volunteers_file, "w", encoding="utf-8") as f:
            json.dump(volunteers, f, ensure_ascii=False, indent=2)

        manifest = self.load_manifest()
        remaining_manifest = []
        deleted_count = 0

        for r in manifest:
            if r["volunteer_id"] == volunteer_id:
                deleted_count += 1
                v_path = self.resolve_recording_path(r.get("file_path", ""))
                if v_path and v_path.exists():
                    try:
                        v_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete video file {v_path}: {e}")
            else:
                remaining_manifest.append(r)

        self.save_manifest(remaining_manifest)

        # Remove volunteer directory
        vol_dir = self.recordings_dir / volunteer_id
        if vol_dir.exists():
            try:
                shutil.rmtree(vol_dir)
            except Exception as e:
                logger.warning(f"Failed to remove directory {vol_dir}: {e}")

        self.invalidate_results()
        logger.info(f"Deleted volunteer {volunteer_id} and {deleted_count} associated recordings.")
        return {
            "success": True,
            "volunteer_id": volunteer_id,
            "deleted_recordings_count": deleted_count,
        }

    # ─── Manifest & Repetition Control ─────────────────────────────────────

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

    def get_next_repetition_index(self, volunteer_id: str, sign_id: str) -> int:
        """Find the next available repetition index (1-5) to prevent duplicate repetition IDs."""
        manifest = self.load_manifest()
        existing_reps = {
            r["rep_idx"]
            for r in manifest
            if r["volunteer_id"] == volunteer_id
            and r["sign_id"] == sign_id
            and r.get("is_accepted", True)
            and not r.get("is_idle", False)
        }
        for rep in range(1, self.target_reps + 1):
            if rep not in existing_reps:
                return rep
        return len(existing_reps) + 1

    def get_volunteer_progress(self, volunteer_id: str) -> dict[str, Any]:
        """Return recording progress for a given volunteer (10 classes x 5 repetitions)."""
        manifest = self.load_manifest()
        v_records = [r for r in manifest if r["volunteer_id"] == volunteer_id and r.get("is_accepted", True)]

        progress_by_sign = {}
        for c in self.classes:
            sign_id = c["sign_id"]
            sign_recs = [r for r in v_records if r["sign_id"] == sign_id and not r.get("is_idle", False)]
            reps_done = len(sign_recs)
            progress_by_sign[sign_id] = {
                "name_ar": c["name_ar"],
                "name_en": c["name_en"],
                "reps_completed": reps_done,
                "target_reps": self.target_reps,
                "is_complete": reps_done >= self.target_reps,
                "next_rep_idx": self.get_next_repetition_index(volunteer_id, sign_id),
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

    def invalidate_results(self) -> None:
        """Invalidate stale evaluation results when recordings or volunteers change."""
        if self.results_file.exists():
            try:
                self.results_file.unlink()
            except Exception:
                pass

    # ─── Recording & Video Storage ─────────────────────────────────────────

    def save_recording(
        self,
        volunteer_id: str,
        sign_id: str,
        rep_idx: int,
        frames: list[np.ndarray],
        fps: float = 30.0,
        is_idle: bool = False,
    ) -> dict[str, Any]:
        """Save a reviewed and accepted video recording as MP4 with repository-root-aware paths."""
        if len(frames) == 0:
            raise ValueError("Cannot save empty recording.")

        h, w = frames[0].shape[:2]
        dest_dir = self.recordings_dir / volunteer_id / sign_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        tag = "idle" if is_idle else f"rep_{rep_idx:02d}"
        filename = f"{volunteer_id}_{sign_id}_{tag}_{int(time.time())}.mp4"
        file_path = dest_dir / filename

        # Write MP4 using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(file_path), fourcc, fps, (w, h))
        for f in frames:
            bgr = f if len(f.shape) == 3 and f.shape[2] == 3 else cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            out.write(bgr)
        out.release()

        # Class metadata
        matched_class = next((c for c in self.classes if c["sign_id"] == sign_id), None)
        class_idx = matched_class["index"] if matched_class else -1
        name_ar = matched_class["name_ar"] if matched_class else ("استراحة" if is_idle else "غير معروف")

        # Normalized path relative to REPO_ROOT (forward slashes)
        try:
            rel_path = file_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except (ValueError, Exception):
            try:
                rel_path = file_path.resolve().relative_to(self.base_dir.resolve()).as_posix()
            except (ValueError, Exception):
                rel_path = file_path.name
        record_id = f"ind_{volunteer_id}_{sign_id}_{tag}_{int(time.time())}"

        record_entry = {
            "record_id": record_id,
            "volunteer_id": volunteer_id,
            "sign_id": sign_id,
            "class_idx": class_idx,
            "name_ar": name_ar,
            "rep_idx": rep_idx,
            "file_path": rel_path,
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
        # Prevent duplicate repetition records
        if not is_idle:
            manifest = [
                r for r in manifest
                if not (r["volunteer_id"] == volunteer_id and r["sign_id"] == sign_id and r.get("rep_idx") == rep_idx)
            ]
        manifest.append(record_entry)
        self.save_manifest(manifest)

        self._update_volunteer_stats(volunteer_id)
        self.invalidate_results()

        logger.info(f"Saved evaluation recording: {rel_path} ({len(frames)} frames)")
        return record_entry

    def save_recording_from_bytes(
        self,
        volunteer_id: str,
        sign_id: str,
        rep_idx: int,
        video_bytes: bytes,
        filename: str = "recording.mp4",
        fps: float = 30.0,
        is_idle: bool = False,
    ) -> dict[str, Any]:
        """Save a recording uploaded as raw video bytes (e.g. from React MediaRecorder)."""
        if len(video_bytes) == 0:
            raise ValueError("Cannot save empty video bytes.")

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            cap = cv2.VideoCapture(str(tmp_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
            cap.release()

            if len(frames) > 0:
                return self.save_recording(
                    volunteer_id=volunteer_id,
                    sign_id=sign_id,
                    rep_idx=rep_idx,
                    frames=frames,
                    fps=fps,
                    is_idle=is_idle,
                )

            # Fallback: cv2 could not decode frames directly, save raw bytes directly to disk
            vol_dir = self.recordings_dir / volunteer_id / sign_id
            vol_dir.mkdir(parents=True, exist_ok=True)
            tag = f"rep_{rep_idx:02d}"
            ext = Path(filename).suffix.lower() if filename and Path(filename).suffix else ".webm"
            final_filename = f"{volunteer_id}_{sign_id}_{tag}_{int(time.time())}{ext}"
            target_file = vol_dir / final_filename
            with open(target_file, "wb") as f_out:
                f_out.write(video_bytes)

            try:
                rel_path = target_file.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            except (ValueError, Exception):
                try:
                    rel_path = target_file.resolve().relative_to(self.base_dir.resolve()).as_posix()
                except (ValueError, Exception):
                    rel_path = target_file.name

            matched_class = next((c for c in self.classes if c["sign_id"] == sign_id), None)
            class_idx = matched_class["index"] if matched_class else -1
            name_ar = matched_class["name_ar"] if matched_class else ("استراحة" if is_idle else "غير معروف")
            record_id = f"ind_{volunteer_id}_{sign_id}_{tag}_{int(time.time())}"

            record_entry = {
                "record_id": record_id,
                "volunteer_id": volunteer_id,
                "sign_id": sign_id,
                "class_idx": class_idx,
                "name_ar": name_ar,
                "rep_idx": rep_idx,
                "file_path": rel_path,
                "frame_count": 0,
                "width": 640,
                "height": 480,
                "fps": fps,
                "duration_sec": 0.0,
                "is_idle": is_idle,
                "is_accepted": True,
                "recorded_at": time.time(),
            }
            manifest = self.load_manifest()
            if not is_idle:
                manifest = [
                    r for r in manifest
                    if not (r["volunteer_id"] == volunteer_id and r["sign_id"] == sign_id and r.get("rep_idx") == rep_idx)
                ]
            manifest.append(record_entry)
            self.save_manifest(manifest)
            self._update_volunteer_stats(volunteer_id)
            self.invalidate_results()
            return record_entry
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def delete_recording(self, record_id: str) -> dict[str, Any]:
        """Delete an individual recording from manifest and disk."""
        manifest = self.load_manifest()
        target = next((r for r in manifest if r["record_id"] == record_id), None)
        if not target:
            return {"success": False, "error": f"Record {record_id} not found."}

        v_path = self.resolve_recording_path(target.get("file_path", ""))
        if v_path and v_path.exists():
            try:
                v_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete recording file {v_path}: {e}")

        manifest = [r for r in manifest if r["record_id"] != record_id]
        self.save_manifest(manifest)
        self._update_volunteer_stats(target["volunteer_id"])
        self.invalidate_results()

        return {"success": True, "record_id": record_id}

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

    # ─── Dual-Model Frozen Evaluation Engine ───────────────────────────────

    def evaluate_independent_dataset(self) -> dict[str, Any]:
        """Execute genuine evaluation of frozen SVM & CNN on independent volunteer recordings.

        Never retrains models. Disentangles tracking failures from classification errors.
        A recording with failed tracking counts as an end-to-end failure in overall accuracy.
        """
        manifest = self.load_manifest()
        accepted_records = [r for r in manifest if r.get("is_accepted", True)]
        sign_records = [r for r in accepted_records if not r.get("is_idle", False)]
        idle_records = [r for r in accepted_records if r.get("is_idle", False)]

        total_accepted = len(sign_records)
        if total_accepted == 0:
            pending_report = {
                "status": "PENDING_VOLUNTEER_RECORDINGS",
                "is_evaluated": False,
                "message": (
                    "No independent volunteer recordings found. In accordance with university research "
                    "integrity standards, performance on unseen individuals remains strictly unmeasured "
                    "until genuine volunteers complete the test protocol."
                ),
                "volunteers_count": len(self.list_volunteers()),
                "total_recordings": 0,
                "total_accepted_recordings": 0,
                "idle_recordings": len(idle_records),
                "decoded_videos": 0,
                "missing_or_corrupt_recordings": 0,
                "evaluation_date": time.time(),
                "idle_false_positives": {
                    "idle_videos_tested": 0,
                    "svm_false_positives": 0,
                    "cnn_false_positives": 0,
                    "svm_false_positive_rate": 0.0,
                    "cnn_false_positive_rate": 0.0,
                    "status": "NOT_EVALUATED",
                    "display": "Not Evaluated",
                },
            }
            with open(self.results_file, "w", encoding="utf-8") as f:
                json.dump(pending_report, f, ensure_ascii=False, indent=2)
            return pending_report

        logger.info(f"Starting Independent Evaluation on {total_accepted} volunteer recordings...")
        processor = self._get_processor()

        ground_truth: list[int] = []
        svm_preds_tracked: list[int] = []
        cnn_preds_tracked: list[int] = []
        gt_tracked: list[int] = []

        tracking_statuses: list[bool] = []
        sample_details: list[dict[str, Any]] = []

        svm_times: list[float] = []
        cnn_times: list[float] = []

        decoded_count = 0
        missing_count = 0

        # Per-class counters
        per_class_stats: dict[int, dict[str, Any]] = {
            c["index"]: {
                "sign_id": c["sign_id"],
                "name_ar": c["name_ar"],
                "name_en": c["name_en"],
                "total_samples": 0,
                "tracked_samples": 0,
                "svm_correct": 0,
                "cnn_correct": 0,
            }
            for c in self.classes
        }

        for r in sign_records:
            v_path = self.resolve_recording_path(r.get("file_path", ""))
            if not v_path or not v_path.exists():
                missing_count += 1
                logger.warning(f"Video file missing on disk: {r.get('file_path')}")
                sample_details.append({
                    "record_id": r["record_id"],
                    "volunteer_id": r["volunteer_id"],
                    "sign_id": r["sign_id"],
                    "ground_truth_class": r["class_idx"],
                    "ground_truth_ar": r["name_ar"],
                    "tracking_success": False,
                    "status": "FILE_MISSING",
                    "svm_correct": False,
                    "cnn_correct": False,
                })
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
                missing_count += 1
                logger.warning(f"Could not decode frames from video: {v_path}")
                continue

            decoded_count += 1
            gt_idx = r["class_idx"]
            ground_truth.append(gt_idx)

            if gt_idx in per_class_stats:
                per_class_stats[gt_idx]["total_samples"] += 1

            # Stage 1: Hand Tracking Disentanglement
            # Hand is tracked if present in >= 5 frames or >= 20% of video
            hand_tracked = (valid_hand_frames >= 5) or (valid_hand_frames / max(len(raw_kps), 1) >= 0.20)
            tracking_statuses.append(hand_tracked)

            if hand_tracked:
                if gt_idx in per_class_stats:
                    per_class_stats[gt_idx]["tracked_samples"] += 1

                # Resample sequence to 30 frames
                seq_arr = np.array(raw_kps, dtype=np.float32)
                seq_30 = self.extractor.resample_sequence(list(seq_arr), target_len=30)

                # Predict SVM
                t0 = time.perf_counter()
                s_class, s_conf, s_ar, _ = self.svm_pipeline.predict_sequence(seq_30)
                svm_times.append((time.perf_counter() - t0) * 1000.0)

                # Predict CNN
                t0 = time.perf_counter()
                c_class, c_conf, c_ar, _ = self.cnn_pipeline.predict_sequence(seq_30)
                cnn_times.append((time.perf_counter() - t0) * 1000.0)

                s_correct = (s_class == gt_idx)
                c_correct = (c_class == gt_idx)

                if s_correct and gt_idx in per_class_stats:
                    per_class_stats[gt_idx]["svm_correct"] += 1
                if c_correct and gt_idx in per_class_stats:
                    per_class_stats[gt_idx]["cnn_correct"] += 1

                gt_tracked.append(gt_idx)
                svm_preds_tracked.append(s_class)
                cnn_preds_tracked.append(c_class)

                sample_details.append({
                    "record_id": r["record_id"],
                    "volunteer_id": r["volunteer_id"],
                    "sign_id": r["sign_id"],
                    "ground_truth_class": gt_idx,
                    "ground_truth_ar": r["name_ar"],
                    "tracking_success": True,
                    "hand_frames_ratio": round(valid_hand_frames / max(len(raw_kps), 1), 2),
                    "svm_pred_class": s_class,
                    "svm_pred_ar": s_ar,
                    "svm_conf": round(s_conf, 3),
                    "svm_correct": s_correct,
                    "cnn_pred_class": c_class,
                    "cnn_pred_ar": c_ar,
                    "cnn_conf": round(c_conf, 3),
                    "cnn_correct": c_correct,
                })
            else:
                # Stage 2: Failed Tracking = End-to-End System Failure
                sample_details.append({
                    "record_id": r["record_id"],
                    "volunteer_id": r["volunteer_id"],
                    "sign_id": r["sign_id"],
                    "ground_truth_class": gt_idx,
                    "ground_truth_ar": r["name_ar"],
                    "tracking_success": False,
                    "hand_frames_ratio": round(valid_hand_frames / max(len(raw_kps), 1), 2),
                    "svm_pred_class": -1,
                    "svm_pred_ar": "فشل التتبع",
                    "svm_conf": 0.0,
                    "svm_correct": False,
                    "cnn_pred_class": -1,
                    "cnn_pred_ar": "فشل التتبع",
                    "cnn_conf": 0.0,
                    "cnn_correct": False,
                })

        # Stage 3: Idle Baseline Testing
        idle_results = self._evaluate_idle_recordings(idle_records, processor)

        # Stage 4: Genuine Scientific Metrics with Correct Denominators
        tracked_count = sum(1 for t in tracking_statuses if t)
        tracking_rate = round(tracked_count / max(decoded_count, 1), 4)

        # Conditioned Accuracy & Macro F1 (Evaluated strictly on tracked recordings)
        if len(gt_tracked) > 0:
            cond_svm_acc = round(float(accuracy_score(gt_tracked, svm_preds_tracked)), 4)
            cond_cnn_acc = round(float(accuracy_score(gt_tracked, cnn_preds_tracked)), 4)
            cond_svm_f1 = round(float(f1_score(gt_tracked, svm_preds_tracked, average="macro", zero_division=0)), 4)
            cond_cnn_f1 = round(float(f1_score(gt_tracked, cnn_preds_tracked, average="macro", zero_division=0)), 4)
        else:
            cond_svm_acc = cond_cnn_acc = cond_svm_f1 = cond_cnn_f1 = 0.0

        # Raw Overall Accuracy across all decoded recordings (failed tracking counted as incorrect)
        raw_svm_correct = sum(1 for d in sample_details if d.get("tracking_success") and d.get("svm_correct"))
        raw_cnn_correct = sum(1 for d in sample_details if d.get("tracking_success") and d.get("cnn_correct"))
        raw_svm_acc = round(raw_svm_correct / max(decoded_count, 1), 4)
        raw_cnn_acc = round(raw_cnn_correct / max(decoded_count, 1), 4)

        # Confusion Matrices on tracked recordings
        num_classes = len(self.classes)
        class_names_ar = [c["name_ar"] for c in self.classes]
        if len(gt_tracked) > 0:
            svm_cm = confusion_matrix(gt_tracked, svm_preds_tracked, labels=list(range(num_classes))).tolist()
            cnn_cm = confusion_matrix(gt_tracked, cnn_preds_tracked, labels=list(range(num_classes))).tolist()
        else:
            svm_cm = [[0] * num_classes for _ in range(num_classes)]
            cnn_cm = [[0] * num_classes for _ in range(num_classes)]

        # Per-class metrics
        per_class_results = []
        for c_idx, s in sorted(per_class_stats.items()):
            tot = s["total_samples"]
            per_class_results.append({
                "class_idx": c_idx,
                "sign_id": s["sign_id"],
                "name_ar": s["name_ar"],
                "name_en": s["name_en"],
                "total_samples": tot,
                "tracked_samples": s["tracked_samples"],
                "svm_correct": s["svm_correct"],
                "cnn_correct": s["cnn_correct"],
                "svm_accuracy": round(s["svm_correct"] / max(tot, 1), 4) if tot > 0 else 0.0,
                "cnn_accuracy": round(s["cnn_correct"] / max(tot, 1), 4) if tot > 0 else 0.0,
            })

        volunteers_evaluated = list(set(r["volunteer_id"] for r in sign_records))

        report: dict[str, Any] = {
            "status": "EVALUATED_ON_AUTHENTIC_VOLUNTEERS",
            "is_evaluated": True,
            "evaluation_date": time.time(),
            "volunteers_count": len(volunteers_evaluated),
            "volunteers": volunteers_evaluated,
            "total_accepted_recordings": total_accepted,
            "decoded_videos": decoded_count,
            "missing_or_corrupt_recordings": missing_count,
            "total_tested_recordings": decoded_count,

            # Primary disentangled metrics (Top-Level for easy frontend binding)
            "tracking_success_rate": tracking_rate,
            "tracking_failure_count": decoded_count - tracked_count,
            "raw_overall_accuracy": {
                "svm": raw_svm_acc,
                "cnn": raw_cnn_acc,
            },
            "conditioned_accuracy": {
                "svm": cond_svm_acc,
                "cnn": cond_cnn_acc,
            },
            "conditioned_macro_f1": {
                "svm": cond_svm_f1,
                "cnn": cond_cnn_f1,
            },
            "latency_ms": {
                "svm": round(float(np.mean(svm_times)), 2) if svm_times else 0.0,
                "cnn": round(float(np.mean(cnn_times)), 2) if cnn_times else 0.0,
            },
            "idle_evaluation": idle_results,
            "confusion_matrices": {
                "classes": class_names_ar,
                "svm": svm_cm,
                "cnn": cnn_cm,
            },
            "per_class_results": per_class_results,
            "sample_details": sample_details,
            "mistakes_count": sum(1 for d in sample_details if not (d.get("tracking_success") and d.get("svm_correct") and d.get("cnn_correct"))),
        }

        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Independent Evaluation completed: Decoded={decoded_count}, Tracking={tracking_rate*100:.1f}%, "
            f"Cond_SVM={cond_svm_acc*100:.1f}%, Cond_CNN={cond_cnn_acc*100:.1f}%, Raw_CNN={raw_cnn_acc*100:.1f}%"
        )
        return report

    def _evaluate_idle_recordings(
        self,
        idle_records: list[dict[str, Any]],
        processor: LandmarkerProcessor,
    ) -> dict[str, Any]:
        """Test whether idle/resting movements falsely trigger sign classifications."""
        if len(idle_records) == 0:
            return {
                "status": "NOT_EVALUATED",
                "display": "Not Evaluated",
                "idle_recordings_tested": 0,
                "false_positive_count_svm": 0,
                "false_positive_count_cnn": 0,
                "svm_false_positive_rate": None,
                "cnn_false_positive_rate": None,
                "note": "No resting baseline clips recorded yet. Recommended to capture 5 idle clips.",
            }

        fp_svm = 0
        fp_cnn = 0
        tested_count = 0

        for r in idle_records:
            v_path = self.resolve_recording_path(r.get("file_path", ""))
            if not v_path or not v_path.exists():
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

            tested_count += 1
            seq_30 = self.extractor.resample_sequence(list(np.array(raw_kps, dtype=np.float32)), target_len=30)
            _, s_conf, _, _ = self.svm_pipeline.predict_sequence(seq_30)
            _, c_conf, _, _ = self.cnn_pipeline.predict_sequence(seq_30)

            if s_conf >= 0.60:
                fp_svm += 1
            if c_conf >= 0.60:
                fp_cnn += 1

        if tested_count == 0:
            return {
                "status": "NOT_EVALUATED",
                "display": "Not Evaluated",
                "idle_recordings_tested": 0,
                "false_positive_count_svm": 0,
                "false_positive_count_cnn": 0,
                "svm_false_positive_rate": None,
                "cnn_false_positive_rate": None,
                "note": "No valid idle clips could be decoded.",
            }

        return {
            "status": "EVALUATED",
            "display": f"{round(fp_cnn / tested_count * 100, 1)}% (CNN)",
            "idle_recordings_tested": tested_count,
            "false_positive_count_svm": fp_svm,
            "false_positive_count_cnn": fp_cnn,
            "svm_false_positive_rate": round(fp_svm / tested_count, 3),
            "cnn_false_positive_rate": round(fp_cnn / tested_count, 3),
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
            "idle_evaluation": {
                "status": "NOT_EVALUATED",
                "display": "Not Evaluated",
                "idle_recordings_tested": 0,
                "false_positive_count_svm": 0,
                "false_positive_count_cnn": 0,
                "svm_false_positive_rate": None,
                "cnn_false_positive_rate": None,
            },
        }
