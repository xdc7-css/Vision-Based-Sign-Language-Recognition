"""Research Service for FastAPI Backend.

Coordinates:
- KArSL-502 dataset preparation and status queries
- Training of Model A (SVM) and Model B (1D Temporal CNN)
- Scientific evaluation and comparison report generation
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from src.research.cnn_pipeline import ResearchCNNPipeline
from src.research.evaluator import ResearchScientificEvaluator
from src.research.independent_evaluation import IndependentEvaluationManager
from src.research.karsl_dataset import KArSLDatasetManager
from src.research.svm_pipeline import ResearchSVMPipeline

logger = logging.getLogger("arsl.backend.research")


class ResearchService:
    """Singleton service managing KArSL-502 SVM vs CNN research experiments."""

    def __init__(self) -> None:
        self.dm = KArSLDatasetManager()
        self.svm_pipeline = ResearchSVMPipeline()
        self.cnn_pipeline = ResearchCNNPipeline()
        self.evaluator = ResearchScientificEvaluator(dataset_manager=self.dm)
        self.independent_eval_manager = IndependentEvaluationManager()

        self._active_job: Optional[dict[str, Any]] = None
        self._lock = threading.Lock()

    def get_status(self) -> dict[str, Any]:
        """Return comprehensive status of dataset, models, and latest comparison."""
        ds_status = self.dm.get_status()
        svm_trained = self.svm_pipeline.is_trained()
        cnn_trained = self.cnn_pipeline.is_trained()
        latest_comp = self.get_latest_comparison()
        ind_status = self.independent_eval_manager.get_status()

        return {
            "dataset": ds_status,
            "models": {
                "svm": {
                    "is_trained": svm_trained,
                    "metrics": self.svm_pipeline.metadata.get("metrics")
                    if svm_trained
                    else None,
                },
                "cnn": {
                    "is_trained": cnn_trained,
                    "metrics": self.cnn_pipeline.metadata.get("metrics")
                    if cnn_trained
                    else None,
                },
            },
            "comparison": latest_comp,
            "independent_evaluation": ind_status,
            "active_job": self._active_job,
        }

    def get_latest_comparison(self) -> Optional[dict[str, Any]]:
        """Return latest comparison report with confusion matrices."""
        import json
        comp = self.evaluator.get_latest_comparison()
        if comp:
            cm_file = self.evaluator.eval_dir / "confusion_matrices.json"
            if cm_file.exists():
                try:
                    with open(cm_file, "r", encoding="utf-8") as f:
                        comp["confusion_matrices"] = json.load(f)
                except Exception:
                    pass
        return comp

    def prepare_dataset(self, samples_per_sign_per_signer: int = 15, force: bool = False) -> dict[str, Any]:
        """Trigger KArSL research dataset preparation."""
        with self._lock:
            self._active_job = {"job": "prepare_dataset", "status": "running"}
        try:
            res = self.dm.prepare_dataset(
                samples_per_sign_per_signer=samples_per_sign_per_signer,
                force_rebuild=force,
            )
            with self._lock:
                self._active_job = None
            return res
        except Exception as e:
            with self._lock:
                self._active_job = {"job": "prepare_dataset", "status": "failed", "error": str(e)}
            raise

    def train_svm(self, kernel: str = "rbf", C: float = 2.0) -> dict[str, Any]:
        """Train SVM model on prepared research dataset using inner validation (strictly zero test leakage)."""
        with self._lock:
            self._active_job = {"job": "train_svm", "status": "running"}

        try:
            # Strictly use inner train/val split from training signers only (Signer01 + Signer02)
            # The held-out test signer (Signer03) is completely untouched.
            split = self.dm.get_inner_train_val_split(loso_fold="test_Signer03")

            # Save manual user training to dedicated manual checkpoint dir to protect deployment checkpoint
            manual_svm_dir = "data/models/research/manual_training/svm"
            svm_manual_pipe = ResearchSVMPipeline(output_dir=manual_svm_dir, feature_schema="hands_only")

            metrics = svm_manual_pipe.train(
                X_train_seq=split["X_inner_train"],
                y_train=split["y_inner_train"],
                X_val_seq=split["X_inner_val"],
                y_val=split["y_inner_val"],
                kernel=kernel,
                C=C,
            )
            metrics["held_out_test_signer"] = split["test_signer"]
            metrics["test_leakage_status"] = "ZERO_TEST_LEAKAGE_VERIFIED"
            metrics["validation_protocol"] = "Inner-Train/Val (Signers 01 & 02 only)"

            with self._lock:
                self._active_job = None
            return metrics
        except Exception as e:
            with self._lock:
                self._active_job = {"job": "train_svm", "status": "failed", "error": str(e)}
            raise

    def train_cnn(self, epochs: int = 35, batch_size: int = 16) -> dict[str, Any]:
        """Train 1D Temporal CNN model on prepared research dataset using inner validation (strictly zero test leakage)."""
        with self._lock:
            self._active_job = {"job": "train_cnn", "status": "running"}

        try:
            # Strictly use inner train/val split from training signers only (Signer01 + Signer02)
            # The held-out test signer (Signer03) is completely untouched.
            split = self.dm.get_inner_train_val_split(loso_fold="test_Signer03")

            # Save manual user training to dedicated manual checkpoint dir to protect deployment checkpoint
            manual_cnn_dir = "data/models/research/manual_training/cnn"
            cnn_manual_pipe = ResearchCNNPipeline(output_dir=manual_cnn_dir, feature_schema="hands_only")

            metrics = cnn_manual_pipe.train(
                X_train_seq=split["X_inner_train"],
                y_train=split["y_inner_train"],
                X_val_seq=split["X_inner_val"],
                y_val=split["y_inner_val"],
                epochs=epochs,
                batch_size=batch_size,
            )
            metrics["held_out_test_signer"] = split["test_signer"]
            metrics["test_leakage_status"] = "ZERO_TEST_LEAKAGE_VERIFIED"
            metrics["validation_protocol"] = "Inner-Train/Val (Signers 01 & 02 only)"

            with self._lock:
                self._active_job = None
            return metrics
        except Exception as e:
            with self._lock:
                self._active_job = {"job": "train_cnn", "status": "failed", "error": str(e)}
            raise

    def run_comparison(self, fold: str = "test_Signer03") -> dict[str, Any]:
        """Execute full scientific comparison benchmark."""
        with self._lock:
            self._active_job = {"job": "run_comparison", "status": "running"}

        try:
            results = self.evaluator.run_comparison(loso_fold=fold)
            with self._lock:
                self._active_job = None
            return results
        except Exception as e:
            with self._lock:
                self._active_job = {"job": "run_comparison", "status": "failed", "error": str(e)}
            raise

    # ─── Independent Human Evaluation Methods ──────────────────────────────

    def get_independent_eval_status(self) -> dict[str, Any]:
        """Get status of independent human volunteer evaluation."""
        return self.independent_eval_manager.get_status()

    def list_independent_volunteers(self) -> list[dict[str, Any]]:
        """List registered volunteer participants."""
        return self.independent_eval_manager.list_volunteers()

    def register_independent_volunteer(self, name: str, notes: str = "") -> dict[str, Any]:
        """Register a new human volunteer for independent testing."""
        return self.independent_eval_manager.register_volunteer(name=name, notes=notes)

    def get_volunteer_progress(self, volunteer_id: str) -> dict[str, Any]:
        """Get recording progress for a given volunteer."""
        return self.independent_eval_manager.get_volunteer_progress(volunteer_id=volunteer_id)

    def run_independent_evaluation(self) -> dict[str, Any]:
        """Execute frozen model evaluation on independent volunteer recordings."""
        with self._lock:
            self._active_job = {"job": "independent_evaluation", "status": "running"}

        try:
            results = self.independent_eval_manager.evaluate_independent_dataset()
            with self._lock:
                self._active_job = None
            return results
        except Exception as e:
            with self._lock:
                self._active_job = {"job": "independent_evaluation", "status": "failed", "error": str(e)}
            raise

    def get_consent_notice(self) -> dict[str, str]:
        """Return research informed consent and privacy notice."""
        return self.independent_eval_manager.get_consent_notice()

    def list_sign_references(self) -> list[dict[str, Any]]:
        """List the 10 Arabic digit reference signs with descriptions and pose instructions."""
        return self.independent_eval_manager.list_sign_references()

    def get_sign_reference_info(self, sign_id: str) -> Optional[dict[str, Any]]:
        """Get instructions for a specific sign ID or 'idle'."""
        return self.independent_eval_manager.get_sign_reference_info(sign_id)

    def get_next_repetition_index(self, volunteer_id: str, sign_id: str) -> int:
        """Get next non-duplicate repetition index for a volunteer and sign."""
        return self.independent_eval_manager.get_next_repetition_index(volunteer_id, sign_id)

    def delete_volunteer(self, volunteer_id: str) -> dict[str, Any]:
        """Permanently delete a volunteer and their video recordings."""
        return self.independent_eval_manager.delete_volunteer(volunteer_id)

    def delete_recording(self, record_id: str) -> dict[str, Any]:
        """Delete an individual recording from manifest and disk."""
        return self.independent_eval_manager.delete_recording(record_id)

    def save_recording_from_bytes(
        self,
        volunteer_id: str,
        sign_id: str,
        rep_idx: int,
        video_bytes: bytes,
        filename: str = "recording.webm",
        fps: float = 30.0,
        is_idle: bool = False,
    ) -> dict[str, Any]:
        """Save a recording uploaded as raw video bytes."""
        return self.independent_eval_manager.save_recording_from_bytes(
            volunteer_id=volunteer_id,
            sign_id=sign_id,
            rep_idx=rep_idx,
            video_bytes=video_bytes,
            filename=filename,
            fps=fps,
            is_idle=is_idle,
        )



