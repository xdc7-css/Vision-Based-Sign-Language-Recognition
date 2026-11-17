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

        self._active_job: Optional[dict[str, Any]] = None
        self._lock = threading.Lock()

    def get_status(self) -> dict[str, Any]:
        """Return comprehensive status of dataset, models, and latest comparison."""
        ds_status = self.dm.get_status()
        svm_trained = self.svm_pipeline.is_trained()
        cnn_trained = self.cnn_pipeline.is_trained()
        latest_comp = self.get_latest_comparison()

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
        """Train SVM model on prepared research dataset."""
        with self._lock:
            self._active_job = {"job": "train_svm", "status": "running"}

        try:
            X_all, y_all, _ = self.dm.load_features()
            splits = self.dm.load_loso_splits()["test_Signer03"]
            train_idx = splits["train_indices"]
            test_idx = splits["test_indices"]

            metrics = self.svm_pipeline.train(
                X_train_seq=X_all[train_idx],
                y_train=y_all[train_idx],
                X_val_seq=X_all[test_idx],
                y_val=y_all[test_idx],
                kernel=kernel,
                C=C,
            )
            with self._lock:
                self._active_job = None
            return metrics
        except Exception as e:
            with self._lock:
                self._active_job = {"job": "train_svm", "status": "failed", "error": str(e)}
            raise

    def train_cnn(self, epochs: int = 35, batch_size: int = 16) -> dict[str, Any]:
        """Train 1D Temporal CNN model on prepared research dataset."""
        with self._lock:
            self._active_job = {"job": "train_cnn", "status": "running"}

        try:
            X_all, y_all, _ = self.dm.load_features()
            splits = self.dm.load_loso_splits()["test_Signer03"]
            train_idx = splits["train_indices"]
            test_idx = splits["test_indices"]

            metrics = self.cnn_pipeline.train(
                X_train_seq=X_all[train_idx],
                y_train=y_all[train_idx],
                X_val_seq=X_all[test_idx],
                y_val=y_all[test_idx],
                epochs=epochs,
                batch_size=batch_size,
            )
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

