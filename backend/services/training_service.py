"""Asynchronous model training service for ArSL Studio."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from src.models.train_multimodal import (
    inspect_multimodal_dataset,
    train_multimodal_model,
)

logger = logging.getLogger(__name__)


class TrainingJob:
    """State tracking for a single background training execution."""

    def __init__(self, job_id: str, max_epochs: int) -> None:
        self.job_id = job_id
        self.status = "running"  # "running", "completed", "failed", "cancelled"
        self.progress = 0.0
        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.loss = 0.0
        self.accuracy = 0.0
        self.metrics: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.should_cancel = False
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "current_epoch": self.current_epoch,
            "max_epochs": self.max_epochs,
            "loss": round(float(self.loss), 4),
            "accuracy": round(float(self.accuracy), 4),
            "metrics": self.metrics,
            "error": self.error,
        }


class TrainingService:
    """Manages thread-safe model training and readiness verification."""

    def __init__(
        self,
        dataset_dir: str = "data/datasets/words",
        models_dir: str = "data/models/multimodal",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()
        self._active_job: Optional[TrainingJob] = None
        self._jobs_history: dict[str, TrainingJob] = {}

    def get_readiness(self) -> dict[str, Any]:
        """Inspect datasets on disk and return training readiness."""
        inspection = inspect_multimodal_dataset(self.dataset_dir)
        
        trainable_classes = len(inspection.get("accepted_classes", {}))
        total_samples = sum(inspection.get("multimodal_class_counts", {}).values())
        
        # Check model file mtime vs sample dirs
        model_path = self.models_dir / "static_model.pkl"
        model_status = "Untrained"
        last_trained = None

        if model_path.exists():
            last_trained = model_path.stat().st_mtime
            model_status = "Ready"
            
            # Check if any sample was added/modified after model training
            if self.dataset_dir.exists():
                for c_dir in self.dataset_dir.iterdir():
                    s_dir = c_dir / "samples"
                    if s_dir.exists() and s_dir.stat().st_mtime > last_trained:
                        model_status = "Outdated"
                        break

        return {
            "is_trainable": bool(inspection.get("is_trainable", False)),
            "trainable_classes_count": trainable_classes,
            "total_multimodal_samples": total_samples,
            "accepted_classes": inspection.get("accepted_classes", {}),
            "insufficient_classes": inspection.get("insufficient_classes", {}),
            "model_status": model_status,
            "reason": inspection.get("reason"),
            "last_trained_timestamp": last_trained,
        }

    def start_training(
        self,
        model_type: str = "mlp",
        epochs: int = 60,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
        allowed_classes: Optional[list[str]] = None,
        on_complete_callback: Optional[Callable[[], None]] = None,
    ) -> TrainingJob:
        """Launch background training job if no job is currently running."""
        with self._lock:
            if self._active_job is not None and self._active_job.status == "running":
                raise RuntimeError("Another training job is already in progress.")

            job_id = str(uuid.uuid4())[:8]
            job = TrainingJob(job_id=job_id, max_epochs=epochs)
            self._active_job = job
            self._jobs_history[job_id] = job

        # Spawn worker thread
        thread = threading.Thread(
            target=self._run_training_worker,
            args=(job, model_type, epochs, batch_size, learning_rate, val_split, allowed_classes, on_complete_callback),
            daemon=True,
        )
        thread.start()
        return job

    def _run_training_worker(
        self,
        job: TrainingJob,
        model_type: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        val_split: float,
        allowed_classes: Optional[list[str]],
        on_complete_callback: Optional[Callable[[], None]],
    ) -> None:
        def progress_cb(ep: int, max_ep: int, loss: float, acc: float) -> None:
            if job.should_cancel:
                raise InterruptedError("Training was cancelled by the user.")
            job.current_epoch = ep
            job.max_epochs = max_ep
            job.loss = loss
            job.accuracy = acc
            job.progress = (ep / max_ep) if max_ep > 0 else 0.0

        try:
            results = train_multimodal_model(
                dataset_path=self.dataset_dir,
                output_dir=self.models_dir,
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_split=val_split,
                progress_callback=progress_cb,
                allowed_classes=allowed_classes,
            )
            job.metrics = results.get("metrics")
            job.status = "completed"
            job.progress = 1.0
            job.end_time = time.time()
            logger.info(f"Training job {job.job_id} completed successfully.")

            if on_complete_callback is not None:
                try:
                    on_complete_callback()
                except Exception as e:
                    logger.error(f"Error in training on_complete_callback: {e}")

        except InterruptedError:
            job.status = "cancelled"
            job.error = "Training was cancelled by user."
            job.end_time = time.time()
            logger.info(f"Training job {job.job_id} was cancelled.")
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.end_time = time.time()
            logger.error(f"Training job {job.job_id} failed: {e}", exc_info=True)

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self._jobs_history.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs_history.get(job_id)
        if job and job.status == "running":
            job.should_cancel = True
            return True
        return False

