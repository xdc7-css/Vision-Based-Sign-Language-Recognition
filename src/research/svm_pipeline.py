"""Model A: Support Vector Machine (SVM) Classifier for KArSL Research.

Uses:
- Scikit-learn Support Vector Classifier (SVC)
- Engineered statistical and kinematic features (2,208 dims)
- StandardScaler strictly fitted on training data (no leakage)
- RBF and linear kernel tuning
- Latency and performance benchmarking
- Checkpoint persistence under data/models/research/svm/
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.research.karsl_dataset import SELECTED_RESEARCH_CLASSES
from src.research.preprocessing import ResearchFeatureExtractor

logger = logging.getLogger("arsl.research.svm")


class ResearchSVMPipeline:
    """Trains, tunes, and evaluates Model A (SVM) on KArSL-502 subsets."""

    def __init__(self, output_dir: Path | str = "data/models/research/svm") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[SVC] = None
        self.scaler: Optional[StandardScaler] = None
        self.classes = SELECTED_RESEARCH_CLASSES
        self.extractor = ResearchFeatureExtractor()
        self.metadata: dict[str, Any] = {}

        if self.is_trained():
            self.load_model()

    def is_trained(self) -> bool:
        model_file = self.output_dir / "svm_model.pkl"
        return model_file.exists()

    def load_model(self) -> bool:
        """Load trained SVM checkpoint, scaler, and metadata."""
        model_file = self.output_dir / "svm_model.pkl"
        if not model_file.exists():
            return False

        try:
            with open(model_file, "rb") as f:
                checkpoint = pickle.load(f)
            self.model = checkpoint["model"]
            self.scaler = checkpoint["scaler"]
            self.classes = checkpoint.get("classes", SELECTED_RESEARCH_CLASSES)
            self.metadata = checkpoint.get("metadata", {})
            logger.info("Research SVM model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Research SVM model: {e}")
            return False

    def train(
        self,
        X_train_seq: np.ndarray,
        y_train: np.ndarray,
        X_val_seq: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        kernel: str = "rbf",
        C: float = 2.0,
        gamma: str | float = "scale",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict[str, Any]:
        """Train SVM on engineered features with execution profiling."""
        logger.info(f"Training Research SVM (kernel={kernel}, C={C}, samples={len(X_train_seq)})...")
        t0 = time.perf_counter()
        self.classes = SELECTED_RESEARCH_CLASSES

        if progress_callback:
            progress_callback(1, 3, "استخراج المميزات وتوحيد القياس (StandardScaler)...")

        # 1. Feature Engineering & Scaling
        X_train_feats, scaler = self.extractor.prepare_svm_dataset(X_train_seq, fit_scaler=True)
        self.scaler = scaler

        if progress_callback:
            progress_callback(2, 3, f"تدريب مصنف SVM ({kernel.upper()} Kernel)...")

        # 2. Fit SVC model
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42,
            cache_size=500,
        )
        self.model.fit(X_train_feats, y_train)

        train_time = time.perf_counter() - t0
        logger.info(f"SVM training completed in {train_time:.2f}s.")

        # 3. Evaluate training metrics
        train_preds = self.model.predict(X_train_feats)
        train_acc = float(accuracy_score(y_train, train_preds))

        val_metrics: dict[str, float] = {}
        if X_val_seq is not None and y_val is not None and len(X_val_seq) > 0:
            X_val_feats, _ = self.extractor.prepare_svm_dataset(X_val_seq, scaler=self.scaler)
            val_preds = self.model.predict(X_val_feats)
            val_metrics = {
                "val_accuracy": float(accuracy_score(y_val, val_preds)),
                "val_precision_macro": float(precision_score(y_val, val_preds, average="macro", zero_division=0)),
                "val_recall_macro": float(recall_score(y_val, val_preds, average="macro", zero_division=0)),
                "val_f1_macro": float(f1_score(y_val, val_preds, average="macro", zero_division=0)),
            }

        # 4. Measure single-sample inference latency
        latency_ms = self._measure_inference_latency(X_train_seq[:1])

        if progress_callback:
            progress_callback(3, 3, "حفظ نموذج SVM والمقاييس...")

        # 5. Persist Checkpoint & Artifacts
        metrics = {
            "model_type": "svm",
            "kernel": kernel,
            "C": C,
            "gamma": str(gamma),
            "train_samples": len(X_train_seq),
            "feature_dim": X_train_feats.shape[1],
            "train_time_sec": round(train_time, 3),
            "inference_latency_ms": round(latency_ms, 2),
            "train_accuracy": round(train_acc, 4),
            **{k: round(v, 4) for k, v in val_metrics.items()},
        }

        self.save_artifacts(metrics)
        return metrics

    def _measure_inference_latency(self, sample_seq: np.ndarray, num_runs: int = 50) -> float:
        """Measure average end-to-end inference latency in milliseconds."""
        if self.model is None or self.scaler is None:
            return 0.0

        # Warm up
        feats = self.extractor.sequence_to_svm_features(sample_seq[0]).reshape(1, -1)
        scaled = self.scaler.transform(feats)
        self.model.predict_proba(scaled)

        # Profile
        times = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            f = self.extractor.sequence_to_svm_features(sample_seq[0]).reshape(1, -1)
            sc = self.scaler.transform(f)
            self.model.predict_proba(sc)
            times.append(time.perf_counter() - t_start)

        return float(np.mean(times) * 1000.0)

    def predict_sequence(self, sequence: np.ndarray) -> tuple[int, float, str, str]:
        """Predict class for a sequence of shape (T, 736).

        Returns:
            (class_index, confidence, name_ar, name_en)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("SVM model is not loaded. Train or load checkpoint first.")

        feats = self.extractor.sequence_to_svm_features(sequence).reshape(1, -1)
        scaled = self.scaler.transform(feats)
        probs = self.model.predict_proba(scaled)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        name_ar = self.classes[class_idx]["name_ar"]
        name_en = self.classes[class_idx]["name_en"]
        return class_idx, confidence, name_ar, name_en

    def save_artifacts(self, metrics: dict[str, Any]) -> None:
        """Persist checkpoint, classes mapping, metadata, and evaluation metrics."""
        model_file = self.output_dir / "svm_model.pkl"
        classes_file = self.output_dir / "classes.json"
        metrics_file = self.output_dir / "metrics.json"
        meta_file = self.output_dir / "metadata.json"

        checkpoint = {
            "model": self.model,
            "scaler": self.scaler,
            "classes": self.classes,
            "metadata": {
                "algorithm": "Support Vector Machine (SVC)",
                "framework": "scikit-learn",
                "kernel": metrics.get("kernel", "rbf"),
                "num_classes": len(self.classes),
                "feature_engineering": "mean, std, displacement (2,208 dims)",
                "timestamp": time.time(),
            },
        }

        with open(model_file, "wb") as f:
            pickle.dump(checkpoint, f)

        with open(classes_file, "w", encoding="utf-8") as f:
            json.dump(self.classes, f, ensure_ascii=False, indent=2)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        model_size_kb = model_file.stat().st_size / 1024.0
        meta = {
            **checkpoint["metadata"],
            "model_size_kb": round(model_size_kb, 1),
            "metrics": metrics,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"SVM checkpoint saved ({model_size_kb:.1f} KB).")

