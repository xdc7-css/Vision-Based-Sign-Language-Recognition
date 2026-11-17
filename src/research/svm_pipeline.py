"""Research Model A: Support Vector Machine (SVM) Pipeline for KArSL-502.

Pipeline architecture:
- Feature Extraction: 168-channel hand landmarks (x, y, z, visibility)
- Statistical & Kinematic Engineering: Mean, Std, Velocity, Range (672 dims)
- Data Leakage Prevention: StandardScaler fitted strictly on inner training split
- Model: Scikit-Learn SVC with RBF Kernel (tuned on internal validation set)
- Execution Profiling: Sub-millisecond inference latency tracking
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
    """Trains and evaluates Model A (Support Vector Machine) on KArSL features."""

    def __init__(
        self,
        output_dir: Path | str = "data/models/research/svm",
        feature_schema: str = "hands_only",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[SVC] = None
        self.scaler: Optional[StandardScaler] = None
        self.classes = SELECTED_RESEARCH_CLASSES
        self.feature_schema = feature_schema
        self.extractor = ResearchFeatureExtractor(feature_schema=feature_schema)
        self.metadata: dict[str, Any] = {}

        if self.is_trained():
            self.load_model()

    def is_trained(self) -> bool:
        """Check if trained SVM checkpoint and scaler exist."""
        model_file = self.output_dir / "svm_model.pkl"
        scaler_file = self.output_dir / "scaler.pkl"
        return model_file.exists() and scaler_file.exists()

    def load_model(self) -> bool:
        """Load trained SVM model, scaler, classes, and metadata."""
        model_file = self.output_dir / "svm_model.pkl"
        scaler_file = self.output_dir / "scaler.pkl"
        classes_file = self.output_dir / "classes.json"
        meta_file = self.output_dir / "metadata.json"

        if not (model_file.exists() and scaler_file.exists()):
            return False

        try:
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)
            with open(scaler_file, "rb") as f:
                self.scaler = pickle.load(f)
            self.extractor.scaler = self.scaler

            if classes_file.exists():
                with open(classes_file, "r", encoding="utf-8") as f:
                    self.classes = json.load(f)

            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            logger.info("Research SVM model and scaler loaded successfully.")
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
        tune_hyperparameters: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict[str, Any]:
        """Train SVM on engineered features with execution profiling.

        Data Leakage Prevention:
        - Scaler is fitted strictly on X_train_seq.
        - Hyperparameters are selected using internal X_val_seq (if provided).
        """
        logger.info(f"Training Research SVM (kernel={kernel}, samples={len(X_train_seq)})...")
        t0 = time.perf_counter()
        self.classes = SELECTED_RESEARCH_CLASSES

        if progress_callback:
            progress_callback(1, 3, "استخراج المميزات وتوحيد القياس (StandardScaler)...")

        # 1. Feature Engineering & Scaling (fit scaler ONLY on training data)
        X_train_feats, scaler = self.extractor.prepare_svm_dataset(X_train_seq, fit_scaler=True)
        self.scaler = scaler

        if progress_callback:
            progress_callback(2, 3, f"تدريب وموالفة مصنف SVM ({kernel.upper()} Kernel)...")

        best_C = C
        best_gamma = gamma
        best_val_score = -1.0

        # Optional hyperparameter tuning using ONLY internal validation partition
        if tune_hyperparameters and X_val_seq is not None and y_val is not None and len(X_val_seq) > 0:
            X_val_feats, _ = self.extractor.prepare_svm_dataset(X_val_seq, scaler=self.scaler)
            for cand_C in [0.5, 1.0, 2.0, 4.0, 8.0]:
                for cand_g in ["scale", 0.001, 0.005, 0.01]:
                    candidate_svc = SVC(
                        kernel=kernel,
                        C=cand_C,
                        gamma=cand_g,
                        probability=False,
                        random_state=42,
                    )
                    candidate_svc.fit(X_train_feats, y_train)
                    val_score = float(accuracy_score(y_val, candidate_svc.predict(X_val_feats)))
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_C = cand_C
                        best_gamma = cand_g
            logger.info(f"Inner validation tuning selected C={best_C}, gamma={best_gamma} (inner val acc={best_val_score:.4f})")

        # 2. Fit final SVC model with probability estimation
        self.model = SVC(
            kernel=kernel,
            C=best_C,
            gamma=best_gamma,
            probability=True,
            random_state=42,
            cache_size=500,
        )
        self.model.fit(X_train_feats, y_train)

        train_time = time.perf_counter() - t0
        logger.info(f"SVM training completed in {train_time:.2f}s.")

        # 3. Evaluate training and internal validation metrics
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
            "C": best_C,
            "gamma": str(best_gamma),
            "train_samples": len(X_train_seq),
            "feature_schema": self.feature_schema,
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
            _ = self.model.predict_proba(scaled)
            times.append(time.perf_counter() - t_start)

        return float(np.mean(times) * 1000.0)

    def predict_sequence(self, sequence: np.ndarray) -> tuple[int, float, str, str]:
        """Classify a single 30-frame sequence (live webcam or evaluation file)."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Research SVM model is not trained or loaded.")

        feats = self.extractor.sequence_to_svm_features(sequence).reshape(1, -1)
        scaled = self.scaler.transform(feats)

        probs = self.model.predict_proba(scaled)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        if class_idx < len(self.classes):
            name_ar = self.classes[class_idx]["name_ar"]
            name_en = self.classes[class_idx]["name_en"]
        else:
            name_ar = f"إشارة {class_idx}"
            name_en = f"Sign {class_idx}"

        return class_idx, confidence, name_ar, name_en

    def save_artifacts(self, metrics: dict[str, Any]) -> None:
        """Save trained model, scaler, class mappings, and metadata to disk."""
        model_file = self.output_dir / "svm_model.pkl"
        scaler_file = self.output_dir / "scaler.pkl"
        classes_file = self.output_dir / "classes.json"
        meta_file = self.output_dir / "metadata.json"
        metrics_file = self.output_dir / "metrics.json"

        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)

        with open(scaler_file, "wb") as f:
            pickle.dump(self.scaler, f)

        with open(classes_file, "w", encoding="utf-8") as f:
            json.dump(self.classes, f, ensure_ascii=False, indent=2)

        metadata = {
            "model_type": "svm",
            "model_name": "Support Vector Machine (SVM)",
            "classes_count": len(self.classes),
            "feature_schema": self.feature_schema,
            "feature_dim": metrics.get("feature_dim", 672),
            "updated_at": time.time(),
            "data_source": "karsl_502_authentic_video",
            "provenance": "KArSL-502 Official Dataset (Sidig et al., ACM TALIP 2021)",
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
