"""StaticSignRecognizer — landmark-based static gesture classification.

Compares Random Forest, SVM, and MLP classifiers. Auto-selects the
best model based on validation accuracy. Trains on flattened MediaPipe
landmark vectors (63 features per hand).
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.domain.hand import Hand
from src.domain.recognition import RecognitionResult, GestureCategory
from src.infrastructure.hand_detection.utils import LandmarkProcessor
from src.infrastructure.inference.base import BaseInferenceEngine


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_time_ms: float
    inference_time_ms: float
    model_size_kb: float
    model_type: str

    def score(self) -> float:
        """Composite score (0-100) for model selection."""
        return (self.accuracy * 50 + self.f1_score * 30 + (1.0 - self.inference_time_ms / 100) * 20)


@dataclass
class StaticModelConfig:
    model_type: str = "mlp"
    test_size: float = 0.2
    random_state: int = 42

    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int = 20

    # SVM
    svm_kernel: str = "rbf"
    svm_C: float = 10.0
    svm_gamma: str = "scale"

    # MLP
    mlp_hidden_layers: tuple[int, ...] = (128, 64)
    mlp_max_iter: int = 500
    mlp_learning_rate_init: float = 0.001


class StaticSignRecognizer:
    """Static hand-shape gesture classifier.

    Supports three backends: 'rf' (Random Forest), 'svm', 'mlp'.
    Auto-selects the best performing model on validation data.
    """

    SUPPORTED_TYPES = ("rf", "svm", "mlp")

    def __init__(self, config: Optional[StaticModelConfig] = None) -> None:
        self._config = config or StaticModelConfig()
        self._model: Any = None
        self._label_encoder: Any = None
        self._model_type: str = ""
        self._metrics: Optional[ModelMetrics] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._inference_engine: Optional[BaseInferenceEngine] = None

    # ── Public API ──────────────────────────────────────────────────

    def set_inference_engine(self, engine: Optional[BaseInferenceEngine]) -> None:
        self._inference_engine = engine

    def predict(self, hand: Hand, confidence_threshold: float = 0.0) -> RecognitionResult:
        """Classify a single hand shape.

        Args:
            hand: Detected hand with 21 landmarks.
            confidence_threshold: Minimum confidence to accept prediction.

        Returns:
            RecognitionResult with gesture_id, arabic_text, confidence.
        """
        features = self._extract_features_from_hand(hand)

        # Prefer inference engine (ONNX/TensorRT) over raw sklearn model
        if self._inference_engine is not None and self._inference_engine.is_loaded:
            ir = self._inference_engine.predict(features, confidence_threshold)
            label_id = ir.label_id
            confidence = ir.confidence
            gesture_id = f"static_{label_id}"
            display_text = str(label_id).replace("_", " ")
        elif self._model is not None:
            features_scaled = self._scale_features(features)
            raw_pred = self._model.predict([features_scaled])[0]
            if self._label_encoder is not None:
                try:
                    class_name = str(self._label_encoder.inverse_transform([raw_pred])[0])
                except Exception:
                    class_name = str(raw_pred)
            else:
                class_name = str(raw_pred)

            probabilities = self._predict_proba(features_scaled)
            confidence = float(np.max(probabilities))
            gesture_id = f"static_{class_name}"
            display_text = class_name.replace("_", " ")
        else:
            return RecognitionResult(
                gesture_id="", arabic_text="",
                category=GestureCategory.STATIC_WORD,
                confidence=0.0, is_rejected=True,
                rejection_reason="Model not loaded",
            )

        if confidence < confidence_threshold:
            return RecognitionResult(
                gesture_id=gesture_id,
                arabic_text=display_text,
                category=GestureCategory.STATIC_WORD,
                confidence=confidence,
                is_rejected=True,
                rejection_reason=f"Low confidence: {confidence:.3f}",
            )

        return RecognitionResult(
            gesture_id=gesture_id,
            arabic_text=display_text,
            category=GestureCategory.STATIC_WORD,
            confidence=confidence,
        )

    def predict_batch(
        self, hands: list[Hand], confidence_threshold: float = 0.0
    ) -> list[RecognitionResult]:
        return [self.predict(h, confidence_threshold) for h in hands]

    # ── Training ────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_types: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> dict[str, ModelMetrics]:
        """Train and evaluate all model types. Auto-selects best.

        Args:
            X: Feature matrix (n_samples × n_features).
            y: Label array (n_samples,).
            model_types: List of model types to train ('rf', 'svm', 'mlp').
            progress_callback: Optional callback(epoch, total_epochs, loss, val_loss).

        Returns:
            Dict of model_type → ModelMetrics.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        model_types = model_types or list(self.SUPPORTED_TYPES)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        # Check if stratification is possible
        from collections import Counter
        class_counts = Counter(y_encoded)
        can_stratify = len(class_counts) >= 2 and all(count >= 2 for count in class_counts.values())

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self._config.test_size,
            random_state=self._config.random_state,
            stratify=y_encoded if can_stratify else None,
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self._feature_mean = scaler.mean_
        self._feature_std = scaler.scale_

        # Train each model type
        all_metrics: dict[str, ModelMetrics] = {}
        best_score = -1.0
        best_type = ""

        for mt in model_types:
            if mt not in self.SUPPORTED_TYPES:
                continue

            metrics = self._train_single(
                mt, X_train_scaled, y_train, X_test_scaled, y_test,
                progress_callback=progress_callback,
            )
            all_metrics[mt] = metrics

            if metrics.score() > best_score:
                best_score = metrics.score()
                best_type = mt

        # Auto-select best model
        if best_type:
            # Reload best model
            self._model_type = best_type
            self._model = all_metrics[best_type]
            # Retrain on full data
            X_all_scaled = scaler.transform(X)
            self._retrain_best(best_type, X_all_scaled, y_encoded)
            self._metrics = all_metrics[best_type]

        return all_metrics

    def load(self, model_path: str | Path) -> bool:
        """Load a trained model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            return False

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self._model = data.get("model")
        self._label_encoder = data.get("label_encoder")
        self._model_type = data.get("model_type", "")
        self._feature_mean = data.get("feature_mean")
        self._feature_std = data.get("feature_std")
        self._metrics = data.get("metrics")

        return self._model is not None

    def save(self, model_dir: str | Path) -> Path:
        """Save the trained model to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"static_{self._model_type}.pkl"
        canonical_path = model_dir / "static_model.pkl"
        data = {
            "model": self._model,
            "label_encoder": self._label_encoder,
            "model_type": self._model_type,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "metrics": self._metrics,
            "config": self._config,
        }
        with open(model_path, "wb") as f:
            pickle.dump(data, f)
        with open(canonical_path, "wb") as f:
            pickle.dump(data, f)

        # Save class vocabulary mapping
        if self._label_encoder is not None:
            classes_path = model_dir / "classes.json"
            with open(classes_path, "w", encoding="utf-8") as f:
                json.dump(list(self._label_encoder.classes_), f, ensure_ascii=False, indent=2)

        # Save metrics separately
        if self._metrics:
            metrics_path = model_dir / f"static_{self._model_type}_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model_type": self._model_type,
                    "accuracy": self._metrics.accuracy,
                    "precision": self._metrics.precision,
                    "recall": self._metrics.recall,
                    "f1_score": self._metrics.f1_score,
                    "train_time_ms": self._metrics.train_time_ms,
                }, f, indent=2)

        return canonical_path

    @property
    def metrics(self) -> Optional[ModelMetrics]:
        return self._metrics

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Private ─────────────────────────────────────────────────────

    def _extract_features_from_hand(self, hand: Hand) -> np.ndarray:
        return LandmarkProcessor.normalize_to_flat(hand.landmarks)

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        if self._feature_mean is None or self._feature_std is None:
            return features
        return (features - self._feature_mean) / (self._feature_std + 1e-10)

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba([features])[0]
        # Fallback: return confidence 1.0 for predicted class
        probs = np.zeros(len(self._label_encoder.classes_))
        probs[self._model.predict([features])[0]] = 1.0
        return probs

    def _train_single(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> ModelMetrics:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        import warnings

        start = time.perf_counter()

        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=self._config.rf_n_estimators,
                max_depth=self._config.rf_max_depth,
                random_state=self._config.random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            if progress_callback:
                progress_callback(1, 1, 0.05, 0.05)
        elif model_type == "svm":
            model = SVC(
                kernel=self._config.svm_kernel,
                C=self._config.svm_C,
                gamma=self._config.svm_gamma,
                probability=True,
                random_state=self._config.random_state,
            )
            model.fit(X_train, y_train)
            if progress_callback:
                progress_callback(1, 1, 0.05, 0.05)
        elif model_type == "mlp":
            epochs = self._config.mlp_max_iter
            model = MLPClassifier(
                hidden_layer_sizes=self._config.mlp_hidden_layers,
                max_iter=1,
                warm_start=True,
                learning_rate_init=self._config.mlp_learning_rate_init,
                random_state=self._config.random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for ep in range(1, epochs + 1):
                    model.max_iter = ep
                    model.fit(X_train, y_train)
                    if progress_callback:
                        loss = float(model.loss_)
                        val_acc = float(model.score(X_test, y_test))
                        val_loss = max(0.0, 1.0 - val_acc)
                        progress_callback(ep, epochs, loss, val_loss)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        train_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = (time.perf_counter() - start) * 1000

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        import pickle
        size = len(pickle.dumps(model)) / 1024

        self._model = model
        self._model_type = model_type

        return ModelMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            train_time_ms=round(train_time, 2),
            inference_time_ms=round(infer_time, 2),
            model_size_kb=round(size, 2),
            model_type=model_type,
        )

    def _retrain_best(self, model_type: str, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC

        if model_type == "rf":
            self._model = RandomForestClassifier(
                n_estimators=self._config.rf_n_estimators,
                max_depth=self._config.rf_max_depth,
                random_state=self._config.random_state,
                n_jobs=-1,
            )
        elif model_type == "svm":
            self._model = SVC(
                kernel=self._config.svm_kernel,
                C=self._config.svm_C,
                gamma=self._config.svm_gamma,
                probability=True,
                random_state=self._config.random_state,
            )
        elif model_type == "mlp":
            self._model = MLPClassifier(
                hidden_layer_sizes=self._config.mlp_hidden_layers,
                max_iter=self._config.mlp_max_iter * 2,
                learning_rate_init=self._config.mlp_learning_rate_init,
                random_state=self._config.random_state,
                early_stopping=True,
            )

        self._model.fit(X, y)
        self._model_type = model_type
