"""DynamicSignRecognizer — temporal sequence classification for motion-based gestures.

Compares LSTM, GRU, and Transformer architectures. Auto-selects the best
model based on validation metrics (accuracy, precision, recall, latency).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from src.domain.motion import HandMotionTrack
from src.domain.recognition import GestureCategory, RecognitionResult
from src.infrastructure.inference.base import BaseInferenceEngine


@dataclass
class DynamicModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_time_s: float
    inference_time_ms: float
    model_size_kb: float
    architecture: str
    params_count: int = 0

    def score(self) -> float:
        return (self.accuracy * 40 + self.f1_score * 30 +
                (1.0 - self.inference_time_ms / 200) * 20 +
                (1.0 - self.model_size_kb / 5000) * 10)


@dataclass
class DynamicModelConfig:
    architecture: str = "lstm"
    sequence_length: int = 30
    n_features: int = 63
    n_hidden: int = 128
    n_layers: int = 2
    n_classes: int = 10
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    test_size: float = 0.2
    random_state: int = 42


class DynamicSignRecognizer:
    """Temporal gesture classifier supporting LSTM, GRU, and Transformer.

    Operates on sequences of MediaPipe landmarks over time
    (shape: T × 63 where T is the number of frames).
    """

    SUPPORTED_ARCHS = ("lstm", "gru", "transformer")

    def __init__(self, config: Optional[DynamicModelConfig] = None) -> None:
        self._config = config or DynamicModelConfig()
        self._model: Any = None
        self._label_encoder: Any = None
        self._architecture: str = ""
        self._metrics: Optional[DynamicModelMetrics] = None
        self._inference_engine: Optional[BaseInferenceEngine] = None

    # ── Public API ──────────────────────────────────────────────────

    def set_inference_engine(self, engine: Optional[BaseInferenceEngine]) -> None:
        self._inference_engine = engine

    def predict(
        self,
        landmark_sequence: np.ndarray,
        confidence_threshold: float = 0.0,
    ) -> RecognitionResult:
        """Classify a sequence of landmark frames.

        Args:
            landmark_sequence: (T × 63) or (T × 21 × 3) array.
            confidence_threshold: Minimum confidence to accept.

        Returns:
            RecognitionResult with gesture classification.
        """
        seq = self._prepare_sequence(landmark_sequence)
        if seq is None:
            return RecognitionResult(
                gesture_id="", arabic_text="",
                category=GestureCategory.DYNAMIC_GESTURE,
                confidence=0.0, is_rejected=True,
                rejection_reason="Invalid sequence",
            )

        # Prefer inference engine over raw PyTorch model
        if self._inference_engine is not None and self._inference_engine.is_loaded:
            ir = self._inference_engine.predict(seq, confidence_threshold)
            pred_class = ir.label_id
            confidence = ir.confidence
            label = ir.label_name
        elif self._model is not None:
            import torch
            self._model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(seq).unsqueeze(0)
                if next(self._model.parameters()).is_cuda:
                    x_tensor = x_tensor.cuda()
                logits = self._model(x_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            confidence = float(np.max(probs))
            pred_class = int(np.argmax(probs))
            label = self._label_encoder.inverse_transform([pred_class])[0] if self._label_encoder else str(pred_class)
        else:
            return RecognitionResult(
                gesture_id="", arabic_text="",
                category=GestureCategory.DYNAMIC_GESTURE,
                confidence=0.0, is_rejected=True,
                rejection_reason="Model not loaded",
            )

        gesture_id = f"dynamic_{pred_class}"

        if confidence < confidence_threshold:
            return RecognitionResult(
                gesture_id=gesture_id, arabic_text="",
                category=GestureCategory.DYNAMIC_GESTURE,
                confidence=confidence, is_rejected=True,
                rejection_reason=f"Low confidence: {confidence:.3f}",
            )

        return RecognitionResult(
            gesture_id=gesture_id,
            arabic_text=str(label),
            category=GestureCategory.DYNAMIC_GESTURE,
            confidence=confidence,
        )

    def predict_from_track(self, track: HandMotionTrack, confidence_threshold: float = 0.0) -> RecognitionResult:
        """Classify from a HandMotionTrack object."""
        if not track.landmark_sequences:
            return RecognitionResult(
                gesture_id="", arabic_text="",
                category=GestureCategory.DYNAMIC_GESTURE,
                confidence=0.0, is_rejected=True,
                rejection_reason="Empty track",
            )
        seq = np.array(track.landmark_sequences)
        return self.predict(seq, confidence_threshold)

    # ── Training ────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        architectures: Optional[list[str]] = None,
    ) -> dict[str, DynamicModelMetrics]:
        """Train all architectures on GPU if available.

        Uses mini-batch training, validation split, early stopping,
        and learning rate scheduling.

        Args:
            X: (n_samples × T × 63) landmark sequences.
            y: (n_samples,) labels.
            architectures: Subset of ('lstm', 'gru', 'transformer').

        Returns:
            Dict of architecture → DynamicModelMetrics.
        """
        import torch

        archs = architectures or list(self.SUPPORTED_ARCHS)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from torch import nn, optim
        from torch.utils.data import DataLoader, TensorDataset

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        n_classes = len(self._label_encoder.classes_)
        self._config.n_classes = n_classes

        # Pad sequences to uniform length
        X_pad = self._pad_sequences(X)

        # Train/val/test split (80/10/10)
        X_train, X_test, y_train, y_test = train_test_split(
            X_pad, y_encoded,
            test_size=self._config.test_size,
            random_state=self._config.random_state,
            stratify=y_encoded,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.125,  # 10% of total
            random_state=self._config.random_state,
            stratify=y_train,
        )

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)

        # Create DataLoader for mini-batch training
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
        )

        all_metrics: dict[str, DynamicModelMetrics] = {}
        best_score = -1.0

        for arch in archs:
            if arch not in self.SUPPORTED_ARCHS:
                continue

            model = self._build_model(arch, n_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self._config.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5,
            )

            # Early stopping state
            best_val_loss = float("inf")
            best_model_state = None
            patience_counter = 0
            patience = 10

            # Train with mini-batches
            train_start = time.perf_counter()
            model.train()
            for epoch in range(self._config.epochs):
                epoch_loss = 0.0
                num_batches = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                model.train()

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if epoch >= 10:
                            break

            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            train_time = time.perf_counter() - train_start

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_t)
                _, predicted = torch.max(outputs, 1)

            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            acc = accuracy_score(y_test, predicted.cpu().numpy())
            prec = precision_score(y_test, predicted.cpu().numpy(), average="weighted", zero_division=0)
            rec = recall_score(y_test, predicted.cpu().numpy(), average="weighted", zero_division=0)
            f1 = f1_score(y_test, predicted.cpu().numpy(), average="weighted", zero_division=0)

            # Measure inference time
            infer_start = time.perf_counter()
            with torch.no_grad():
                _ = model(X_test_t[:1])
            infer_time = (time.perf_counter() - infer_start) * 1000

            # Model size
            import pickle
            size = len(pickle.dumps(model.state_dict())) / 1024
            params = sum(p.numel() for p in model.parameters())

            metrics = DynamicModelMetrics(
                accuracy=acc, precision=prec, recall=rec, f1_score=f1,
                train_time_s=round(train_time, 2), inference_time_ms=round(infer_time, 3),
                model_size_kb=round(size, 2), architecture=arch, params_count=params,
            )
            all_metrics[arch] = metrics

            s = metrics.score()
            if s > best_score:
                best_score = s
                self._model = model
                self._architecture = arch
                self._metrics = metrics

        return all_metrics

    def load(self, model_path: str | Path) -> bool:
        import torch
        model_path = Path(model_path)
        if not model_path.exists():
            return False

        checkpoint = torch.load(model_path, map_location="cpu")
        n_classes = checkpoint.get("n_classes", self._config.n_classes)
        arch = checkpoint.get("architecture", "lstm")

        self._model = self._build_model(arch, n_classes)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._architecture = arch
        self._label_encoder = checkpoint.get("label_encoder")
        self._metrics = checkpoint.get("metrics")
        self._model.eval()
        return True

    def save(self, model_dir: str | Path) -> Path:
        import torch
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        path = model_dir / f"dynamic_{self._architecture}.pt"
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "architecture": self._architecture,
            "label_encoder": self._label_encoder,
            "n_classes": self._config.n_classes,
            "config": self._config,
            "metrics": self._metrics,
        }, path)

        if self._metrics:
            m = self._metrics
            with open(model_dir / f"dynamic_{self._architecture}_metrics.json", "w") as f:
                json.dump({
                    "architecture": m.architecture,
                    "accuracy": m.accuracy,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "train_time_s": m.train_time_s,
                    "inference_time_ms": m.inference_time_ms,
                    "params_count": m.params_count,
                    "model_size_kb": m.model_size_kb,
                }, f, indent=2)

        return path

    # ── Private ─────────────────────────────────────────────────────

    def _build_model(self, architecture: str, n_classes: int):

        if architecture == "lstm":
            return _LSTMModel(self._config, n_classes)
        elif architecture == "gru":
            return _GRUModel(self._config, n_classes)
        elif architecture == "transformer":
            return _TransformerModel(self._config, n_classes)
        raise ValueError(f"Unknown architecture: {architecture}")

    def _prepare_sequence(self, seq: np.ndarray) -> Optional[np.ndarray]:
        if seq.ndim == 3 and seq.shape[2] == 3:
            seq = seq.reshape(seq.shape[0], -1)
        if seq.ndim != 2 or seq.shape[1] != 63:
            return None
        return self._pad_sequences(seq[np.newaxis, ...])[0]

    def _pad_sequences(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.ndim == 2:
            sequences = sequences[np.newaxis, ...]
        target_len = self._config.sequence_length
        n_samples, n_frames, n_feats = sequences.shape

        if n_frames >= target_len:
            return sequences[:, :target_len, :]

        padded = np.zeros((n_samples, target_len, n_feats), dtype=np.float32)
        padded[:, :n_frames, :] = sequences
        # Repeat last frame for padding (more natural than zeros)
        if n_frames > 0:
            padded[:, n_frames:, :] = sequences[:, -1:, :]
        return padded

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metrics(self) -> Optional[DynamicModelMetrics]:
        return self._metrics

    @property
    def architecture(self) -> str:
        return self._architecture


# ── PyTorch model definitions ─────────────────────────────────────────

if torch is None:
    _LSTMModel = None  # type: ignore
    _GRUModel = None
    _TransformerModel = None
    _PositionalEncoding = None
else:

    class _LSTMModel(torch.nn.Module):
        def __init__(self, config: DynamicModelConfig, n_classes: int):
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=config.n_features,
                hidden_size=config.n_hidden,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                batch_first=True,
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(config.n_hidden, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(64, n_classes),
            )

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            out = h_n[-1]
            return self.classifier(out)

    class _GRUModel(torch.nn.Module):
        def __init__(self, config: DynamicModelConfig, n_classes: int):
            super().__init__()
            self.gru = torch.nn.GRU(
                input_size=config.n_features,
                hidden_size=config.n_hidden,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                batch_first=True,
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(config.n_hidden, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(64, n_classes),
            )

        def forward(self, x):
            _, h_n = self.gru(x)
            out = h_n[-1]
            return self.classifier(out)

    class _TransformerModel(torch.nn.Module):
        def __init__(self, config: DynamicModelConfig, n_classes: int):
            super().__init__()
            self.input_proj = torch.nn.Linear(config.n_features, config.n_hidden)
            self.pos_encoding = _PositionalEncoding(config.n_hidden, config.sequence_length)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=config.n_hidden,
                nhead=4,
                dim_feedforward=config.n_hidden * 2,
                dropout=config.dropout,
                batch_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
            self.classifier = torch.nn.Linear(config.n_hidden, n_classes)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_encoding(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    class _PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model: int, max_len: int):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]
