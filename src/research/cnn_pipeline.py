"""Research Model B: 1D Temporal Convolutional Neural Network (PyTorch).

Architecture for KArSL-502 Dynamic Gesture Recognition:
- Input: 168-channel hand landmark trajectories across T=30 frames: (Batch, Channels=168, Time=30)
- Temporal Convolution Blocks:
  - Conv1D(168 -> 64, k=3) + BatchNorm + ReLU + MaxPool1D(2) + Dropout(0.3)
  - Conv1D(64 -> 128, k=3) + BatchNorm + ReLU + MaxPool1D(2) + Dropout(0.3)
  - Conv1D(128 -> 128, k=3) + BatchNorm + ReLU + AdaptiveAvgPool1D(1)
- Multi-layer classification head: Dropout(0.3) -> Linear(128, 64) -> ReLU -> Linear(64, 10)
- Zero-Leakage Protocol: Early stopping and LR scheduling monitor internal validation loss ONLY
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.research.karsl_dataset import SELECTED_RESEARCH_CLASSES
from src.research.preprocessing import ResearchFeatureExtractor

logger = logging.getLogger("arsl.research.cnn")


class TemporalSignCNN(nn.Module):
    """1D Temporal Convolutional Neural Network for sign gesture sequence classification."""

    def __init__(self, in_channels: int = 168, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1: Channels=168 -> 64, Temporal resolution 30 -> 15
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

        # Block 2: 64 -> 128, Temporal resolution 15 -> 7
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

        # Block 3: 128 -> 128, Global Temporal Aggregation
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects x of shape (Batch, in_channels, seq_len=30)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x).squeeze(-1)
        logits = self.classifier(x)
        return logits


class ResearchCNNPipeline:
    """Trains and evaluates Model B (1D Temporal CNN) using PyTorch."""

    def __init__(
        self,
        output_dir: Path | str = "data/models/research/cnn",
        feature_schema: str = "hands_only",
        in_channels: int = 168,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TemporalSignCNN] = None
        self.classes = SELECTED_RESEARCH_CLASSES
        self.feature_schema = feature_schema
        self.in_channels = in_channels
        self.extractor = ResearchFeatureExtractor(feature_schema=feature_schema)
        self.metadata: dict[str, Any] = {}

        if self.is_trained():
            self.load_model()

    def is_trained(self) -> bool:
        model_file = self.output_dir / "cnn_model.pt"
        return model_file.exists()

    def load_model(self) -> bool:
        """Load trained PyTorch CNN checkpoint."""
        model_file = self.output_dir / "cnn_model.pt"
        if not model_file.exists():
            return False

        try:
            checkpoint = torch.load(model_file, map_location=self.device)
            num_classes = checkpoint.get("num_classes", len(self.classes))
            in_channels = checkpoint.get("in_channels", self.in_channels)
            self.in_channels = in_channels

            self.model = TemporalSignCNN(in_channels=in_channels, num_classes=num_classes)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self.classes = checkpoint.get("classes", SELECTED_RESEARCH_CLASSES)
            self.metadata = checkpoint.get("metadata", {})
            logger.info(f"Research CNN model loaded successfully on {self.device} (in_channels={in_channels}).")
            return True
        except Exception as e:
            logger.error(f"Failed to load Research CNN model: {e}")
            return False

    def train(
        self,
        X_train_seq: np.ndarray,
        y_train: np.ndarray,
        X_val_seq: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 40,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-3,
        patience: int = 8,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> dict[str, Any]:
        """Train 1D Temporal CNN with zero-leakage internal early stopping.

        Data Leakage Prevention:
        - Early stopping and ReduceLROnPlateau monitor internal validation loss ONLY.
        - Outer held-out test data must never be passed as X_val_seq.
        """
        logger.info(f"Training Research CNN on {self.device} (epochs={epochs}, batch={batch_size}, channels={self.in_channels})...")
        t0 = time.perf_counter()

        # Fix seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        self.classes = SELECTED_RESEARCH_CLASSES

        # 1. Transform to (N, C, T=30)
        X_train_prep = self.extractor.prepare_cnn_dataset(X_train_seq)
        self.in_channels = X_train_prep.shape[1]

        X_train_tensor = torch.tensor(X_train_prep, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader: Optional[DataLoader] = None
        if X_val_seq is not None and y_val is not None and len(X_val_seq) > 0:
            X_val_prep = self.extractor.prepare_cnn_dataset(X_val_seq)
            X_val_tensor = torch.tensor(X_val_prep, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 2. Initialize Model
        num_classes = len(self.classes)
        self.model = TemporalSignCNN(in_channels=self.in_channels, num_classes=num_classes).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

        best_val_loss = float("inf")
        best_state = None
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_y)

            train_loss /= len(train_dataset)

            # Validation step (Internal Validation Only)
            val_loss = train_loss
            if val_loader:
                self.model.eval()
                val_loss_acc = 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        vo = self.model(vx)
                        v_loss = criterion(vo, vy)
                        val_loss_acc += v_loss.item() * len(vy)
                val_loss = val_loss_acc / len(val_dataset)
                scheduler.step(val_loss)

                # Early stopping check on internal validation
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (best internal val loss: {best_val_loss:.4f})")
                    break

            if progress_callback:
                progress_callback(epoch, epochs, float(train_loss), float(val_loss))

        # Restore best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        train_time = time.perf_counter() - t0
        self.model.eval()

        # Compute training accuracy
        with torch.no_grad():
            train_preds = self.model(X_train_tensor.to(self.device)).argmax(dim=1).cpu().numpy()
            train_acc = float((train_preds == y_train).mean())

        # Measure latency
        latency_ms = self._measure_inference_latency(X_train_seq[:1])

        metrics = {
            "model_type": "cnn",
            "in_channels": self.in_channels,
            "feature_schema": self.feature_schema,
            "train_samples": len(X_train_seq),
            "epochs_trained": epoch,
            "best_internal_val_loss": round(float(best_val_loss), 4) if best_val_loss != float("inf") else None,
            "train_time_sec": round(train_time, 3),
            "inference_latency_ms": round(latency_ms, 2),
            "train_accuracy": round(train_acc, 4),
        }

        self.save_artifacts(metrics)
        return metrics

    def _measure_inference_latency(self, sample_seq: np.ndarray, num_runs: int = 50) -> float:
        """Measure average model inference latency in milliseconds."""
        if self.model is None:
            return 0.0

        sample_prep = self.extractor.prepare_cnn_dataset(sample_seq)
        sample_tensor = torch.tensor(sample_prep, dtype=torch.float32, device=self.device)

        self.model.eval()
        # Warm up
        with torch.no_grad():
            _ = self.model(sample_tensor)

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = self.model(sample_tensor)
                times.append(time.perf_counter() - t0)

        return float(np.mean(times) * 1000.0)

    def predict_sequence(self, sequence: np.ndarray) -> tuple[int, float, str, str]:
        """Classify a single 30-frame sequence (live webcam or evaluation file)."""
        if self.model is None:
            raise RuntimeError("Research CNN model is not trained or loaded.")

        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        seq_prep = self.extractor.prepare_cnn_dataset(sequence)
        tensor = torch.tensor(seq_prep, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

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
        """Save model checkpoint, classes, and metadata to disk."""
        model_file = self.output_dir / "cnn_model.pt"
        classes_file = self.output_dir / "classes.json"
        meta_file = self.output_dir / "metadata.json"
        metrics_file = self.output_dir / "metrics.json"

        if self.model is not None:
            checkpoint = {
                "in_channels": self.in_channels,
                "num_classes": len(self.classes),
                "state_dict": self.model.state_dict(),
                "classes": self.classes,
                "metadata": {
                    "architecture": "TemporalSignCNN-1D",
                    "in_channels": self.in_channels,
                    "seq_len": 30,
                    "feature_schema": self.feature_schema,
                    "updated_at": time.time(),
                },
            }
            torch.save(checkpoint, model_file)

        with open(classes_file, "w", encoding="utf-8") as f:
            json.dump(self.classes, f, ensure_ascii=False, indent=2)

        metadata = {
            "model_type": "cnn",
            "model_name": "1D Temporal CNN",
            "architecture": "TemporalSignCNN-1D (PyTorch)",
            "in_channels": self.in_channels,
            "classes_count": len(self.classes),
            "feature_schema": self.feature_schema,
            "updated_at": time.time(),
            "data_source": "karsl_502_authentic_video",
            "provenance": "KArSL-502 Official Dataset (Sidig et al., ACM TALIP 2021)",
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
