"""Model B: 1D Temporal Convolutional Neural Network (CNN) for KArSL Research.

Uses:
- PyTorch nn.Module with 1D temporal convolutions along the time dimension
- Input representation: (Batch, Channels=736, Time=30)
- Conv1D -> BatchNorm1d -> ReLU -> Dropout -> MaxPool1d blocks
- Global Adaptive Average Pooling -> Dense classification head
- Early stopping, learning rate scheduling, and execution profiling
- Checkpoint persistence under data/models/research/cnn/
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from src.research.karsl_dataset import SELECTED_RESEARCH_CLASSES
from src.research.preprocessing import ResearchFeatureExtractor

logger = logging.getLogger("arsl.research.cnn")


class TemporalSignCNN(nn.Module):
    """1D Temporal Convolutional Neural Network for Sign Language Recognition.

    Learns spatial-temporal gesture patterns across ordered sequences of
    MediaPipe body pose, face mesh, and dual-hand keypoints.
    """

    def __init__(self, in_channels: int = 736, num_classes: int = 12, dropout: float = 0.3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1: Initial temporal pattern detection (kernel size 5)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.MaxPool1d(2),  # T: 30 -> 15
        )

        # Block 2: Intermediate motion representation (kernel size 3)
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),  # T: 15 -> 7
        )

        # Block 3: Higher-level temporal dynamics
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),  # T: 7 -> 1
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects x of shape (Batch, in_channels=736, seq_len=30)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.classifier(x)
        return logits


class ResearchCNNPipeline:
    """Trains and evaluates Model B (1D Temporal CNN) using PyTorch."""

    def __init__(self, output_dir: Path | str = "data/models/research/cnn") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TemporalSignCNN] = None
        self.classes = SELECTED_RESEARCH_CLASSES
        self.extractor = ResearchFeatureExtractor()
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
            in_channels = checkpoint.get("in_channels", 736)

            self.model = TemporalSignCNN(in_channels=in_channels, num_classes=num_classes)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self.classes = checkpoint.get("classes", SELECTED_RESEARCH_CLASSES)
            self.metadata = checkpoint.get("metadata", {})
            logger.info(f"Research CNN model loaded successfully on {self.device}.")
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
        weight_decay: float = 1e-4,
        patience: int = 10,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> dict[str, Any]:
        """Train 1D Temporal CNN with early stopping and learning rate scheduling."""
        logger.info(f"Training Research CNN on {self.device} (epochs={epochs}, batch={batch_size})...")
        t0 = time.perf_counter()

        # Fix seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        self.classes = SELECTED_RESEARCH_CLASSES

        # 1. Transform to (N, C=736, T=30)
        X_train_tensor = torch.tensor(self.extractor.prepare_cnn_dataset(X_train_seq), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader: Optional[DataLoader] = None
        if X_val_seq is not None and y_val is not None and len(X_val_seq) > 0:
            X_val_tensor = torch.tensor(self.extractor.prepare_cnn_dataset(X_val_seq), dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 2. Initialize Model
        num_classes = len(self.classes)
        self.model = TemporalSignCNN(in_channels=736, num_classes=num_classes).to(self.device)

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

            # Validation step
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

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

            if progress_callback:
                progress_callback(epoch, epochs, float(train_loss), float(val_loss))

            if no_improve_epochs >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        train_time = time.perf_counter() - t0
        self.model.eval()

        # 3. Calculate final metrics
        with torch.no_grad():
            train_out = self.model(X_train_tensor.to(self.device))
            train_preds = torch.argmax(train_out, dim=1).cpu().numpy()
            train_acc = float(accuracy_score(y_train, train_preds))

        val_metrics: dict[str, float] = {}
        if val_loader:
            with torch.no_grad():
                val_out = self.model(X_val_tensor.to(self.device))
                val_preds = torch.argmax(val_out, dim=1).cpu().numpy()
                val_metrics = {
                    "val_accuracy": float(accuracy_score(y_val, val_preds)),
                    "val_precision_macro": float(precision_score(y_val, val_preds, average="macro", zero_division=0)),
                    "val_recall_macro": float(recall_score(y_val, val_preds, average="macro", zero_division=0)),
                    "val_f1_macro": float(f1_score(y_val, val_preds, average="macro", zero_division=0)),
                }

        # 4. Measure single-sample inference latency
        latency_ms = self._measure_inference_latency(X_train_seq[:1])

        # 5. Persist Checkpoint & Artifacts
        metrics = {
            "model_type": "cnn_1d_temporal",
            "epochs_run": epoch,
            "train_samples": len(X_train_seq),
            "train_time_sec": round(train_time, 3),
            "inference_latency_ms": round(latency_ms, 2),
            "train_accuracy": round(train_acc, 4),
            **{k: round(v, 4) for k, v in val_metrics.items()},
        }

        self.save_artifacts(metrics)
        return metrics

    def _measure_inference_latency(self, sample_seq: np.ndarray, num_runs: int = 50) -> float:
        """Measure average CNN forward-pass latency in milliseconds."""
        if self.model is None:
            return 0.0

        sample_tensor = torch.tensor(
            self.extractor.prepare_cnn_dataset(sample_seq),
            dtype=torch.float32,
            device=self.device,
        )

        # Warm up
        with torch.no_grad():
            self.model(sample_tensor)

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t_start = time.perf_counter()
                self.model(sample_tensor)
                times.append(time.perf_counter() - t_start)

        return float(np.mean(times) * 1000.0)

    def predict_sequence(self, sequence: np.ndarray) -> tuple[int, float, str, str]:
        """Predict class for a sequence of shape (T, 736).

        Returns:
            (class_index, confidence, name_ar, name_en)
        """
        if self.model is None:
            raise RuntimeError("CNN model is not loaded. Train or load checkpoint first.")

        batch_seq = sequence[np.newaxis, ...]  # (1, T, 736)
        x_tensor = torch.tensor(
            self.extractor.prepare_cnn_dataset(batch_seq),
            dtype=torch.float32,
            device=self.device,
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        name_ar = self.classes[class_idx]["name_ar"]
        name_en = self.classes[class_idx]["name_en"]
        return class_idx, confidence, name_ar, name_en

    def save_artifacts(self, metrics: dict[str, Any]) -> None:
        """Persist PyTorch model checkpoint, classes mapping, metadata, and evaluation metrics."""
        model_file = self.output_dir / "cnn_model.pt"
        classes_file = self.output_dir / "classes.json"
        metrics_file = self.output_dir / "metrics.json"
        meta_file = self.output_dir / "metadata.json"

        checkpoint = {
            "state_dict": self.model.state_dict() if self.model else {},
            "in_channels": 736,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "metadata": {
                "algorithm": "1D Temporal Convolutional Neural Network (Conv1D)",
                "framework": "PyTorch",
                "device": str(self.device),
                "num_classes": len(self.classes),
                "timestamp": time.time(),
            },
        }

        torch.save(checkpoint, model_file)

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

        logger.info(f"CNN checkpoint saved ({model_size_kb:.1f} KB).")

