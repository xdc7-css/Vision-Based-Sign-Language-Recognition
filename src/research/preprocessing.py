"""Shared Data Preprocessing for KArSL-502 Scientific Comparison (SVM vs CNN).

Prepares consistent feature representations from landmark sequences:
- 1D Engineered statistical features for Model A (SVM)
- Temporal sequence tensors (Channels x Time) for Model B (1D CNN)
- StandardScaler training/transform with zero data leakage
- Live sliding-window feature transformation for real-time webcam inference
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("arsl.research.preprocessing")


class ResearchFeatureExtractor:
    """Transforms raw (T, 736) landmark sequences into model-specific representations."""

    def __init__(self, seq_len: int = 30, feat_dim: int = 736) -> None:
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.scaler: Optional[StandardScaler] = None

    # ─── SVM Feature Engineering ───────────────────────────────────────────

    def sequence_to_svm_features(self, sequence: np.ndarray) -> np.ndarray:
        """Engineers statistical and kinematic features from a single temporal sequence.

        Args:
            sequence: Array of shape (T, feat_dim) or (feat_dim,)

        Returns:
            1D array of engineered features combining:
            - Mean coordinates (736)
            - Standard deviation (dispersion) (736)
            - Velocity / Displacement (end - start) (736)
            Total: 2,208 dimensions
        """
        if sequence.ndim == 1:
            # Single-frame fallback: pad with zeros to form full descriptor
            mean_f = sequence
            std_f = np.zeros_like(sequence)
            vel_f = np.zeros_like(sequence)
            return np.concatenate([mean_f, std_f, vel_f], axis=0)

        # Multi-frame sequence
        mean_f = np.mean(sequence, axis=0)
        std_f = np.std(sequence, axis=0)
        vel_f = sequence[-1] - sequence[0]

        return np.concatenate([mean_f, std_f, vel_f], axis=0).astype(np.float32)

    def prepare_svm_dataset(
        self,
        sequences: np.ndarray,
        fit_scaler: bool = False,
        scaler: Optional[StandardScaler] = None,
    ) -> tuple[np.ndarray, StandardScaler]:
        """Convert batch of sequences (N, T, 736) to scaled SVM feature matrix (N, 2208)."""
        num_samples = len(sequences)
        svm_feats = np.zeros((num_samples, self.feat_dim * 3), dtype=np.float32)

        for i in range(num_samples):
            svm_feats[i] = self.sequence_to_svm_features(sequences[i])

        if fit_scaler:
            self.scaler = StandardScaler()
            scaled_feats = self.scaler.fit_transform(svm_feats)
        elif scaler is not None:
            self.scaler = scaler
            scaled_feats = self.scaler.transform(svm_feats)
        elif self.scaler is not None:
            scaled_feats = self.scaler.transform(svm_feats)
        else:
            raise ValueError("StandardScaler must be fitted on training data or provided.")

        return scaled_feats.astype(np.float32), self.scaler

    # ─── CNN Feature Preparation ───────────────────────────────────────────

    def prepare_cnn_dataset(self, sequences: np.ndarray) -> np.ndarray:
        """Convert batch of sequences (N, T, 736) to PyTorch Conv1d format: (N, 736, T).

        Conv1D in PyTorch expects input shape (Batch, Channels, Time).
        Here Channels = 736 landmark features, Time = 30 frames.
        """
        # sequences shape: (N, T, C) -> transpose to (N, C, T)
        cnn_feats = np.transpose(sequences, (0, 2, 1)).astype(np.float32)
        return cnn_feats

    # ─── Live Webcam Sliding Window Processing ────────────────────────────

    def resample_sequence(self, buffer: list[np.ndarray], target_len: int = 30) -> np.ndarray:
        """Resample a variable-length frame buffer to fixed target_len frames."""
        cur_len = len(buffer)
        if cur_len == 0:
            return np.zeros((target_len, self.feat_dim), dtype=np.float32)

        arr = np.array(buffer, dtype=np.float32)
        if cur_len == target_len:
            return arr

        # Linear interpolation across temporal dimension
        orig_indices = np.linspace(0, cur_len - 1, cur_len)
        target_indices = np.linspace(0, cur_len - 1, target_len)

        resampled = np.zeros((target_len, self.feat_dim), dtype=np.float32)
        for d in range(self.feat_dim):
            resampled[:, d] = np.interp(target_indices, orig_indices, arr[:, d])

        return resampled

