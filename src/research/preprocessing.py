"""Shared Data Preprocessing for KArSL-502 Scientific Comparison (SVM vs CNN).

Prepares consistent feature representations from landmark sequences:
- Focuses on hands (168 dims: Left Hand 84 + Right Hand 84) to eliminate noise from
  544 unobserved/zero facial channels and static upper-body pose.
- 1D Engineered statistical & kinematic features for Model A (SVM)
- Temporal sequence tensors (Channels x Time) for Model B (1D Temporal CNN)
- StandardScaler training/transform with zero data leakage (fit strictly on inner training split)
- Seamless sliding-window feature transformation for real-time webcam and video inference
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("arsl.research.preprocessing")

# Feature Slices in 736-dim vector:
# Pose: 0:24 (6 keypoints * 4)
# Face: 24:568 (136 keypoints * 4 - all 0.0 in KArSL video recordings)
# Right Hand: 568:652 (21 keypoints * 4)
# Left Hand: 652:736 (21 keypoints * 4)
HANDS_SLICE = slice(568, 736)  # 168 dimensions
HANDS_CHANNELS = 168


class ResearchFeatureExtractor:
    """Transforms raw landmark sequences into model-specific representations."""

    def __init__(
        self,
        seq_len: int = 30,
        feat_dim: int = 736,
        feature_schema: str = "hands_only",
    ) -> None:
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.feature_schema = feature_schema
        self.scaler: Optional[StandardScaler] = None

    @property
    def in_channels(self) -> int:
        return HANDS_CHANNELS if self.feature_schema == "hands_only" else self.feat_dim

    def select_channels(self, sequence: np.ndarray) -> np.ndarray:
        """Select relevant channels according to feature_schema.

        If sequence has 736 channels and schema is 'hands_only', extract 568:736 (168 dims).
        """
        if self.feature_schema == "hands_only":
            if sequence.shape[-1] == 736:
                return sequence[..., HANDS_SLICE]
            elif sequence.shape[-1] == HANDS_CHANNELS:
                return sequence
            else:
                # Fallback / clip if unexpected dimension
                return sequence[..., :HANDS_CHANNELS]
        return sequence

    # ─── SVM Feature Engineering ───────────────────────────────────────────

    def sequence_to_svm_features(self, sequence: np.ndarray) -> np.ndarray:
        """Engineers statistical and kinematic features from a single temporal sequence.

        Args:
            sequence: Array of shape (T, C) or (C,)

        Returns:
            1D array of engineered features combining:
            - Mean coordinates (168)
            - Standard deviation (dispersion) (168)
            - Velocity / Displacement (end - start) (168)
            - Range / Span (max - min) (168)
            Total: 672 dimensions (for 168 channels) or 2,944 dimensions (for 736 channels)
        """
        seq_active = self.select_channels(sequence)
        num_channels = seq_active.shape[-1]

        if seq_active.ndim == 1:
            # Single-frame fallback: pad with zeros to form full descriptor
            mean_f = seq_active
            std_f = np.zeros_like(seq_active)
            vel_f = np.zeros_like(seq_active)
            rng_f = np.zeros_like(seq_active)
            return np.concatenate([mean_f, std_f, vel_f, rng_f], axis=0).astype(np.float32)

        # Multi-frame sequence
        mean_f = np.mean(seq_active, axis=0)
        std_f = np.std(seq_active, axis=0)
        vel_f = seq_active[-1] - seq_active[0]
        rng_f = np.max(seq_active, axis=0) - np.min(seq_active, axis=0)

        return np.concatenate([mean_f, std_f, vel_f, rng_f], axis=0).astype(np.float32)

    def prepare_svm_dataset(
        self,
        sequences: np.ndarray,
        fit_scaler: bool = False,
        scaler: Optional[StandardScaler] = None,
    ) -> tuple[np.ndarray, StandardScaler]:
        """Convert batch of sequences (N, T, C) to scaled SVM feature matrix (N, feat_dim*4)."""
        num_samples = len(sequences)
        sample_feats = self.sequence_to_svm_features(sequences[0])
        total_feats_dim = len(sample_feats)

        svm_feats = np.zeros((num_samples, total_feats_dim), dtype=np.float32)
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
        """Convert batch of sequences (N, T, C) to PyTorch Conv1d format: (N, in_channels, T).

        Conv1D in PyTorch expects input shape (Batch, Channels, Time).
        For hands_only, Channels = 168 (Right Hand 84 + Left Hand 84), Time = 30 frames.
        """
        seq_active = self.select_channels(sequences)
        # sequences shape: (N, T, C) -> transpose to (N, C, T)
        cnn_feats = np.transpose(seq_active, (0, 2, 1)).astype(np.float32)
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

        dim = arr.shape[1]
        resampled = np.zeros((target_len, dim), dtype=np.float32)
        for d in range(dim):
            resampled[:, d] = np.interp(target_indices, orig_indices, arr[:, d])

        return resampled
