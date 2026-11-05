"""DatasetValidator — integrity checking, outlier detection, balance analysis, quality reporting."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class SampleValidation:
    sample_id: str
    path: Path
    is_valid: bool
    issues: list[str] = field(default_factory=list)
    num_frames: int = 0
    num_landmarks: int = 0
    frame_shape: Optional[tuple] = None
    has_metadata: bool = False
    has_motion_data: bool = False
    is_outlier: bool = False
    outlier_score: float = 0.0


@dataclass
class ClassBalance:
    class_name: str
    sample_count: int
    is_balanced: bool = True
    recommended_samples: int = 0


@dataclass
class ClassStatistics:
    class_name: str
    sample_count: int
    avg_frames: float
    min_frames: int
    max_frames: int
    avg_duration: float
    valid_count: int
    invalid_count: int
    outlier_count: int


@dataclass
class DatasetReport:
    dataset_type: str
    total_samples: int
    total_classes: int
    valid_samples: int
    invalid_samples: int
    outlier_samples: int
    corrupt_samples: int
    incomplete_samples: int
    quality_score: float
    is_balanced: bool
    class_balances: list[ClassBalance] = field(default_factory=list)
    class_statistics: list[ClassStatistics] = field(default_factory=list)
    sample_validations: list[SampleValidation] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class DatasetValidator:
    """Validates dataset integrity, detects issues, and generates quality reports."""

    MIN_FRAMES = 5
    MIN_LANDMARKS = 21
    LANDMARK_DIM = 3

    def __init__(self, dataset_dir: str | Path = "data/datasets") -> None:
        self._dataset_dir = Path(dataset_dir)

    # ── Public API ──────────────────────────────────────────────────

    def validate_sample(self, sample_dir: Path) -> SampleValidation:
        """Validate a single sample directory."""
        sample_id = sample_dir.name
        issues: list[str] = []

        # Check metadata
        meta_file = sample_dir / "metadata.json"
        has_metadata = meta_file.exists()
        if not has_metadata:
            issues.append("Missing metadata.json")

        landmarks_file = sample_dir / "landmarks.npy"
        motion_file = sample_dir / "motion_data.npy"
        has_motion = motion_file.exists()

        num_frames = 0
        num_landmarks = 0
        frame_shape = None

        if landmarks_file.exists():
            try:
                data = np.load(landmarks_file, allow_pickle=False)
                shape = data.shape
                frame_shape = shape

                if data.ndim != 3:
                    issues.append(f"Landmarks should be 3D array, got {data.ndim}D")
                else:
                    num_frames = shape[0]
                    num_landmarks = shape[1]

                    if num_frames < self.MIN_FRAMES:
                        issues.append(f"Too few frames: {num_frames} < {self.MIN_FRAMES}")
                    if shape[1] != self.MIN_LANDMARKS:
                        issues.append(f"Wrong landmark count: {shape[1]} != {self.MIN_LANDMARKS}")
                    if shape[2] != self.LANDMARK_DIM:
                        issues.append(f"Wrong landmark dimension: {shape[2]} != {self.LANDMARK_DIM}")

                    if np.isnan(data).any():
                        issues.append("Landmarks contain NaN values")
                    if np.isinf(data).any():
                        issues.append("Landmarks contain inf values")
                    if (data[:, :, :2] < 0).any() or (data[:, :, :2] > 1).any():
                        issues.append("Landmarks out of normalized range [0, 1]")

            except Exception as e:
                issues.append(f"Corrupt landmarks file: {e}")
        else:
            issues.append("Missing landmarks.npy")

        return SampleValidation(
            sample_id=sample_id,
            path=sample_dir,
            is_valid=len(issues) == 0,
            issues=issues,
            num_frames=num_frames,
            num_landmarks=num_landmarks,
            frame_shape=frame_shape,
            has_metadata=has_metadata,
            has_motion_data=has_motion,
        )

    def detect_outliers(self, samples: list[Path], z_threshold: float = 2.0) -> list[SampleValidation]:
        """Detect outlier samples based on frame count z-scores."""
        validations = [self.validate_sample(s) for s in samples]
        frame_counts = [v.num_frames for v in validations if v.is_valid]

        if len(frame_counts) < 3:
            return validations

        mean_frames = statistics.mean(frame_counts)
        stdev_frames = statistics.stdev(frame_counts) if len(frame_counts) > 1 else 0.0

        if stdev_frames == 0:
            return validations

        for v in validations:
            if v.is_valid and v.num_frames > 0:
                z = abs(v.num_frames - mean_frames) / stdev_frames
                v.is_outlier = z > z_threshold
                v.outlier_score = z

        return validations

    def check_balance(self, dataset_type: str) -> list[ClassBalance]:
        """Check class balance for a dataset type."""
        type_dir = self._dataset_dir / dataset_type
        if not type_dir.exists():
            return []

        class_counts: dict[str, int] = {}
        for class_dir in type_dir.iterdir():
            if class_dir.is_dir():
                samples_dir = class_dir / "samples"
                count = len(list(samples_dir.iterdir())) if samples_dir.exists() else 0
                class_counts[class_dir.name] = count

        if not class_counts:
            return []

        avg_count = statistics.mean(class_counts.values())
        balances = [
            ClassBalance(
                class_name=name,
                sample_count=count,
                is_balanced=abs(count - avg_count) / max(avg_count, 1) < 0.3,
                recommended_samples=int(avg_count),
            )
            for name, count in sorted(class_counts.items())
        ]
        return balances

    def generate_report(self, dataset_type: str) -> DatasetReport:
        """Generate a comprehensive dataset quality report."""
        type_dir = self._dataset_dir / dataset_type
        if not type_dir.exists():
            return DatasetReport(
                dataset_type=dataset_type,
                total_samples=0, total_classes=0,
                valid_samples=0, invalid_samples=0,
                outlier_samples=0, corrupt_samples=0,
                incomplete_samples=0, quality_score=0.0,
                is_balanced=True,
                recommendations=["Dataset directory does not exist."],
            )

        # Collect all sample directories
        all_samples: list[Path] = []
        for class_dir in type_dir.iterdir():
            if class_dir.is_dir():
                samples_dir = class_dir / "samples"
                if samples_dir.exists():
                    all_samples.extend(samples_dir.iterdir())

        # Validate all samples
        validations = self.detect_outliers(all_samples)
        balances = self.check_balance(dataset_type)

        # Compute statistics
        valid = [v for v in validations if v.is_valid]
        invalid = [v for v in validations if not v.is_valid]
        outliers = [v for v in validations if v.is_outlier]
        corrupt = [v for v in validations if "corrupt" in str(v.issues).lower() or not v.has_metadata]
        incomplete = [v for v in validations if "frames" in str(v.issues)]

        # Class-level statistics
        class_stats: list[ClassStatistics] = []
        for class_dir in type_dir.iterdir():
            if class_dir.is_dir():
                class_samples = [
                    v for v in validations
                    if str(v.path.parent.parent) == str(class_dir)
                ]
                frame_counts = [v.num_frames for v in class_samples if v.is_valid]
                durations = [0.0]

                for v in class_samples:
                    meta_file = v.path / "metadata.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file) as f:
                                meta = json.load(f)
                            durations.append(meta.get("duration_seconds", 0.0))
                        except (json.JSONDecodeError, OSError):
                            pass

                avg_frames = statistics.mean(frame_counts) if frame_counts else 0.0
                avg_duration = statistics.mean(durations)
                min_frames = min(frame_counts) if frame_counts else 0
                max_frames = max(frame_counts) if frame_counts else 0

                class_stats.append(ClassStatistics(
                    class_name=class_dir.name,
                    sample_count=len(class_samples),
                    avg_frames=avg_frames,
                    min_frames=min_frames,
                    max_frames=max_frames,
                    avg_duration=avg_duration,
                    valid_count=len(class_samples) - invalid.count(class_samples),
                    invalid_count=invalid.count(class_samples),
                    outlier_count=outliers.count(class_samples),
                ))

        # Quality score (0-100)
        n = len(validations)
        quality_score = 0.0
        if n > 0:
            valid_ratio = len(valid) / n
            outlier_penalty = len(outliers) / max(n, 1) * 0.3
            balance_penalty = sum(1 for b in balances if not b.is_balanced) / max(len(balances), 1) * 0.2
            quality_score = max(0.0, (valid_ratio - outlier_penalty - balance_penalty) * 100)

        # Recommendations
        recommendations: list[str] = []
        if len(valid) < n:
            recommendations.append(f"Fix or remove {n - len(valid)} invalid samples.")
        if outliers:
            recommendations.append(f"Review {len(outliers)} outlier samples (z-score > 2.0).")
        for b in balances:
            if not b.is_balanced:
                recommendations.append(
                    f"Class '{b.class_name}' has {b.sample_count} samples "
                    f"(recommended ~{b.recommended_samples})."
                )
        if not recommendations:
            recommendations.append("Dataset looks healthy. Continue collecting samples.")

        return DatasetReport(
            dataset_type=dataset_type,
            total_samples=n,
            total_classes=len(class_stats),
            valid_samples=len(valid),
            invalid_samples=len(invalid),
            outlier_samples=len(outliers),
            corrupt_samples=len(corrupt),
            incomplete_samples=len(incomplete),
            quality_score=round(quality_score, 1),
            is_balanced=all(b.is_balanced for b in balances),
            class_balances=balances,
            class_statistics=class_stats,
            sample_validations=validations,
            recommendations=recommendations,
        )

    def print_report(self, report: DatasetReport) -> None:
        """Pretty-print a dataset report to the console."""
        print(f"\n{'='*60}")
        print(f"  Dataset Report: {report.dataset_type}")
        print(f"{'='*60}")
        print(f"  Total samples:    {report.total_samples}")
        print(f"  Total classes:    {report.total_classes}")
        print(f"  Valid samples:    {report.valid_samples}")
        print(f"  Invalid samples:  {report.invalid_samples}")
        print(f"  Outliers:         {report.outlier_samples}")
        print(f"  Corrupt:          {report.corrupt_samples}")
        print(f"  Incomplete:       {report.incomplete_samples}")
        print(f"  Quality score:    {report.quality_score:.1f}/100")
        print(f"  Balanced:         {report.is_balanced}")
        print(f"\n  {'Class':<20} {'Samples':>8} {'Valid':>6} {'Invalid':>8} {'Avg Frames':>10}")
        print(f"  {'-'*52}")
        for cs in report.class_statistics:
            print(f"  {cs.class_name:<20} {cs.sample_count:>8} {cs.valid_count:>6} "
                  f"{cs.invalid_count:>8} {cs.avg_frames:>8.1f}")
        print(f"\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    • {rec}")
        print(f"{'='*60}\n")
