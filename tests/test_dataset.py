"""Tests for dataset collection, organization, and validation modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.domain.recognition import GestureCategory
from src.infrastructure.dataset.collector import DatasetCollector
from src.infrastructure.dataset.organizer import DatasetOrganizer
from src.infrastructure.dataset.validator import DatasetValidator


# ── DatasetCollector tests ────────────────────────────────────────────


class TestDatasetCollector:
    @pytest.fixture
    def collector(self) -> DatasetCollector:
        return DatasetCollector(base_dir=Path(tempfile.mkdtemp()))

    def test_start_stop_session(self, collector: DatasetCollector) -> None:
        session = collector.start_session(GestureCategory.STATIC_WORD, "hello")
        assert session.is_recording
        assert session.gesture_label == "hello"

        collector.stop_session()
        assert not collector.is_recording

    def test_save_sample(self, collector: DatasetCollector) -> None:
        session = collector.start_session(GestureCategory.ALPHABET, "alif")
        # Add synthetic frame data
        from src.domain.hand import HandDetectionResult
        result = HandDetectionResult([], 640, 480, 5.0)
        session.add_frame(result, (0.5, 0.5))

        # Add synthetic landmarks
        session.landmarks.append(np.zeros((21, 3), dtype=np.float32))

        collector.stop_session()
        sample_dir = collector.save_sample()

        assert sample_dir is not None
        assert sample_dir.exists()
        assert (sample_dir / "landmarks.npy").exists()
        assert (sample_dir / "metadata.json").exists()

    def test_delete_sample(self, collector: DatasetCollector) -> None:
        # Create and save a sample
        session = collector.start_session(GestureCategory.STATIC_WORD, "test")
        session.landmarks.append(np.zeros((21, 3), dtype=np.float32))
        collector.stop_session()
        sample_dir = collector.save_sample()

        assert sample_dir is not None
        assert sample_dir.exists()

        assert collector.delete_sample(sample_dir)
        assert not sample_dir.exists()

    def test_list_samples_empty(self, collector: DatasetCollector) -> None:
        samples = collector.list_samples()
        assert samples == []

    def test_metadata_fields(self, collector: DatasetCollector) -> None:
        session = collector.start_session(GestureCategory.DYNAMIC_GESTURE, "how_are_you")
        session.landmarks = [np.zeros((21, 3), dtype=np.float32) for _ in range(10)]
        collector.stop_session()
        meta = collector._build_metadata(session)

        assert meta.gesture_label == "how_are_you"
        assert meta.num_frames == 0  # no frames added via add_frame
        assert "dynamic" in meta.gesture_type


# ── DatasetOrganizer tests ────────────────────────────────────────────


class TestDatasetOrganizer:
    @pytest.fixture
    def organizer(self) -> DatasetOrganizer:
        return DatasetOrganizer(base_dir=Path(tempfile.mkdtemp()))

    def test_ensure_structure(self, organizer: DatasetOrganizer) -> None:
        organizer.ensure_structure()
        for t in ["alphabet", "words", "dynamic"]:
            assert (Path(organizer._base_dir) / t).exists()

    def test_create_class(self, organizer: DatasetOrganizer) -> None:
        class_dir = organizer.create_class("alphabet", "alif")
        assert class_dir.exists()
        assert (class_dir / "samples").exists()

    def test_list_classes(self, organizer: DatasetOrganizer) -> None:
        organizer.create_class("alphabet", "alif")
        organizer.create_class("alphabet", "baa")
        classes = organizer.list_classes("alphabet")
        assert "alif" in classes
        assert "baa" in classes

    def test_count_samples(self, organizer: DatasetOrganizer) -> None:
        organizer.create_class("words", "hello")
        samples_dir = organizer.get_samples_dir("words", "hello")
        (samples_dir / "sample_1").mkdir()
        (samples_dir / "sample_2").mkdir()
        assert organizer.count_samples("words", "hello") == 2

    def test_class_statistics(self, organizer: DatasetOrganizer) -> None:
        organizer.create_class("alphabet", "alif")
        organizer.create_class("alphabet", "baa")
        stats = organizer.class_statistics("alphabet")
        assert stats == {"alif": 0, "baa": 0}

    def test_write_class_info(self, organizer: DatasetOrganizer) -> None:
        organizer.create_class("alphabet", "alif")
        organizer.write_class_info("alphabet", "alif", {"arabic": "أ", "english": "Alif"})
        info = organizer.read_class_info("alphabet", "alif")
        assert info["arabic"] == "أ"

    def test_rename_class(self, organizer: DatasetOrganizer) -> None:
        organizer.create_class("alphabet", "old_name")
        assert organizer.rename_class("alphabet", "old_name", "new_name")
        assert "new_name" in organizer.list_classes("alphabet")

    def test_delete_class(self, organizer: DatasetOrganizer) -> None:
        organizer.create_class("alphabet", "temp_class")
        assert organizer.delete_class("alphabet", "temp_class")
        assert "temp_class" not in organizer.list_classes("alphabet")


# ── DatasetValidator tests ────────────────────────────────────────────


class TestDatasetValidator:
    @pytest.fixture
    def validator(self) -> DatasetValidator:
        return DatasetValidator(dataset_dir=Path(tempfile.mkdtemp()))

    def _create_valid_sample(self, base_dir: Path, class_name: str, sample_id: str) -> Path:
        """Create a synthetic valid sample."""
        sample_dir = base_dir / class_name / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Create landmarks: 30 frames × 21 landmarks × 3 coords
        landmarks = np.random.rand(30, 21, 3).astype(np.float32)
        np.save(sample_dir / "landmarks.npy", landmarks)

        # Create metadata
        meta = {
            "sample_id": sample_id,
            "gesture_type": "static",
            "gesture_label": class_name,
            "num_frames": 30,
            "num_hands": 1,
            "duration_seconds": 1.0,
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

        return sample_dir

    def test_validate_valid_sample(self, validator: DatasetValidator) -> None:
        sample_dir = self._create_valid_sample(
            Path(validator._dataset_dir) / "words", "hello", "sample_001"
        )
        result = validator.validate_sample(sample_dir)
        assert result.is_valid
        assert result.num_frames == 30
        assert result.num_landmarks == 21
        assert result.has_metadata

    def test_validate_missing_landmarks(self, validator: DatasetValidator) -> None:
        sample_dir = Path(validator._dataset_dir) / "words" / "hello" / "samples" / "bad_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)

        result = validator.validate_sample(sample_dir)
        assert not result.is_valid
        assert any("landmarks" in issue.lower() for issue in result.issues)

    def test_validate_too_few_frames(self, validator: DatasetValidator) -> None:
        sample_dir = Path(validator._dataset_dir) / "words" / "hello" / "samples" / "short"
        sample_dir.mkdir(parents=True, exist_ok=True)

        landmarks = np.random.rand(2, 21, 3).astype(np.float32)
        np.save(sample_dir / "landmarks.npy", landmarks)

        result = validator.validate_sample(sample_dir)
        assert not result.is_valid
        assert any("frames" in issue.lower() for issue in result.issues)

    def test_validate_wrong_landmark_count(self, validator: DatasetValidator) -> None:
        sample_dir = Path(validator._dataset_dir) / "words" / "hello" / "samples" / "wrong"
        sample_dir.mkdir(parents=True, exist_ok=True)

        landmarks = np.random.rand(10, 15, 3).astype(np.float32)
        np.save(sample_dir / "landmarks.npy", landmarks)

        result = validator.validate_sample(sample_dir)
        assert not result.is_valid

    def test_generate_report(self, validator: DatasetValidator) -> None:
        # Create some samples
        for i, cls in enumerate(["hello", "thanks", "sorry"]):
            for j in range(3):
                self._create_valid_sample(
                    Path(validator._dataset_dir) / "words", cls, f"{cls}_{j:03d}"
                )

        report = validator.generate_report("words")
        assert report.total_samples == 9
        assert report.total_classes == 3
        assert report.valid_samples == 9
        assert report.invalid_samples == 0
        assert report.quality_score > 50
        assert len(report.class_statistics) == 3

    def test_detect_outliers(self, validator: DatasetValidator) -> None:
        # Create samples with varying frame counts
        for i in range(5):
            sample_dir = Path(validator._dataset_dir) / "words" / "hello" / "samples" / f"sample_{i:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            n_frames = 30 + i * 5
            landmarks = np.random.rand(n_frames, 21, 3).astype(np.float32)
            np.save(sample_dir / "landmarks.npy", landmarks)
            with open(sample_dir / "metadata.json", "w") as f:
                import json
                json.dump({"sample_id": f"sample_{i:03d}", "num_frames": n_frames}, f)

        samples = list((Path(validator._dataset_dir) / "words" / "hello" / "samples").iterdir())
        results = validator.detect_outliers(samples)
        # All should be valid (not enough variance for outliers with small sample)
        assert all(r.is_valid for r in results)

    def test_check_balance(self, validator: DatasetValidator) -> None:
        self._create_valid_sample(Path(validator._dataset_dir) / "words", "hello", "s1")
        self._create_valid_sample(Path(validator._dataset_dir) / "words", "hello", "s2")
        self._create_valid_sample(Path(validator._dataset_dir) / "words", "thanks", "s3")

        balances = validator.check_balance("words")
        assert len(balances) == 2
        for b in balances:
            assert b.sample_count >= 1
