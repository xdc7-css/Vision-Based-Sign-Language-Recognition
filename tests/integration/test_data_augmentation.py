"""Integration tests — dataset augmentation module."""

import numpy as np
import pytest

from src.infrastructure.dataset.augmenter import DatasetAugmenter


class TestDatasetAugmenter:

    @pytest.fixture
    def augmenter(self) -> DatasetAugmenter:
        return DatasetAugmenter()

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_augment_returns_same_shape(self, augmenter: DatasetAugmenter, sample_image: np.ndarray):
        result = augmenter.augment(sample_image)
        assert result.shape == sample_image.shape

    def test_augment_returns_uint8(self, augmenter: DatasetAugmenter, sample_image: np.ndarray):
        result = augmenter.augment(sample_image)
        assert result.dtype == np.uint8

    def test_augment_batch(self, augmenter: DatasetAugmenter):
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        results = augmenter.augment_batch(images)
        assert len(results) == 5
        for r in results:
            assert r.shape == (480, 640, 3)

    def test_augment_landmarks(self, augmenter: DatasetAugmenter):
        landmarks = np.zeros((21, 3), dtype=np.float32)
        result = augmenter.augment_landmarks(landmarks)
        assert result.shape == (21, 3)
        assert not np.allclose(result, landmarks)  # noise added

    def test_deterministic_augmentation(self, augmenter: DatasetAugmenter):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result1 = augmenter.augment(img)
        result2 = augmenter.augment(img)
        # Due to randomness, results should differ
        assert not np.array_equal(result1, result2)

    def test_augment_with_max_quality(self):
        augmenter = DatasetAugmenter(
            rotation_deg=30.0,
            scale_range=(0.8, 1.2),
            translate_range=0.1,
            noise_std=0.01,
        )
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = augmenter.augment(img)
        assert result.shape == img.shape
