"""DatasetAugmenter — data augmentation for sign language gesture images.

Techniques:
    - Random rotation (±15 degrees)
    - Random scaling (±10%)
    - Random translation (±5%)
    - Horizontal flip (maintain hand laterality)
    - Gaussian noise
    - Contrast/brightness adjustment
    - Hand-cropped augmentation (only augment hand ROI)
"""

from __future__ import annotations

import logging
import random

import cv2
import numpy as np


class DatasetAugmenter:
    """Augments gesture images for robust model training."""

    def __init__(
        self,
        rotation_deg: float = 15.0,
        scale_range: tuple[float, float] = (0.9, 1.1),
        translate_range: float = 0.05,
        noise_std: float = 0.005,
        contrast_range: tuple[float, float] = (0.8, 1.2),
        brightness_range: tuple[float, float] = (-0.1, 0.1),
        flip_probability: float = 0.3,
    ) -> None:
        self._rotation_deg = rotation_deg
        self._scale_range = scale_range
        self._translate_range = translate_range
        self._noise_std = noise_std
        self._contrast_range = contrast_range
        self._brightness_range = brightness_range
        self._flip_probability = flip_probability
        self._logger = logging.getLogger(__name__)

    def augment(self, image: np.ndarray) -> np.ndarray:
        result = image.astype(np.float32) / 255.0 if image.dtype != np.float32 else image.copy()

        if random.random() < 0.5:
            result = self._random_rotation(result)
        if random.random() < 0.5:
            result = self._random_scaling(result)
        if random.random() < 0.3:
            result = self._random_translation(result)
        if random.random() < 0.3:
            result = self._add_noise(result)
        if random.random() < 0.5:
            result = self._adjust_contrast_brightness(result)

        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return result

    def augment_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        return [self.augment(img) for img in images]

    def augment_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self._noise_std, landmarks.shape)
        return landmarks + noise

    def _random_rotation(self, img: np.ndarray) -> np.ndarray:
        angle = random.uniform(-self._rotation_deg, self._rotation_deg)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _random_scaling(self, img: np.ndarray) -> np.ndarray:
        scale = random.uniform(*self._scale_range)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _random_translation(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        tx = random.uniform(-self._translate_range, self._translate_range) * w
        ty = random.uniform(-self._translate_range, self._translate_range) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _add_noise(img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 0.005, img.shape).astype(np.float32)
        return img + noise

    @staticmethod
    def _adjust_contrast_brightness(img: np.ndarray) -> np.ndarray:
        contrast = random.uniform(0.8, 1.2)
        brightness = random.uniform(-0.1, 0.1)
        return img * contrast + brightness
