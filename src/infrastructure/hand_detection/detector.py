"""HandDetector service — wraps MediaPipe Hands for real-time hand landmark detection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.domain.hand import (
    Hand,
    HandDetectionResult,
    HandLandmark,
    Handedness,
)


@dataclass
class HandDetectorConfig:
    """Configuration for the MediaPipe hand detection pipeline."""

    max_num_hands: int = 2
    model_complexity: int = 0
    min_detection_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    static_image_mode: bool = False


class HandDetector:
    """Real-time hand landmark detector powered by MediaPipe Hands.

    Usage:
        detector = HandDetector()
        result = detector.detect(frame)
        for hand in result.hands:
            print(hand.handedness, hand.confidence)
        detector.close()
    """

    def __init__(self, config: Optional[HandDetectorConfig] = None) -> None:
        self._config = config or HandDetectorConfig()
        self._validate_config()
        self._mp_hands: object = None
        self._pipeline: object = None
        self._initialize_mediapipe()

    # ── Public API ──────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> HandDetectionResult:
        """Run hand detection on a single BGR or RGB image.

        Args:
            image: BGR or RGB numpy array (H×W×3).

        Returns:
            HandDetectionResult with zero, one, or two hands.
        """
        if image is None or image.size == 0:
            return HandDetectionResult([], 0, 0, 0.0)

        h, w = image.shape[:2]
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        rgb = np.ascontiguousarray(rgb)

        start = time.perf_counter()
        mp_result = self._pipeline.process(rgb)  # type: ignore
        elapsed = (time.perf_counter() - start) * 1000.0

        hands = self._extract_hands(mp_result)  # type: ignore
        return HandDetectionResult(hands, w, h, elapsed)

    def detect_on_rgb(self, rgb_image: np.ndarray) -> HandDetectionResult:
        """Run detection on an already-RGB image (avoids redundant conversion)."""
        if rgb_image is None or rgb_image.size == 0:
            return HandDetectionResult([], 0, 0, 0.0)

        h, w = rgb_image.shape[:2]
        if not rgb_image.flags.c_contiguous:
            rgb_image = np.ascontiguousarray(rgb_image)

        start = time.perf_counter()
        mp_result = self._pipeline.process(rgb_image)  # type: ignore
        elapsed = (time.perf_counter() - start) * 1000.0

        hands = self._extract_hands(mp_result)  # type: ignore
        return HandDetectionResult(hands, w, h, elapsed)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self._pipeline, "close"):
            self._pipeline.close()  # type: ignore

    @property
    def config(self) -> HandDetectorConfig:
        return self._config

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    # ── Private ──────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        if self._config.max_num_hands < 1 or self._config.max_num_hands > 4:
            raise ValueError(f"max_num_hands must be 1-4, got {self._config.max_num_hands}")
        if self._config.model_complexity not in (0, 1):
            raise ValueError(f"model_complexity must be 0 (lite) or 1 (full), got {self._config.model_complexity}")
        if not (0.0 < self._config.min_detection_confidence <= 1.0):
            raise ValueError(f"min_detection_confidence must be in (0, 1]")
        if not (0.0 < self._config.min_tracking_confidence <= 1.0):
            raise ValueError(f"min_tracking_confidence must be in (0, 1]")

    def _initialize_mediapipe(self) -> None:
        try:
            import mediapipe as mp
        except ImportError:
            msg = (
                "MediaPipe is not installed. Install it with: pip install mediapipe"
            )
            raise RuntimeError(msg) from None

        try:
            self._mp_hands = mp.solutions.hands
        except AttributeError:
            from src.infrastructure.hand_detection.mediapipe_compatibility import patch_mediapipe
            patch_mediapipe()
            self._mp_hands = mp.solutions.hands
        self._pipeline = self._mp_hands.Hands(
            static_image_mode=self._config.static_image_mode,
            max_num_hands=self._config.max_num_hands,
            model_complexity=self._config.model_complexity,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )

    def _extract_hands(self, mp_result) -> list[Hand]:
        """Convert raw MediaPipe result to domain Hand objects."""
        hands: list[Hand] = []

        if not mp_result or not mp_result.multi_hand_landmarks:
            return hands

        handedness_list = mp_result.multi_handedness or [None] * len(mp_result.multi_hand_landmarks)

        for landmarks, handedness in zip(mp_result.multi_hand_landmarks, handedness_list):
            hand_landmarks = [
                HandLandmark(
                    max(0.0, min(1.0, float(lm.x))),
                    max(0.0, min(1.0, float(lm.y))),
                    float(lm.z)
                )
                for lm in landmarks.landmark
            ]

            if handedness and handedness.classification:
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                handedness_enum = Handedness.LEFT if label == "Left" else Handedness.RIGHT
            else:
                handedness_enum = Handedness.UNKNOWN
                score = 0.0

            hands.append(Hand(
                landmarks=hand_landmarks,
                handedness=handedness_enum,
                confidence=score,
            ))

        return hands

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, *args) -> None:
        self.close()
