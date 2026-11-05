"""Tests for hand detection engine — models, detector, renderer, utils."""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.hand import (
    HAND_CONNECTIONS,
    Hand,
    HandDetectionResult,
    HandLandmark,
    HandLandmarkType,
    Handedness,
)
from src.infrastructure.hand_detection.renderer import HandRenderer, RenderConfig
from src.infrastructure.hand_detection.utils import LandmarkProcessor


# ── Domain model tests ────────────────────────────────────────────────


class TestHandLandmark:
    def test_creation(self) -> None:
        lm = HandLandmark(0.5, 0.3, -0.1)
        assert lm.x == 0.5
        assert lm.y == 0.3
        assert lm.z == -0.1

    def test_invalid_x_raises(self) -> None:
        with pytest.raises(ValueError):
            HandLandmark(1.5, 0.5, 0.0)

    def test_invalid_y_raises(self) -> None:
        with pytest.raises(ValueError):
            HandLandmark(0.5, -0.1, 0.0)

    def test_to_pixel(self) -> None:
        lm = HandLandmark(0.5, 0.25, 0.0)
        x, y = lm.to_pixel(640, 480)
        assert x == 320
        assert y == 120


class TestHand:
    @pytest.fixture
    def hand(self) -> Hand:
        landmarks = [HandLandmark(0.5, 0.5, 0.0) for _ in range(21)]
        return Hand(landmarks=landmarks, handedness=Handedness.RIGHT, confidence=0.95)

    def test_creation(self, hand: Hand) -> None:
        assert hand.handedness == Handedness.RIGHT
        assert hand.confidence == 0.95
        assert len(hand.landmarks) == 21

    def test_wrong_landmark_count_raises(self) -> None:
        with pytest.raises(ValueError):
            Hand(landmarks=[HandLandmark(0.5, 0.5, 0.0) for _ in range(5)],
                 handedness=Handedness.LEFT, confidence=0.8)

    def test_getitem(self, hand: Hand) -> None:
        wrist = hand[HandLandmarkType.WRIST]
        assert wrist.x == 0.5

    def test_iteration(self, hand: Hand) -> None:
        count = sum(1 for _ in hand)
        assert count == 21

    def test_landmark_dict(self, hand: Hand) -> None:
        d = hand.landmark_dict
        assert len(d) == 21
        assert HandLandmarkType.WRIST in d


class TestHandDetectionResult:
    def test_empty(self) -> None:
        result = HandDetectionResult([], 640, 480, 5.0)
        assert result.num_hands == 0
        assert not result.has_hands
        assert result.left_hand is None
        assert result.right_hand is None

    def test_with_hands(self) -> None:
        landmarks = [HandLandmark(0.5, 0.5, 0.0) for _ in range(21)]
        left = Hand(landmarks=landmarks, handedness=Handedness.LEFT, confidence=0.9)
        right = Hand(landmarks=landmarks, handedness=Handedness.RIGHT, confidence=0.95)
        result = HandDetectionResult([left, right], 640, 480, 8.0)

        assert result.num_hands == 2
        assert result.has_hands
        assert result.left_hand is not None
        assert result.left_hand.handedness == Handedness.LEFT
        assert result.right_hand is not None
        assert result.right_hand.handedness == Handedness.RIGHT
        assert result.processing_time_ms == 8.0


class TestHandLandmarkType:
    def test_all_values_present(self) -> None:
        assert len(HandLandmarkType) == 21

    def test_ordering(self) -> None:
        assert HandLandmarkType.WRIST.value == 0
        assert HandLandmarkType.THUMB_TIP.value == 4
        assert HandLandmarkType.INDEX_FINGER_TIP.value == 8
        assert HandLandmarkType.PINKY_TIP.value == 20


class TestHAND_CONNECTIONS:
    def test_has_connections(self) -> None:
        assert len(HAND_CONNECTIONS) > 0

    def test_all_types_valid(self) -> None:
        all_ids = {lt.value for lt in HandLandmarkType}
        for a, b in HAND_CONNECTIONS:
            assert a.value in all_ids
            assert b.value in all_ids


# ── Renderer tests ───────────────────────────────────────────────────


class TestHandRenderer:
    @pytest.fixture
    def renderer(self) -> HandRenderer:
        return HandRenderer()

    @pytest.fixture
    def hand(self) -> Hand:
        landmarks = [HandLandmark(0.5, 0.5, 0.0) for _ in range(21)]
        return Hand(landmarks=landmarks, handedness=Handedness.RIGHT, confidence=0.95)

    @pytest.fixture
    def image(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_with_no_hands(self, renderer: HandRenderer, image: np.ndarray) -> None:
        result = HandDetectionResult([], 640, 480, 0.0)
        output = renderer.draw(image, result)
        assert output.shape == image.shape
        assert np.array_equal(output, image)  # unchanged

    def test_draw_with_hand(self, renderer: HandRenderer, image: np.ndarray, hand: Hand) -> None:
        result = HandDetectionResult([hand], 640, 480, 5.0)
        output = renderer.draw(image, result)
        assert output.shape == image.shape
        # Image should be modified (landmarks drawn)
        assert not np.array_equal(output, image)

    def test_draw_landmarks(self, renderer: HandRenderer, image: np.ndarray, hand: Hand) -> None:
        output = renderer.draw_landmarks(image, hand)
        assert output.shape == image.shape

    def test_draw_skeleton(self, renderer: HandRenderer, image: np.ndarray, hand: Hand) -> None:
        output = renderer.draw_skeleton(image, hand)
        assert output.shape == image.shape

    def test_draw_label(self, renderer: HandRenderer, image: np.ndarray, hand: Hand) -> None:
        result = HandDetectionResult([hand], 640, 480, 0.0)
        output = renderer.draw_label(image, hand, result)
        assert output.shape == image.shape

    def test_draw_bounding_box(self, renderer: HandRenderer, image: np.ndarray, hand: Hand) -> None:
        output = renderer.draw_bounding_box(image, hand)
        assert output.shape == image.shape

    def test_left_hand_color(self) -> None:
        """Left hand should use a different color than right hand."""
        config = RenderConfig()
        renderer = HandRenderer(config)
        assert config.left_hand_color != config.right_hand_color


# ── LandmarkProcessor tests ──────────────────────────────────────────


class TestLandmarkProcessor:
    @pytest.fixture
    def landmarks(self) -> list[HandLandmark]:
        """Standard hand landmarks — wrist at (0.5, 0.5), fingers spread upward."""
        lms = [HandLandmark(0.5, 0.5, 0.0) for _ in range(21)]
        # Wrist
        lms[0] = HandLandmark(0.5, 0.6, 0.0)
        # Index finger MCP → TIP (extending upward)
        lms[5] = HandLandmark(0.45, 0.45, 0.0)
        lms[6] = HandLandmark(0.45, 0.35, 0.0)
        lms[7] = HandLandmark(0.45, 0.25, 0.0)
        lms[8] = HandLandmark(0.45, 0.15, 0.0)
        # Middle finger (extending upward)
        lms[9] = HandLandmark(0.50, 0.45, 0.0)
        lms[10] = HandLandmark(0.50, 0.33, 0.0)
        lms[11] = HandLandmark(0.50, 0.22, 0.0)
        lms[12] = HandLandmark(0.50, 0.12, 0.0)
        # Ring finger (curled — TIP pulled toward palm, closer to MCP than PIP)
        lms[13] = HandLandmark(0.55, 0.45, 0.0)
        lms[14] = HandLandmark(0.55, 0.52, 0.0)
        lms[15] = HandLandmark(0.55, 0.49, 0.0)
        lms[16] = HandLandmark(0.55, 0.48, 0.0)
        return lms

    @pytest.fixture
    def hand(self, landmarks: list[HandLandmark]) -> Hand:
        return Hand(landmarks=landmarks, handedness=Handedness.RIGHT, confidence=0.95)

    def test_to_array_shape(self, landmarks: list[HandLandmark]) -> None:
        arr = LandmarkProcessor.to_array(landmarks)
        assert arr.shape == (21, 3)

    def test_to_flat_array_length(self, landmarks: list[HandLandmark]) -> None:
        flat = LandmarkProcessor.to_flat_array(landmarks)
        assert len(flat) == 63

    def test_normalize(self, landmarks: list[HandLandmark]) -> None:
        normalized = LandmarkProcessor.normalize(landmarks)
        assert normalized.shape == (21, 3)
        # Wrist should be at origin after normalization
        assert abs(normalized[0][0]) < 1e-6
        assert abs(normalized[0][1]) < 1e-6

    def test_bounding_box(self, landmarks: list[HandLandmark]) -> None:
        x_min, y_min, x_max, y_max = LandmarkProcessor.bounding_box(landmarks, 640, 480)
        assert 0 <= x_min < x_max <= 640
        assert 0 <= y_min < y_max <= 480

    def test_bounding_box_normalized(self, landmarks: list[HandLandmark]) -> None:
        x_min, y_min, x_max, y_max = LandmarkProcessor.bounding_box_normalized(landmarks)
        assert 0.0 <= x_min < x_max <= 1.0
        assert 0.0 <= y_min < y_max <= 1.0

    def test_distance(self, landmarks: list[HandLandmark]) -> None:
        d = LandmarkProcessor.distance(landmarks[5], landmarks[8])
        assert d > 0

    def test_distance_between_same(self) -> None:
        a = HandLandmark(0.5, 0.5, 0.0)
        assert LandmarkProcessor.distance(a, a) == 0.0

    def test_angle(self) -> None:
        a = HandLandmark(0.0, 0.0, 0.0)
        b = HandLandmark(0.0, 1.0, 0.0)
        c = HandLandmark(1.0, 0.0, 0.0)
        angle = LandmarkProcessor.angle(a, b, c)
        assert abs(angle - 45.0) < 1.0  # 45 degrees

    def test_center(self, landmarks: list[HandLandmark]) -> None:
        c = LandmarkProcessor.center(landmarks)
        assert 0.0 < c.x < 1.0
        assert 0.0 < c.y < 1.0

    def test_palm_center(self, hand: Hand) -> None:
        pc = LandmarkProcessor.palm_center(hand)
        assert 0.0 < pc.x < 1.0
        assert 0.0 < pc.y < 1.0

    def test_finger_states(self, hand: Hand) -> None:
        states = LandmarkProcessor.finger_states(hand)
        assert set(states.keys()) == {"thumb", "index", "middle", "ring", "pinky"}
        # Index and middle should be extended (upward TIP above PIP)
        assert states["index"] is True
        assert states["middle"] is True
        # Ring should be curled (TIP below PIP)
        assert states["ring"] is False

    def test_num_extended_fingers(self, hand: Hand) -> None:
        n = LandmarkProcessor.num_extended_fingers(hand)
        assert 0 <= n <= 5

    def test_extract_features(self, hand: Hand) -> None:
        features = LandmarkProcessor.extract_features(hand)
        assert "landmarks_normalized" in features
        assert "landmarks_flat" in features
        assert "finger_states" in features
        assert "num_extended" in features
        assert "palm_center_x" in features
        assert "palm_center_y" in features
        assert "handedness" in features
        assert features["handedness"] == "Right"
        assert features["num_extended"] >= 2

    def test_hand_orientation(self, hand: Hand) -> None:
        orientation = LandmarkProcessor.hand_orientation(hand)
        assert 0.0 <= orientation <= 180.0
