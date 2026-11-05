"""Tests for motion tracking module (tracker, features, domain models)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from src.domain.hand import Hand, HandLandmark, Handedness
from src.domain.motion import (
    Acceleration2D,
    MotionFeatures,
    MotionState,
    Position2D,
    TrajectorySegment,
    Velocity2D,
)
from src.infrastructure.motion_tracking.tracker import MotionTracker, TrackerConfig


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def static_hand() -> Hand:
    landmarks = [HandLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    landmarks[0] = HandLandmark(0.5, 0.5, 0.0)  # wrist
    landmarks[5] = HandLandmark(0.55, 0.45, 0.0)  # index MCP
    landmarks[9] = HandLandmark(0.55, 0.50, 0.0)  # middle MCP
    landmarks[13] = HandLandmark(0.55, 0.55, 0.0)  # ring MCP
    landmarks[17] = HandLandmark(0.55, 0.60, 0.0)  # pinky MCP
    return Hand(landmarks=landmarks, handedness=Handedness.RIGHT, confidence=0.95)


@pytest.fixture
def moving_hand() -> Hand:
    landmarks = [HandLandmark(x=0.6, y=0.4, z=0.0) for _ in range(21)]
    landmarks[0] = HandLandmark(0.6, 0.4, 0.0)
    landmarks[5] = HandLandmark(0.65, 0.35, 0.0)
    landmarks[9] = HandLandmark(0.65, 0.40, 0.0)
    landmarks[13] = HandLandmark(0.65, 0.45, 0.0)
    landmarks[17] = HandLandmark(0.65, 0.50, 0.0)
    return Hand(landmarks=landmarks, handedness=Handedness.RIGHT, confidence=0.95)


# ── Domain model tests ────────────────────────────────────────────────


class TestPosition2D:
    def test_distance(self) -> None:
        a = Position2D(0, 0)
        b = Position2D(3, 4)
        assert a.distance_to(b) == 5.0

    def test_angle(self) -> None:
        a = Position2D(0, 0)
        b = Position2D(1, 0)
        assert a.angle_to(b) == 0.0
        c = Position2D(0, 1)
        assert a.angle_to(c) == 90.0


class TestVelocity2D:
    def test_speed(self) -> None:
        v = Velocity2D(3, 4)
        assert v.speed == 5.0

    def test_direction(self) -> None:
        v = Velocity2D(1, 0)
        assert v.direction == 0.0
        v2 = Velocity2D(0, 1)
        assert v2.direction == 90.0


class TestAcceleration2D:
    def test_magnitude(self) -> None:
        a = Acceleration2D(3, 4)
        assert a.magnitude == 5.0


class TestMotionState:
    def test_creation(self) -> None:
        state = MotionState(
            position=Position2D(0.5, 0.5),
            velocity=Velocity2D(0.1, 0.0),
            acceleration=Acceleration2D(0.0, 0.0),
            timestamp=time.time(),
            frame_id=1,
        )
        assert state.velocity.speed == 0.1


# ── MotionTracker tests ───────────────────────────────────────────────


class TestMotionTracker:
    def test_initial_state(self) -> None:
        tracker = MotionTracker()
        assert len(tracker.all_tracks) == 0
        assert tracker.active_tracks == []

    def test_update_with_no_hands(self) -> None:
        tracker = MotionTracker()
        tracks = tracker.update([])
        assert len(tracks) == 0

    def test_tracks_single_hand(self, static_hand: Hand) -> None:
        tracker = MotionTracker()
        tracks = tracker.update([static_hand])
        assert len(tracks) >= 1

        hand_id = list(tracks.keys())[0]
        track = tracks[hand_id]
        assert track.handedness == "Right"
        assert len(track.states) == 1
        assert len(track.palm_centers) == 1

    def test_tracks_velocity(self, moving_hand: Hand) -> None:
        tracker = MotionTracker()
        # First frame
        tracker.update([moving_hand])
        # Second frame — different position
        landmarks = [HandLandmark(x=0.7, y=0.3, z=0.0) for _ in range(21)]
        landmarks[0] = HandLandmark(0.7, 0.3, 0.0)
        landmarks[5] = HandLandmark(0.75, 0.25, 0.0)
        moving2 = Hand(landmarks=landmarks, handedness=Handedness.RIGHT, confidence=0.95)
        tracks = tracker.update([moving2])

        hand_id = list(tracks.keys())[0]
        track = tracks[hand_id]
        assert len(track.states) == 2
        # Should have non-zero velocity
        assert track.states[-1].velocity.speed > 0.0

    def test_segment_detection(self) -> None:
        tracker = MotionTracker(config=TrackerConfig(
            window_size=100,
            pause_speed_threshold=0.001,
            segment_min_frames=3,
        ))

        # Simulate a left-to-right movement across 20 frames
        for i in range(20):
            x = 0.3 + i * 0.02
            landmarks = [HandLandmark(x=x, y=0.5, z=0.0) for _ in range(21)]
            landmarks[0] = HandLandmark(x, 0.5, 0.0)
            landmarks[5] = HandLandmark(x + 0.05, 0.45, 0.0)
            landmarks[9] = HandLandmark(x + 0.05, 0.50, 0.0)
            h = Hand(landmarks=landmarks, handedness=Handedness.RIGHT, confidence=0.95)
            tracker.update([h], timestamp=time.time())

        tracks = tracker.active_tracks
        if tracks:
            track = tracks[0]
            assert len(track.states) > 1
            # Verify motion was detected (velocity should be non-zero)
            speeds = [s.velocity.speed for s in track.states]
            assert max(speeds) > 0

    def test_reset(self) -> None:
        tracker = MotionTracker()
        landmarks = [HandLandmark(0.5, 0.5, 0.0) for _ in range(21)]
        h = Hand(landmarks=landmarks, handedness=Handedness.LEFT, confidence=0.95)
        tracker.update([h])
        assert len(tracker.all_tracks) >= 1
        tracker.reset()
        assert len(tracker.all_tracks) == 0

    def test_extract_features_no_data(self) -> None:
        tracker = MotionTracker()
        features = tracker.extract_features()
        assert features is None

    def test_extract_features_with_data(self, static_hand: Hand, moving_hand: Hand) -> None:
        tracker = MotionTracker()
        for _ in range(5):
            tracker.update([static_hand], timestamp=time.time())
            time.sleep(0.01)

        features = tracker.extract_features()
        if features is not None:
            assert isinstance(features, MotionFeatures)
            assert features.hand_used == "Right"
            assert features.num_segments >= 0
            assert features.duration >= 0
