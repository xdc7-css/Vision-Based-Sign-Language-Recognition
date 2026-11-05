"""MotionTracker — real-time position, velocity, acceleration, and trajectory tracking."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.domain.hand import Hand, HandLandmarkType
from src.domain.motion import (
    Acceleration2D,
    HandMotionTrack,
    MotionFeatures,
    MotionState,
    Position2D,
    TrajectorySegment,
    Velocity2D,
)


@dataclass
class TrackerConfig:
    window_size: int = 60
    velocity_samples: int = 5
    pause_speed_threshold: float = 0.005
    segment_min_frames: int = 10
    max_hands: int = 2


class MotionTracker:
    """Tracks motion for up to two hands over a sliding time window.

    For each hand, computes:
    - Position (palm center)
    - Velocity (first derivative)
    - Acceleration (second derivative)
    - Trajectory segments (start → pause → end)
    - Aggregated motion features
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self._config = config or TrackerConfig()
        self._tracks: dict[str, HandMotionTrack] = {}
        self._last_positions: dict[str, deque[Position2D]] = {}
        self._frame_count: int = 0

    # ── Public API ──────────────────────────────────────────────────

    def update(self, hands: list[Hand], timestamp: Optional[float] = None) -> dict[str, HandMotionTrack]:
        """Feed detected hands and update motion tracking.

        Args:
            hands: Detected hands from HandDetector.
            timestamp: Override timestamp (uses time.time() if None).

        Returns:
            Dict of hand_id → HandMotionTrack for all tracked hands.
        """
        ts = timestamp if timestamp is not None else time.time()
        self._frame_count += 1
        active_ids: set[str] = set()

        for hand in hands:
            hand_id = f"hand_{hand.handedness.value}"
            palm = self._palm_center(hand)
            active_ids.add(hand_id)

            if hand_id not in self._tracks:
                self._tracks[hand_id] = HandMotionTrack(
                    hand_id=hand_id,
                    handedness=hand.handedness.value,
                )
                self._last_positions[hand_id] = deque(maxlen=self._config.velocity_samples)

            self._update_hand(hand, hand_id, palm, ts)

        return dict(self._tracks)

    def get_track(self, hand_id: str) -> Optional[HandMotionTrack]:
        return self._tracks.get(hand_id)

    @property
    def all_tracks(self) -> list[HandMotionTrack]:
        return list(self._tracks.values())

    @property
    def active_tracks(self) -> list[HandMotionTrack]:
        return [t for t in self._tracks.values() if len(t.states) > 1]

    def reset(self, hand_id: Optional[str] = None) -> None:
        if hand_id:
            self._tracks.pop(hand_id, None)
            self._last_positions.pop(hand_id, None)
        else:
            self._tracks.clear()
            self._last_positions.clear()
        self._frame_count = 0

    def extract_features(self, hand_id: Optional[str] = None) -> MotionFeatures | None:
        """Extract aggregated features from tracking data."""
        if hand_id:
            track = self._tracks.get(hand_id)
        else:
            tracks = self.active_tracks
            track = tracks[0] if tracks else None

        if track is None or not track.states:
            return None

        speeds = [s.velocity.speed for s in track.states]
        directions = [s.velocity.direction for s in track.states if s.velocity.speed > 0.001]
        pause_frames = sum(1 for s in track.states if s.velocity.speed < self._config.pause_speed_threshold)

        avg_dir = sum(directions) / len(directions) if directions else 0.0
        dir_var = (
            sum((d - avg_dir) ** 2 for d in directions) / len(directions)
            if directions
            else 0.0
        )

        return MotionFeatures(
            avg_speed=track.average_speed,
            peak_speed=track.peak_speed,
            total_distance=track.total_distance,
            net_displacement=track.net_displacement,
            motion_energy=track.motion_energy,
            duration=track.duration,
            num_segments=len(track.segments),
            dominant_direction=avg_dir,
            direction_variance=dir_var,
            pause_ratio=pause_frames / len(track.states) if track.states else 0.0,
            has_significant_motion=track.is_motion_significant,
            trajectory_complexity=track.total_distance / max(track.net_displacement, 0.001),
            hand_used=track.handedness,
        )

    # ── Private ─────────────────────────────────────────────────────

    def _palm_center(self, hand: Hand) -> Position2D:
        """Compute palm center from wrist + all MCP landmarks."""
        indices = [
            HandLandmarkType.WRIST,
            HandLandmarkType.THUMB_MCP,
            HandLandmarkType.INDEX_FINGER_MCP,
            HandLandmarkType.MIDDLE_FINGER_MCP,
            HandLandmarkType.RING_FINGER_MCP,
            HandLandmarkType.PINKY_MCP,
        ]
        xs = [hand[i].x for i in indices]
        ys = [hand[i].y for i in indices]
        return Position2D(sum(xs) / len(xs), sum(ys) / len(ys))

    def _update_hand(self, hand: Hand, hand_id: str, palm: Position2D, ts: float) -> None:
        track = self._tracks[hand_id]
        pos_history = self._last_positions[hand_id]
        pos_history.append(palm)
        track.palm_centers.append(palm)

        # Store raw landmarks for temporal sequences
        track.landmark_sequences.append(
            [(lm.x, lm.y, lm.z) for lm in hand.landmarks]
        )

        # Compute velocity from position history
        if len(pos_history) >= 2:
            vx = palm.x - pos_history[-2].x
            vy = palm.y - pos_history[-2].y
            velocity = Velocity2D(vx, vy)

            acc = Acceleration2D(0.0, 0.0)
            if len(pos_history) >= 3:
                prev_vx = pos_history[-2].x - pos_history[-3].x
                prev_vy = pos_history[-2].y - pos_history[-3].y
                acc = Acceleration2D(vx - prev_vx, vy - prev_vy)
        else:
            velocity = Velocity2D(0.0, 0.0)
            acc = Acceleration2D(0.0, 0.0)

        state = MotionState(
            position=palm,
            velocity=velocity,
            acceleration=acc,
            timestamp=ts,
            frame_id=self._frame_count,
            confidence=hand.confidence,
        )
        track.states.append(state)

        # Detect trajectory segments
        self._detect_segment(track, state, velocity)

        # Maintain window size
        if len(track.states) > self._config.window_size:
            trim = len(track.states) - self._config.window_size
            track.states = track.states[trim:]
            track.palm_centers = track.palm_centers[trim:]
            track.landmark_sequences = track.landmark_sequences[trim:]

    def _detect_segment(
        self,
        track: HandMotionTrack,
        state: MotionState,
        velocity: Velocity2D,
    ) -> None:
        """Segment trajectory by detecting pause → motion → pause boundaries."""
        is_paused = velocity.speed < self._config.pause_speed_threshold

        if not track.segments:
            track.segments.append(TrajectorySegment(
                start_time=state.timestamp,
                end_time=state.timestamp,
                start_position=state.position,
                end_position=state.position,
                peak_speed=0.0,
                avg_speed=0.0,
                total_displacement=0.0,
                total_distance=0.0,
                dominant_direction=0.0,
                states=[state],
            ))
            return

        current = track.segments[-1]

        if is_paused and len(current.states) >= self._config.segment_min_frames:
            # End current segment
            self._finalize_segment(current)
            # Start new segment
            track.segments.append(TrajectorySegment(
                start_time=state.timestamp,
                end_time=state.timestamp,
                start_position=state.position,
                end_position=state.position,
                peak_speed=0.0,
                avg_speed=0.0,
                total_displacement=0.0,
                total_distance=0.0,
                dominant_direction=0.0,
                states=[state],
            ))
        else:
            # Continue segment
            current.end_time = state.timestamp
            current.end_position = state.position
            current.states.append(state)
            current.peak_speed = max(current.peak_speed, velocity.speed)

    def _finalize_segment(self, segment: TrajectorySegment) -> None:
        if len(segment.states) < 2:
            return
        speeds = [s.velocity.speed for s in segment.states]
        directions = [s.velocity.direction for s in segment.states if s.velocity.speed > 0.001]

        segment.avg_speed = sum(speeds) / len(speeds)
        segment.total_displacement = segment.start_position.distance_to(segment.end_position)
        segment.total_distance = sum(
            segment.states[i].position.distance_to(segment.states[i + 1].position)
            for i in range(len(segment.states) - 1)
        )
        segment.dominant_direction = (
            sum(directions) / len(directions) if directions else 0.0
        )
