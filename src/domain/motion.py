"""Motion tracking domain models — position, velocity, trajectory, temporal sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Position2D:
    x: float
    y: float

    def distance_to(self, other: Position2D) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def angle_to(self, other: Position2D) -> float:
        """Angle in degrees from self to other."""
        return math.degrees(math.atan2(other.y - self.y, other.x - self.x))


@dataclass(frozen=True)
class Velocity2D:
    vx: float
    vy: float

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy)

    @property
    def direction(self) -> float:
        """Direction angle in degrees (0° = right, 90° = down)."""
        return math.degrees(math.atan2(self.vy, self.vx))


@dataclass(frozen=True)
class Acceleration2D:
    ax: float
    ay: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.ax * self.ax + self.ay * self.ay)


@dataclass
class MotionState:
    """Complete motion state at a single timestep."""
    position: Position2D
    velocity: Velocity2D
    acceleration: Acceleration2D
    timestamp: float
    frame_id: int
    confidence: float = 1.0


@dataclass
class TrajectorySegment:
    """A continuous segment of a motion trajectory.

    A segment begins when significant motion starts and ends
    when the hand pauses or changes direction sharply.
    """
    start_time: float
    end_time: float
    start_position: Position2D
    end_position: Position2D
    peak_speed: float
    avg_speed: float
    total_displacement: float
    total_distance: float
    dominant_direction: float
    states: list[MotionState] = field(default_factory=list)


@dataclass
class HandMotionTrack:
    """Complete motion tracking data for one hand over a window of time."""
    hand_id: str
    handedness: str
    states: list[MotionState] = field(default_factory=list)
    segments: list[TrajectorySegment] = field(default_factory=list)
    palm_centers: list[Position2D] = field(default_factory=list)
    landmark_sequences: list[list] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if len(self.states) < 2:
            return 0.0
        return self.states[-1].timestamp - self.states[0].timestamp

    @property
    def average_speed(self) -> float:
        if not self.states:
            return 0.0
        return sum(s.velocity.speed for s in self.states) / len(self.states)

    @property
    def peak_speed(self) -> float:
        return max((s.velocity.speed for s in self.states), default=0.0)

    @property
    def total_distance(self) -> float:
        if len(self.palm_centers) < 2:
            return 0.0
        return sum(
            self.palm_centers[i].distance_to(self.palm_centers[i + 1])
            for i in range(len(self.palm_centers) - 1)
        )

    @property
    def net_displacement(self) -> float:
        if len(self.palm_centers) < 2:
            return 0.0
        return self.palm_centers[0].distance_to(self.palm_centers[-1])

    @property
    def motion_energy(self) -> float:
        """Sum of squared velocities (proxy for motion intensity)."""
        return sum(s.velocity.speed ** 2 for s in self.states)

    @property
    def is_motion_significant(self) -> bool:
        """True if the hand moved more than a threshold during this track."""
        return self.total_distance > 0.05  # 5% of normalized space


@dataclass
class MotionFeatures:
    """Aggregated motion features extracted from a HandMotionTrack."""
    avg_speed: float
    peak_speed: float
    total_distance: float
    net_displacement: float
    motion_energy: float
    duration: float
    num_segments: int
    dominant_direction: float
    direction_variance: float
    pause_ratio: float
    has_significant_motion: bool
    trajectory_complexity: float
    hand_used: str
