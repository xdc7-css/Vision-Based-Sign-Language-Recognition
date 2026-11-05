"""Hand detection domain models — landmark types, hand entities, skeleton topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Iterator


class Handedness(Enum):
    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"


class HandLandmarkType(IntEnum):
    """The 21 MediaPipe hand landmarks in canonical order."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass(frozen=True)
class HandLandmark:
    """A single 3D hand landmark with normalized coordinates."""
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0):
            raise ValueError(f"Landmark x must be in [0, 1], got {self.x}")
        if not (0.0 <= self.y <= 1.0):
            raise ValueError(f"Landmark y must be in [0, 1], got {self.y}")

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        return (int(self.x * width), int(self.y * height))


# ── Skeleton topology ──────────────────────────────────────────────────

HAND_CONNECTIONS: list[tuple[HandLandmarkType, HandLandmarkType]] = [
    # Thumb
    (HandLandmarkType.WRIST, HandLandmarkType.THUMB_CMC),
    (HandLandmarkType.THUMB_CMC, HandLandmarkType.THUMB_MCP),
    (HandLandmarkType.THUMB_MCP, HandLandmarkType.THUMB_IP),
    (HandLandmarkType.THUMB_IP, HandLandmarkType.THUMB_TIP),
    # Index finger
    (HandLandmarkType.WRIST, HandLandmarkType.INDEX_FINGER_MCP),
    (HandLandmarkType.INDEX_FINGER_MCP, HandLandmarkType.INDEX_FINGER_PIP),
    (HandLandmarkType.INDEX_FINGER_PIP, HandLandmarkType.INDEX_FINGER_DIP),
    (HandLandmarkType.INDEX_FINGER_DIP, HandLandmarkType.INDEX_FINGER_TIP),
    # Middle finger
    (HandLandmarkType.WRIST, HandLandmarkType.MIDDLE_FINGER_MCP),
    (HandLandmarkType.MIDDLE_FINGER_MCP, HandLandmarkType.MIDDLE_FINGER_PIP),
    (HandLandmarkType.MIDDLE_FINGER_PIP, HandLandmarkType.MIDDLE_FINGER_DIP),
    (HandLandmarkType.MIDDLE_FINGER_DIP, HandLandmarkType.MIDDLE_FINGER_TIP),
    # Ring finger
    (HandLandmarkType.WRIST, HandLandmarkType.RING_FINGER_MCP),
    (HandLandmarkType.RING_FINGER_MCP, HandLandmarkType.RING_FINGER_PIP),
    (HandLandmarkType.RING_FINGER_PIP, HandLandmarkType.RING_FINGER_DIP),
    (HandLandmarkType.RING_FINGER_DIP, HandLandmarkType.RING_FINGER_TIP),
    # Pinky
    (HandLandmarkType.WRIST, HandLandmarkType.PINKY_MCP),
    (HandLandmarkType.PINKY_MCP, HandLandmarkType.PINKY_PIP),
    (HandLandmarkType.PINKY_PIP, HandLandmarkType.PINKY_DIP),
    (HandLandmarkType.PINKY_DIP, HandLandmarkType.PINKY_TIP),
    # Palm base connections
    (HandLandmarkType.THUMB_MCP, HandLandmarkType.INDEX_FINGER_MCP),
    (HandLandmarkType.INDEX_FINGER_MCP, HandLandmarkType.MIDDLE_FINGER_MCP),
    (HandLandmarkType.MIDDLE_FINGER_MCP, HandLandmarkType.RING_FINGER_MCP),
    (HandLandmarkType.RING_FINGER_MCP, HandLandmarkType.PINKY_MCP),
]

FINGER_LANDMARKS: dict[str, list[HandLandmarkType]] = {
    "thumb": [HandLandmarkType.THUMB_MCP, HandLandmarkType.THUMB_IP, HandLandmarkType.THUMB_TIP],
    "index": [HandLandmarkType.INDEX_FINGER_MCP, HandLandmarkType.INDEX_FINGER_PIP,
              HandLandmarkType.INDEX_FINGER_DIP, HandLandmarkType.INDEX_FINGER_TIP],
    "middle": [HandLandmarkType.MIDDLE_FINGER_MCP, HandLandmarkType.MIDDLE_FINGER_PIP,
               HandLandmarkType.MIDDLE_FINGER_DIP, HandLandmarkType.MIDDLE_FINGER_TIP],
    "ring": [HandLandmarkType.RING_FINGER_MCP, HandLandmarkType.RING_FINGER_PIP,
             HandLandmarkType.RING_FINGER_DIP, HandLandmarkType.RING_FINGER_TIP],
    "pinky": [HandLandmarkType.PINKY_MCP, HandLandmarkType.PINKY_PIP,
              HandLandmarkType.PINKY_DIP, HandLandmarkType.PINKY_TIP],
}


# ── Hand entity ────────────────────────────────────────────────────────


@dataclass
class Hand:
    """A single detected hand with landmarks, handedness, and confidence."""
    landmarks: list[HandLandmark]
    handedness: Handedness = Handedness.UNKNOWN
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if len(self.landmarks) != 21:
            raise ValueError(f"Hand requires exactly 21 landmarks, got {len(self.landmarks)}")

    @property
    def landmark_dict(self) -> dict[HandLandmarkType, HandLandmark]:
        return {HandLandmarkType(i): self.landmarks[i] for i in range(21)}

    def __getitem__(self, landmark_type: HandLandmarkType) -> HandLandmark:
        return self.landmarks[landmark_type.value]

    def __iter__(self) -> Iterator[HandLandmark]:
        return iter(self.landmarks)


# ── Detection result ───────────────────────────────────────────────────


@dataclass
class HandDetectionResult:
    """Complete result of a hand detection pass on one image."""
    hands: list[Hand]
    image_width: int
    image_height: int
    processing_time_ms: float

    @property
    def num_hands(self) -> int:
        return len(self.hands)

    @property
    def has_hands(self) -> bool:
        return len(self.hands) > 0

    @property
    def left_hand(self) -> Hand | None:
        for hand in self.hands:
            if hand.handedness == Handedness.LEFT:
                return hand
        return None

    @property
    def right_hand(self) -> Hand | None:
        for hand in self.hands:
            if hand.handedness == Handedness.RIGHT:
                return hand
        return None
