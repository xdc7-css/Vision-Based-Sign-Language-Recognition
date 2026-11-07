"""Multimodal Sign Language domain models — dual hands, face, pose, and feature schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.domain.hand import Hand


# ── Constants & Schema Version ─────────────────────────────────────────

MULTIMODAL_SCHEMA_VERSION = "v2_multimodal"
LEGACY_SCHEMA_VERSION = "v1_single_hand"

# Compact Face Landmark Indices (from MediaPipe 468/478 Face Mesh)
# Carefully selected 24 expressive landmarks:
# Eyebrows (6): outer, mid, inner for both sides
# Eyes (8): corners and top/bottom apertures for both sides
# Lips/Mouth (6): corners, upper/lower outer, upper/lower inner
# Anchor/Orientation (4): nose tip, chin, left cheek, right cheek
FACE_EXPRESSIVE_INDICES: tuple[int, ...] = (
    # Eyebrows (left: 70, 63, 105; right: 300, 293, 334)
    70, 63, 105, 300, 293, 334,
    # Eyes (left: 33, 133, 159, 145; right: 263, 362, 386, 374)
    33, 133, 159, 145, 263, 362, 386, 374,
    # Mouth (corners: 61, 291; lips: 0, 17, 13, 14)
    61, 291, 0, 17, 13, 14,
    # Face Anchors: nose tip (1), chin (152), left cheek (234), right cheek (454)
    1, 152, 234, 454,
)

# Upper-body pose indices (from MediaPipe 33 Pose landmarks)
# 8 keypoints: nose (0), shoulders (11, 12), elbows (13, 14), wrists (15, 16), mid-chest/neck proxy
POSE_UPPER_BODY_INDICES: tuple[int, ...] = (
    0,   # Nose
    11,  # Left shoulder
    12,  # Right shoulder
    13,  # Left elbow
    14,  # Right elbow
    15,  # Left wrist
    16,  # Right wrist
    23,  # Left hip (for torso angle/posture)
)

HAND_LANDMARKS_COUNT = 21
FACE_LANDMARKS_COUNT = len(FACE_EXPRESSIVE_INDICES)       # 24
POSE_LANDMARKS_COUNT = len(POSE_UPPER_BODY_INDICES)       # 8

# Feature dimensions
HAND_DIM = HAND_LANDMARKS_COUNT * 3                       # 63
DUAL_HAND_DIM = HAND_DIM * 2                              # 126
FACE_DIM = FACE_LANDMARKS_COUNT * 3                       # 72
POSE_DIM = POSE_LANDMARKS_COUNT * 3                       # 24
QUALITY_METRICS_DIM = 8                                   # presence + geometric distances

FEATURE_DIM_HANDS = DUAL_HAND_DIM + QUALITY_METRICS_DIM                # 134
FEATURE_DIM_HANDS_FACE = DUAL_HAND_DIM + FACE_DIM + QUALITY_METRICS_DIM  # 206
FEATURE_DIM_FULL = DUAL_HAND_DIM + FACE_DIM + POSE_DIM + QUALITY_METRICS_DIM  # 230


class ModalityMode(Enum):
    """Supported multimodal recording and recognition modes."""
    ONE_HAND = "one_hand"
    TWO_HANDS = "two_hands"
    TWO_HANDS_FACE = "two_hands_face"
    TWO_HANDS_FACE_POSE = "two_hands_face_pose"


@dataclass
class ModalityConfig:
    """Runtime configuration for multimodal tracking."""
    mode: ModalityMode = ModalityMode.TWO_HANDS_FACE
    enable_left_hand: bool = True
    enable_right_hand: bool = True
    enable_face: bool = True
    enable_pose: bool = False
    mirror_camera: bool = True
    min_detection_confidence: float = 0.35
    min_tracking_confidence: float = 0.35


@dataclass(frozen=True)
class FaceLandmark:
    """Normalized 3D facial landmark coordinate."""
    x: float
    y: float
    z: float

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        return (int(np.clip(self.x * width, 0, width - 1)), int(np.clip(self.y * height, 0, height - 1)))


@dataclass(frozen=True)
class PoseLandmark:
    """Normalized 3D upper-body pose landmark coordinate with visibility."""
    x: float
    y: float
    z: float
    visibility: float = 1.0

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        return (int(np.clip(self.x * width, 0, width - 1)), int(np.clip(self.y * height, 0, height - 1)))


@dataclass
class MultimodalFrame:
    """Synchronized multimodal observation at a single instant."""
    left_hand: Optional[Hand] = None
    right_hand: Optional[Hand] = None
    face_landmarks: list[FaceLandmark] = field(default_factory=list)
    pose_landmarks: list[PoseLandmark] = field(default_factory=list)
    
    # Presence flags
    has_left_hand: bool = False
    has_right_hand: bool = False
    has_face: bool = False
    has_pose: bool = False

    # Geometric face & posture indicators
    mouth_openness: float = 0.0
    eyebrow_elevation: float = 0.0
    eye_openness_left: float = 0.0
    eye_openness_right: float = 0.0
    left_hand_to_face_dist: float = 0.0
    right_hand_to_face_dist: float = 0.0

    geometric_indicators: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    processing_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.left_hand is not None:
            self.has_left_hand = True
        if self.right_hand is not None:
            self.has_right_hand = True
        if self.face_landmarks:
            self.has_face = True
        if self.pose_landmarks:
            self.has_pose = True
        if not self.geometric_indicators:
            self.geometric_indicators = {
                "mouth_openness": self.mouth_openness,
                "eyebrow_elevation": self.eyebrow_elevation,
                "eye_openness_left": self.eye_openness_left,
                "eye_openness_right": self.eye_openness_right,
                "left_wrist_to_nose": self.left_hand_to_face_dist,
                "right_wrist_to_nose": self.right_hand_to_face_dist,
            }

    @property
    def has_left(self) -> bool:
        return self.has_left_hand or (self.left_hand is not None)

    @property
    def has_right(self) -> bool:
        return self.has_right_hand or (self.right_hand is not None)

    @property
    def num_hands(self) -> int:
        return self.total_hands_detected

    @property
    def total_hands_detected(self) -> int:
        return (1 if self.has_left else 0) + (1 if self.has_right else 0)


@dataclass
class MultimodalFeatureSchema:
    """Metadata detailing the feature serialization layout."""
    version: str = MULTIMODAL_SCHEMA_VERSION
    mode: str = ModalityMode.TWO_HANDS_FACE.value
    feature_dim: int = FEATURE_DIM_HANDS_FACE
    has_face: bool = True
    has_pose: bool = False
    left_hand_slice: tuple[int, int] = (0, 63)
    right_hand_slice: tuple[int, int] = (63, 126)
    face_slice: tuple[int, int] = (126, 198)
    pose_slice: Optional[tuple[int, int]] = None
    quality_slice: tuple[int, int] = (198, 206)

    @property
    def schema_version(self) -> str:
        return self.version

    @property
    def includes_face(self) -> bool:
        return self.has_face

    @property
    def includes_pose(self) -> bool:
        return self.has_pose

    @property
    def feature_names(self) -> list[str]:
        return [f"f_{i}" for i in range(self.feature_dim)]

    @classmethod
    def generate(cls, mode: ModalityMode = ModalityMode.TWO_HANDS_FACE) -> MultimodalFeatureSchema:
        dim = {
            ModalityMode.ONE_HAND: 63,
            ModalityMode.TWO_HANDS: FEATURE_DIM_HANDS,
            ModalityMode.TWO_HANDS_FACE: FEATURE_DIM_HANDS_FACE,
            ModalityMode.TWO_HANDS_FACE_POSE: FEATURE_DIM_FULL,
        }.get(mode, FEATURE_DIM_HANDS_FACE)
        has_face = mode in (ModalityMode.TWO_HANDS_FACE, ModalityMode.TWO_HANDS_FACE_POSE)
        has_pose = mode == ModalityMode.TWO_HANDS_FACE_POSE
        return cls(
            version=MULTIMODAL_SCHEMA_VERSION,
            mode=mode.value,
            feature_dim=dim,
            has_face=has_face,
            has_pose=has_pose,
        )
