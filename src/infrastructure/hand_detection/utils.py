"""Landmark processing utilities — geometry, normalization, finger state detection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.domain.hand import (
    FINGER_LANDMARKS,
    HAND_CONNECTIONS,
    Hand,
    HandLandmark,
    HandLandmarkType,
    Handedness,
)


class LandmarkProcessor:
    """Static utility class for landmark geometry and feature extraction.

    All methods accept domain HandLandmark objects and return
    numpy arrays or Python primitives.
    """

    # ── Conversion ──────────────────────────────────────────────────

    @staticmethod
    def to_array(landmarks: list[HandLandmark]) -> np.ndarray:
        """Convert a list of 21 HandLandmarks to an (N, 3) float32 array."""
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    @staticmethod
    def to_flat_array(landmarks: list[HandLandmark]) -> np.ndarray:
        """Convert 21 landmarks to a flat 63-element array (for ML input)."""
        return LandmarkProcessor.to_array(landmarks).flatten()

    # ── Normalization ───────────────────────────────────────────────

    @staticmethod
    def normalize(landmarks: list[HandLandmark]) -> np.ndarray:
        """Normalize landmarks relative to the wrist (landmark 0).

        Returns an (N, 3) array where each landmark is shifted so that
        the wrist is at (0, 0, 0) and scaled by the wrist-to-middle-MCP
        distance.
        """
        arr = LandmarkProcessor.to_array(landmarks)
        wrist = arr[HandLandmarkType.WRIST]

        # Center at wrist
        centered = arr - wrist

        # Scale by wrist-to-middle_mcp distance
        middle_mcp = arr[HandLandmarkType.MIDDLE_FINGER_MCP]
        scale = np.linalg.norm(middle_mcp - wrist)
        if scale > 0:
            centered = centered / scale

        return centered

    @staticmethod
    def normalize_to_flat(landmarks: list[HandLandmark]) -> np.ndarray:
        """Normalize and flatten to a 63-element feature vector."""
        return LandmarkProcessor.normalize(landmarks).flatten()

    @staticmethod
    def normalize_array(arr: np.ndarray) -> np.ndarray:
        """Normalize a numpy array of landmarks relative to the wrist (landmark 0).
        
        Accepts:
            (21, 3) -> returns (21, 3) normalized array
            (63,)   -> returns (63,) normalized flat vector
            (N, 21, 3) -> returns (N, 21, 3) normalized sequence
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == 63:
            reshaped = arr.reshape(21, 3)
            return LandmarkProcessor.normalize_array(reshaped).flatten()
        elif arr.ndim == 2 and arr.shape == (21, 3):
            wrist = arr[0]
            centered = arr - wrist
            middle_mcp = arr[9]  # HandLandmarkType.MIDDLE_FINGER_MCP
            scale = float(np.linalg.norm(middle_mcp - wrist))
            if scale > 1e-6:
                centered = centered / scale
            return centered
        elif arr.ndim == 3 and arr.shape[1:] == (21, 3):
            out = np.empty_like(arr)
            for i in range(arr.shape[0]):
                out[i] = LandmarkProcessor.normalize_array(arr[i])
            return out
        return arr

    # ── Bounding box ────────────────────────────────────────────────

    @staticmethod
    def bounding_box(
        landmarks: list[HandLandmark],
        image_width: int,
        image_height: int,
        padding: float = 0.05,
    ) -> tuple[int, int, int, int]:
        """Compute pixel-space bounding box (x_min, y_min, x_max, y_max)."""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        pw = int(padding * image_width)
        ph = int(padding * image_height)

        x_min = max(0, int(min(xs) * image_width) - pw)
        y_min = max(0, int(min(ys) * image_height) - ph)
        x_max = min(image_width, int(max(xs) * image_width) + pw)
        y_max = min(image_height, int(max(ys) * image_height) + ph)

        return (x_min, y_min, x_max, y_max)

    @staticmethod
    def bounding_box_normalized(
        landmarks: list[HandLandmark],
        padding: float = 0.05,
    ) -> tuple[float, float, float, float]:
        """Normalized bounding box (x_min, y_min, x_max, y_max) in [0, 1]."""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x_min = max(0.0, min(xs) - padding)
        y_min = max(0.0, min(ys) - padding)
        x_max = min(1.0, max(xs) + padding)
        y_max = min(1.0, max(ys) + padding)

        return (x_min, y_min, x_max, y_max)

    # ── Finger state detection ──────────────────────────────────────

    @staticmethod
    def finger_states(hand: Hand) -> dict[str, bool]:
        """Detect which fingers are extended (True) vs curled (False).

        Returns a dict keyed by finger name: thumb, index, middle, ring, pinky.
        """
        return {
            "thumb": LandmarkProcessor._is_thumb_extended(hand),
            "index": LandmarkProcessor._is_finger_extended(
                hand, HandLandmarkType.INDEX_FINGER_MCP, HandLandmarkType.INDEX_FINGER_TIP,
            ),
            "middle": LandmarkProcessor._is_finger_extended(
                hand, HandLandmarkType.MIDDLE_FINGER_MCP, HandLandmarkType.MIDDLE_FINGER_TIP,
            ),
            "ring": LandmarkProcessor._is_finger_extended(
                hand, HandLandmarkType.RING_FINGER_MCP, HandLandmarkType.RING_FINGER_TIP,
            ),
            "pinky": LandmarkProcessor._is_finger_extended(
                hand, HandLandmarkType.PINKY_MCP, HandLandmarkType.PINKY_TIP,
            ),
        }

    @staticmethod
    def num_extended_fingers(hand: Hand) -> int:
        """Count how many fingers are extended (0-5)."""
        return sum(1 for v in LandmarkProcessor.finger_states(hand).values() if v)

    @staticmethod
    def _is_finger_extended(hand: Hand, mcp: HandLandmarkType, tip: HandLandmarkType) -> bool:
        """A finger is extended if tip-to-MCP distance exceeds PIP-to-MCP distance."""
        mcp_lm = hand[mcp]
        tip_lm = hand[tip]

        pip = {
            HandLandmarkType.INDEX_FINGER_MCP: HandLandmarkType.INDEX_FINGER_PIP,
            HandLandmarkType.MIDDLE_FINGER_MCP: HandLandmarkType.MIDDLE_FINGER_PIP,
            HandLandmarkType.RING_FINGER_MCP: HandLandmarkType.RING_FINGER_PIP,
            HandLandmarkType.PINKY_MCP: HandLandmarkType.PINKY_PIP,
        }.get(mcp)

        if pip is None:
            return False

        pip_lm = hand[pip]

        d_mcp_tip = LandmarkProcessor.distance(mcp_lm, tip_lm)
        d_mcp_pip = LandmarkProcessor.distance(mcp_lm, pip_lm)

        return d_mcp_tip > d_mcp_pip * 1.15

    @staticmethod
    def _is_thumb_extended(hand: Hand) -> bool:
        """Thumb is extended if TIP is farther from INDEX_MCP than IP is."""
        tip = hand[HandLandmarkType.THUMB_TIP]
        ip = hand[HandLandmarkType.THUMB_IP]
        index_mcp = hand[HandLandmarkType.INDEX_FINGER_MCP]

        d_index_tip = LandmarkProcessor.distance(index_mcp, tip)
        d_index_ip = LandmarkProcessor.distance(index_mcp, ip)

        return d_index_tip > d_index_ip * 1.1

    # ── Geometry ────────────────────────────────────────────────────

    @staticmethod
    def distance(a: HandLandmark, b: HandLandmark) -> float:
        """Euclidean distance between two landmarks in normalized space."""
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def distance_2d(a: HandLandmark, b: HandLandmark) -> float:
        """2D Euclidean distance (ignoring z)."""
        dx = a.x - b.x
        dy = a.y - b.y
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def angle(a: HandLandmark, b: HandLandmark, c: HandLandmark) -> float:
        """Angle ABC in degrees (vertex at b)."""
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        if mag_ba * mag_bc == 0:
            return 0.0

        cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def center(landmarks: list[HandLandmark]) -> HandLandmark:
        """Centroid of all landmarks."""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        zs = [lm.z for lm in landmarks]
        return HandLandmark(
            x=sum(xs) / len(xs),
            y=sum(ys) / len(ys),
            z=sum(zs) / len(zs),
        )

    @staticmethod
    def palm_center(hand: Hand) -> HandLandmark:
        """Center of the palm (average of wrist + MCPs)."""
        indices = [
            HandLandmarkType.WRIST,
            HandLandmarkType.THUMB_MCP,
            HandLandmarkType.INDEX_FINGER_MCP,
            HandLandmarkType.MIDDLE_FINGER_MCP,
            HandLandmarkType.RING_FINGER_MCP,
            HandLandmarkType.PINKY_MCP,
        ]
        return LandmarkProcessor.center([hand[i] for i in indices])

    @staticmethod
    def hand_orientation(hand: Hand) -> float:
        """Approximate hand orientation in degrees (angle of palm normal).

        0° = palm facing camera, 90° = side view, 180° = back of hand.
        """
        wrist = hand[HandLandmarkType.WRIST]
        index_mcp = hand[HandLandmarkType.INDEX_FINGER_MCP]
        pinky_mcp = hand[HandLandmarkType.PINKY_MCP]

        return LandmarkProcessor.angle(wrist, index_mcp, pinky_mcp)

    # ── Feature extraction for ML ───────────────────────────────────

    @staticmethod
    def extract_features(hand: Hand) -> dict[str, Any]:
        """Extract a feature dictionary suitable for ML classifiers.

        Includes: normalized landmarks, finger states, pairwise distances,
        palm center, bounding box.
        """
        landmarks_normalized = LandmarkProcessor.normalize(hand.landmarks)
        finger_st = LandmarkProcessor.finger_states(hand)

        # Pairwise distances between all finger tips and MCPs
        tips = [
            HandLandmarkType.THUMB_TIP,
            HandLandmarkType.INDEX_FINGER_TIP,
            HandLandmarkType.MIDDLE_FINGER_TIP,
            HandLandmarkType.RING_FINGER_TIP,
            HandLandmarkType.PINKY_TIP,
        ]
        pairwise: list[float] = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                pairwise.append(LandmarkProcessor.distance(hand[tips[i]], hand[tips[j]]))

        return {
            "landmarks_normalized": landmarks_normalized,
            "landmarks_flat": landmarks_normalized.flatten(),
            "finger_states": finger_st,
            "num_extended": sum(1 for v in finger_st.values() if v),
            "palm_center_x": LandmarkProcessor.palm_center(hand).x,
            "palm_center_y": LandmarkProcessor.palm_center(hand).y,
            "wrist_z": hand[HandLandmarkType.WRIST].z,
            "handedness": hand.handedness.value,
            "confidence": hand.confidence,
            "tip_distances": pairwise,
        }
