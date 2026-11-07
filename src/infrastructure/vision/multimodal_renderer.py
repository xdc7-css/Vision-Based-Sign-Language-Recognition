"""MultimodalRenderer — clean, high-performance visualization for hands, face, and pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.domain.hand import (
    HAND_CONNECTIONS,
    Hand,
)
from src.domain.multimodal import MultimodalFrame


# Selected facial connection lines for clean visualization without screen clutter
FACE_CONTOURS: list[tuple[int, int]] = [
    # Eyebrows
    (70, 63), (63, 105),
    (300, 293), (293, 334),
    # Left eye outline
    (33, 159), (159, 133), (133, 145), (145, 33),
    # Right eye outline
    (263, 386), (386, 362), (362, 374), (374, 263),
    # Mouth outline
    (61, 0), (0, 291), (291, 17), (17, 61),
    (61, 13), (13, 291), (291, 14), (14, 61),
    # Nose bridge
    (1, 152),
]

# Upper body pose connections (shoulders, elbows, wrists)
POSE_CONNECTIONS: list[tuple[int, int]] = [
    (11, 12),  # Left shoulder to Right shoulder
    (11, 13),  # Left shoulder to Left elbow
    (13, 15),  # Left elbow to Left wrist
    (12, 14),  # Right shoulder to Right elbow
    (14, 16),  # Right elbow to Right wrist
]


@dataclass
class MultimodalRenderConfig:
    """Styling and toggles for multimodal rendering."""
    show_hands: bool = True
    show_face: bool = True
    show_pose: bool = False
    show_labels: bool = True

    # Colors in BGR format
    left_hand_color: tuple[int, int, int] = (255, 200, 65)     # Cyan/Sky-blue (BGR: 255, 200, 65 -> RGB: 65, 200, 255)
    right_hand_color: tuple[int, int, int] = (65, 150, 255)    # Orange/Amber (BGR: 65, 150, 255 -> RGB: 255, 150, 65)
    face_color: tuple[int, int, int] = (180, 240, 100)         # Soft spring green/teal
    pose_color: tuple[int, int, int] = (240, 100, 200)         # Bright magenta/purple

    landmark_radius: int = 4
    face_landmark_radius: int = 2
    connection_thickness: int = 2
    pose_thickness: int = 3


class MultimodalRenderer:
    """Renders multimodal skeletons and landmarks onto video frames."""

    def __init__(self, config: Optional[MultimodalRenderConfig] = None) -> None:
        self.config = config or MultimodalRenderConfig()

    def render(
        self,
        frame_bgr: np.ndarray,
        mm_frame: MultimodalFrame,
        show_hands: Optional[bool] = None,
        show_face: Optional[bool] = None,
        show_pose: Optional[bool] = None,
        show_labels: Optional[bool] = None,
    ) -> np.ndarray:
        if show_hands is not None:
            self.config.show_hands = show_hands
        if show_face is not None:
            self.config.show_face = show_face
        if show_pose is not None:
            self.config.show_pose = show_pose
        if show_labels is not None:
            self.config.show_labels = show_labels
        return self.draw(frame_bgr, mm_frame)

    def draw(self, frame_bgr: np.ndarray, mm_frame: MultimodalFrame) -> np.ndarray:
        """Render active modalities onto a copy of frame_bgr."""
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        output = frame_bgr.copy()
        h, w = output.shape[:2]

        # ── 1. Draw Upper-Body Pose ──────────────────────────────────
        if self.config.show_pose and mm_frame.has_pose:
            raw_pts = getattr(mm_frame, "_raw_pose_pts", None)
            if raw_pts:
                # Draw bones
                for idx1, idx2 in POSE_CONNECTIONS:
                    if idx1 < len(raw_pts) and idx2 < len(raw_pts):
                        p1 = raw_pts[idx1]
                        p2 = raw_pts[idx2]
                        if p1 != (0, 0) and p2 != (0, 0):
                            cv2.line(output, p1, p2, self.config.pose_color, self.config.pose_thickness)

                # Draw joint dots
                for idx in (11, 12, 13, 14, 15, 16):
                    if idx < len(raw_pts):
                        p = raw_pts[idx]
                        if p != (0, 0):
                            cv2.circle(output, p, 5, (255, 255, 255), -1)
                            cv2.circle(output, p, 3, self.config.pose_color, -1)
            elif mm_frame.pose_landmarks:
                for plm in mm_frame.pose_landmarks:
                    px, py = plm.to_pixel(w, h)
                    cv2.circle(output, (px, py), 4, self.config.pose_color, -1)

        # ── 2. Draw Face Mesh Expressive Outlines ────────────────────
        if self.config.show_face and mm_frame.has_face:
            raw_face = getattr(mm_frame, "_raw_face_pts", None)
            if raw_face and len(raw_face) > 400:
                # Draw facial contours
                for idx1, idx2 in FACE_CONTOURS:
                    if idx1 < len(raw_face) and idx2 < len(raw_face):
                        p1 = raw_face[idx1]
                        p2 = raw_face[idx2]
                        cv2.line(output, p1, p2, self.config.face_color, 1, cv2.LINE_AA)

            # Draw subtle dots for expressive anchor points
            for lm in mm_frame.face_landmarks:
                px, py = lm.to_pixel(w, h)
                cv2.circle(output, (px, py), self.config.face_landmark_radius, self.config.face_color, -1)

        # ── 3. Draw Hands ───────────────────────────────────────────
        if self.config.show_hands:
            if mm_frame.has_left_hand and mm_frame.left_hand is not None:
                self._draw_hand(output, mm_frame.left_hand, self.config.left_hand_color, "L", w, h)

            if mm_frame.has_right_hand and mm_frame.right_hand is not None:
                self._draw_hand(output, mm_frame.right_hand, self.config.right_hand_color, "R", w, h)

        return output

    def _draw_hand(
        self,
        image: np.ndarray,
        hand: Hand,
        color: tuple[int, int, int],
        label_prefix: str,
        w: int,
        h: int,
    ) -> None:
        """Draw landmarks and skeleton for one hand."""
        pts = [lm.to_pixel(w, h) for lm in hand.landmarks]

        # Draw bones
        for start_type, end_type in HAND_CONNECTIONS:
            p1 = pts[start_type.value]
            p2 = pts[end_type.value]
            cv2.line(image, p1, p2, color, self.config.connection_thickness, cv2.LINE_AA)

        # Draw joint circles
        for pt in pts:
            cv2.circle(image, pt, self.config.landmark_radius, (255, 255, 255), -1)
            cv2.circle(image, pt, self.config.landmark_radius - 1, color, -1)

        # Draw hand badge near wrist
        if self.config.show_labels and pts:
            wx, wy = pts[0]
            label_text = f"{label_prefix} Hand"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            bx, by = max(5, wx - 15), max(th + 5, wy - 10)
            cv2.rectangle(image, (bx - 2, by - th - 3), (bx + tw + 2, by + 3), (15, 20, 28), -1)
            cv2.putText(image, label_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
