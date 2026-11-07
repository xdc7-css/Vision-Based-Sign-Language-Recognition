"""HandRenderer service — draws landmarks, skeleton, and labels on video frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from src.domain.hand import (
    HAND_CONNECTIONS,
    Hand,
    HandDetectionResult,
    Handedness,
)


@dataclass
class RenderConfig:
    """Visual configuration for hand rendering."""

    landmark_color: tuple[int, int, int] = (0, 255, 255)
    landmark_radius: int = 5
    landmark_thickness: int = -1

    connection_color: tuple[int, int, int] = (255, 255, 255)
    connection_thickness: int = 2

    left_hand_color: tuple[int, int, int] = (65, 200, 255)
    right_hand_color: tuple[int, int, int] = (255, 150, 65)

    label_color: tuple[int, int, int] = (0, 255, 0)
    label_font_scale: float = 0.6
    label_thickness: int = 2
    label_background_alpha: float = 0.6

    show_handedness: bool = True
    show_confidence: bool = True
    show_connections: bool = True
    show_landmarks: bool = True

    bounding_box_color: tuple[int, int, int] = (100, 100, 255)
    bounding_box_thickness: int = 2


class HandRenderer:
    """Renders hand landmarks, skeleton, and labels onto a copy of the image.

    Usage:
        renderer = HandRenderer()
        output = renderer.draw(frame, detection_result)
    """

    def __init__(self, config: Optional[RenderConfig] = None) -> None:
        self._config = config or RenderConfig()
        self._label_bg_cache: dict[str, tuple[int, int, int, int, str]] = {}

    # ── Public API ──────────────────────────────────────────────────

    def draw(self, image: np.ndarray, result: HandDetectionResult) -> np.ndarray:
        """Draw all hands on a copy of *image*.

        Returns a new array; the original is not modified.
        """
        output = image.copy()

        if not result.has_hands:
            return output

        for hand in result.hands:
            if self._config.show_connections:
                output = self.draw_skeleton(output, hand)
            if self._config.show_landmarks:
                output = self.draw_landmarks(output, hand)
            if self._config.show_handedness or self._config.show_confidence:
                output = self.draw_label(output, hand, result)
            output = self.draw_bounding_box(output, hand)

        return output

    def render(self, image: np.ndarray, result: Any) -> np.ndarray:
        """Alias for draw to support both HandDetectionResult and list[Hand]."""
        if isinstance(result, HandDetectionResult):
            return self.draw(image, result)
        elif isinstance(result, list):
            res_obj = HandDetectionResult(hands=result, image_width=image.shape[1], image_height=image.shape[0], fps=0.0)
            return self.draw(image, res_obj)
        return image.copy()

    def draw_landmarks(self, image: np.ndarray, hand: Hand) -> np.ndarray:
        """Draw landmark dots for one hand (no skeleton)."""
        output = image.copy()
        color = self._hand_color(hand.handedness)
        h, w = output.shape[:2]

        for lm in hand.landmarks:
            px, py = lm.to_pixel(w, h)
            cv2.circle(output, (px, py), self._config.landmark_radius, color, self._config.landmark_thickness)

        return output

    def draw_skeleton(self, image: np.ndarray, hand: Hand) -> np.ndarray:
        """Draw connection lines for one hand (no landmarks)."""
        output = image.copy()
        color = self._hand_color(hand.handedness)
        h, w = output.shape[:2]

        for start_type, end_type in HAND_CONNECTIONS:
            p1 = hand[start_type].to_pixel(w, h)
            p2 = hand[end_type].to_pixel(w, h)
            cv2.line(output, p1, p2, color, self._config.connection_thickness)

        return output

    def draw_label(
        self,
        image: np.ndarray,
        hand: Hand,
        result: HandDetectionResult,
    ) -> np.ndarray:
        """Draw handedness label and confidence score above the hand."""
        output = image.copy()
        h, w = output.shape[:2]

        bbox = self._compute_bounding_box_pixels(hand, w, h)
        x_min, y_min = bbox[0], bbox[1]
        label_parts = []

        if self._config.show_handedness:
            icon = "☛" if hand.handedness == Handedness.RIGHT else "☚"
            label_parts.append(f"{icon} {hand.handedness.value}")

        if self._config.show_confidence:
            label_parts.append(f"{hand.confidence:.0%}")

        label_text = "  ".join(label_parts)
        if not label_text:
            return output

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._config.label_font_scale
        thickness = self._config.label_thickness
        (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thickness)

        # Draw background rectangle
        bx = x_min
        by = max(y_min - th - 8, 0)
        bg = self._config.label_background_alpha
        overlay = output.copy()
        cv2.rectangle(overlay, (bx, by), (bx + tw + 8, by + th + 8), self._config.label_color, -1)
        output = cv2.addWeighted(overlay, bg, output, 1 - bg, 0)

        # Draw text
        text_color = (0, 0, 0)
        cv2.putText(output, label_text, (bx + 4, by + th + 4), font, scale, text_color, thickness)

        return output

    def draw_bounding_box(self, image: np.ndarray, hand: Hand) -> np.ndarray:
        """Draw a bounding box around the hand."""
        output = image.copy()
        h, w = output.shape[:2]
        x_min, y_min, x_max, y_max = self._compute_bounding_box_pixels(hand, w, h)

        cv2.rectangle(
            output,
            (x_min, y_min),
            (x_max, y_max),
            self._config.bounding_box_color,
            self._config.bounding_box_thickness,
        )

        return output

    # ── Private ─────────────────────────────────────────────────────

    def _hand_color(self, handedness: Handedness) -> tuple[int, int, int]:
        if handedness == Handedness.LEFT:
            return self._config.left_hand_color
        elif handedness == Handedness.RIGHT:
            return self._config.right_hand_color
        return self._config.landmark_color

    def _draw_landmarks(self, image: np.ndarray, hand: Hand) -> np.ndarray:
        color = self._hand_color(hand.handedness)
        h, w = image.shape[:2]

        for lm in hand.landmarks:
            px, py = lm.to_pixel(w, h)
            cv2.circle(image, (px, py), self._config.landmark_radius, color, self._config.landmark_thickness)

        return image

    def _draw_skeleton(self, image: np.ndarray, hand: Hand) -> np.ndarray:
        color = self._hand_color(hand.handedness)
        h, w = image.shape[:2]

        for start_type, end_type in HAND_CONNECTIONS:
            p1 = hand[start_type].to_pixel(w, h)
            p2 = hand[end_type].to_pixel(w, h)
            cv2.line(image, p1, p2, color, self._config.connection_thickness)

        return image

    def _compute_bounding_box_pixels(
        self, hand: Hand, img_w: int, img_h: int
    ) -> tuple[int, int, int, int]:
        xs = [lm.x for lm in hand.landmarks]
        ys = [lm.y for lm in hand.landmarks]

        padding = 0.05
        x_min = max(0, int(min(xs) * img_w) - int(padding * img_w))
        y_min = max(0, int(min(ys) * img_h) - int(padding * img_h))
        x_max = min(img_w, int(max(xs) * img_w) + int(padding * img_w))
        y_max = min(img_h, int(max(ys) * img_h) + int(padding * img_h))

        return (x_min, y_min, x_max, y_max)
