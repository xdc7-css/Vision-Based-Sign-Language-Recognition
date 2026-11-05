"""ArabicTextRenderer — RTL Arabic text rendering with reshaping and bidi.

Used by the overlay system to correctly display Arabic text.
Integrates arabic-reshaper + python-bidi for proper Arabic glyph shaping
and bidirectional text layout.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np


class ArabicTextRenderer:
    """Render Arabic text onto images with proper RTL shaping.

    Handles:
        - Arabic letter reshaping (initial, medial, final, isolated forms)
        - Bidirectional text (Arabic + numbers + Latin)
        - Right-to-left layout
        - Configurable font, size, color
    """

    def __init__(self) -> None:
        self._reshaping_available = False
        self._bidi_available = False
        self._logger = logging.getLogger(__name__)

        try:
            from arabic_reshaper import reshape
            self._reshape = reshape
            self._reshaping_available = True
        except ImportError:
            self._reshape = lambda t: t
            self._logger.warning("arabic-reshaper not installed — Arabic text may not display correctly")

        try:
            from bidi.algorithm import get_display
            self._get_display = get_display
            self._bidi_available = True
        except ImportError:
            self._get_display = lambda t: t
            self._logger.warning("python-bidi not installed — Arabic text may not display correctly")

    def reshape(self, text: str) -> str:
        """Reshape Arabic letters and apply bidirectional algorithm."""
        try:
            reshaped = self._reshape(text)
            return self._get_display(reshaped)
        except Exception as e:
            self._logger.warning(f"Arabic reshaping failed: {e}")
            return text

    def render_text(
        self,
        text: str,
        font_path: str = "",
        font_size: int = 36,
        color: tuple[int, int, int] = (255, 255, 255),
        bg_color: Optional[tuple[int, int, int, int]] = None,
        width: int = 600,
        height: int = 120,
    ) -> np.ndarray:
        """Render Arabic text onto a transparent/non-transparent image.

        Args:
            text: Arabic text to render
            font_path: Path to TTF font file (empty = default)
            font_size: Font size in points
            color: RGB text color
            bg_color: Optional RGBA background color (None = transparent)
            width, height: Output image dimensions

        Returns:
            RGBA numpy array ready for overlay display
        """
        reshaped = self.reshape(text)

        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new("RGBA", (width, height), bg_color or (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()

            # Get text bounding box
            bbox = draw.textbbox((0, 0), reshaped, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # Center text
            x = (width - text_w) // 2
            y = (height - text_h) // 2

            draw.text((x, y), reshaped, font=font, fill=(*color, 255))
            return np.array(img)

        except ImportError:
            self._logger.warning("PIL not available, cannot render Arabic text")
            return np.zeros((height, width, 4), dtype=np.uint8)

    def render_text_pil(
        self,
        text: str,
        image: object,
        position: tuple[int, int] = (10, 10),
        font_path: str = "",
        font_size: int = 36,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> object:
        """Draw reshaped Arabic text onto an existing PIL Image."""
        reshaped = self.reshape(text)

        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            draw.text(position, reshaped, font=font, fill=(*color, 255))
            return image
        except ImportError:
            return image
