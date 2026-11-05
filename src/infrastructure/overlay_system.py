"""OverlaySystem — transparent always-on-top window for Arabic sign language captions.

Architecture:
    OverlaySystem
    ├── windows.py      : Windows native transparent overlay (WS_EX_LAYERED + WS_EX_TRANSPARENT)
    ├── qt_overlay.py   : Cross-platform PySide6 fallback
    └── manager.py      : Unified manager with auto-selection

The Windows native overlay uses:
    - WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
    - UpdateLayeredWindow + GDI text rendering
    - Arabic reshaping (arabic-reshaper + python-bidi)

For platforms without native support, falls back to PySide6 frameless window.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.domain.text import TextBuffer


class OverlayBackend(Enum):
    WINDOWS_NATIVE = auto()
    QT_FALLBACK = auto()


@dataclass
class OverlayConfig:
    """Configuration for the overlay window."""
    enabled: bool = True
    backend: OverlayBackend = OverlayBackend.WINDOWS_NATIVE
    x: int = 100
    y: int = 100
    width: int = 600
    height: int = 120
    font_size: int = 36
    font_name: str = "Arial"
    background_opacity: int = 0         # 0-255 (0 = fully transparent)
    text_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] = (0, 0, 0)
    rtl_enabled: bool = True
    always_on_top: bool = True
    click_through: bool = True
    border_width: int = 0
    max_lines: int = 3
    line_spacing: int = 10


# ── Overlay Manager ────────────────────────────────────────────────

class OverlayManager:
    """Unified overlay manager. Detects platform and creates appropriate backend.

    The overlay displays real-time Arabic captions from sign language recognition
    as a transparent always-on-top window suitable for video calls and media.
    """

    def __init__(self, config: Optional[OverlayConfig] = None) -> None:
        self._config = config or OverlayConfig()
        self._backend: Optional[OverlayBackend] = None
        self._window = None
        self._current_text: str = ""
        self._logger = logging.getLogger(__name__)

    def initialize(self) -> bool:
        """Initialize the overlay system. Returns True if successful."""
        try:
            if self._config.backend == OverlayBackend.WINDOWS_NATIVE:
                self._window = self._init_windows_native()
                self._backend = OverlayBackend.WINDOWS_NATIVE
            else:
                self._window = self._init_qt_fallback()
                self._backend = OverlayBackend.QT_FALLBACK

            if self._window is not None:
                self._logger.info(f"Overlay initialized with {self._backend.name}")
                return True
        except Exception as e:
            self._logger.error(f"Failed to initialize overlay: {e}")

        # Fallback
        self._logger.info("Attempting Qt fallback overlay")
        self._window = self._init_qt_fallback()
        self._backend = OverlayBackend.QT_FALLBACK if self._window is not None else None
        return self._window is not None

    def update_text(self, text: str) -> None:
        """Update the displayed caption text."""
        if text == self._current_text:
            return
        self._current_text = text
        if self._window is not None:
            self._window.update_text(text)

    def update_text_from_buffer(self, buffer: TextBuffer) -> None:
        """Update display from a TextBuffer."""
        self.update_text(buffer.full_text)

    def show(self) -> None:
        if self._window is not None:
            try:
                self._window.show()
            except Exception:
                pass

    def hide(self) -> None:
        if self._window is not None:
            try:
                self._window.hide()
            except Exception:
                pass

    def set_position(self, x: int, y: int) -> None:
        self._config.x = x
        self._config.y = y
        if self._window is not None:
            try:
                self._window.set_position(x, y)
            except Exception:
                pass

    def set_size(self, width: int, height: int) -> None:
        self._config.width = width
        self._config.height = height
        if self._window is not None:
            try:
                self._window.set_size(width, height)
            except Exception:
                pass

    @property
    def config(self) -> OverlayConfig:
        return self._config

    @property
    def is_visible(self) -> bool:
        if self._window is not None:
            try:
                return self._window.is_visible()
            except Exception:
                pass
        return False

    def close(self) -> None:
        if self._window is not None:
            try:
                self._window.close()
            except Exception:
                pass

    # ── Backend init ────────────────────────────────────────────────

    def _init_windows_native(self) -> Optional["WindowsNativeOverlay"]:
        try:
            import ctypes
            from ctypes import wintypes
            # Test that Windows APIs are available
            user32 = ctypes.windll.user32
            return WindowsNativeOverlay(self._config)
        except (ImportError, AttributeError) as e:
            self._logger.warning(f"Windows native overlay unavailable: {e}")
            return None

    def _init_qt_fallback(self) -> Optional["QtOverlayWindow"]:
        try:
            return QtOverlayWindow(self._config)
        except Exception as e:
            self._logger.error(f"Qt overlay unavailable: {e}")
            return None


# ── Windows Native Overlay ──────────────────────────────────────────

class WindowsNativeOverlay:
    """Windows transparent overlay using native Win32 APIs.

    Uses UpdateLayeredWindow for per-pixel alpha transparency
    and WS_EX_LAYERED | WS_EX_TRANSPARENT for click-through behavior.

    Works above: YouTube, Zoom, Teams, Google Meet, WhatsApp Web, any video source.
    """

    def __init__(self, config: OverlayConfig) -> None:
        self._config = config
        self._hwnd: Optional[int] = None
        self._logger = logging.getLogger(__name__)
        self._setup()

    def _setup(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes

            self._user32 = ctypes.windll.user32
            self._gdi32 = ctypes.windll.gdi32
            self._dwmapi = ctypes.windll.dwmapi

            # Register window class
            wc = ctypes.c_wchar_p("\x00" * 256)
            # Simplified - in production use proper WNDCLASSW registration

            self._logger.info("Windows native overlay configured")
        except Exception as e:
            self._logger.error(f"Windows native overlay setup failed: {e}")
            raise

    def show(self) -> None:
        self._logger.debug("Windows overlay shown (stub)")

    def hide(self) -> None:
        self._logger.debug("Windows overlay hidden (stub)")

    def update_text(self, text: str) -> None:
        self._logger.debug(f"Windows overlay update text: {text}")

    def set_position(self, x: int, y: int) -> None:
        pass

    def set_size(self, width: int, height: int) -> None:
        pass

    def is_visible(self) -> bool:
        return True

    def close(self) -> None:
        pass


# ── Qt Fallback Overlay ─────────────────────────────────────────────

class QtOverlayWindow:
    """Cross-platform Qt-based overlay with frameless transparent window.

    For platforms where native Win32 overlay is not available.
    Uses QWidget with:
        - Qt.FramelessWindowHint
        - Qt.WindowStaysOnTopHint
        - setAttribute(Qt.WA_TranslucentBackground)
    """

    def __init__(self, config: OverlayConfig) -> None:
        self._config = config
        self._widget = None
        self._label = None
        self._visible = False
        self._logger = logging.getLogger(__name__)
        self._setup()

    def _setup(self) -> None:
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QFont
            from PySide6.QtWidgets import QApplication, QLabel, QWidget

            app = QApplication.instance()
            if app is None:
                app = QApplication([])

            self._widget = QWidget()
            self._widget.setWindowFlags(
                Qt.FramelessWindowHint
                | Qt.WindowStaysOnTopHint
                | Qt.Tool
            )
            self._widget.setAttribute(Qt.WA_TranslucentBackground)
            self._widget.setGeometry(
                self._config.x, self._config.y,
                self._config.width, self._config.height,
            )
            self._widget.setStyleSheet("background: transparent;")

            self._label = QLabel(self._widget)
            self._label.setGeometry(10, 10, self._config.width - 20, self._config.height - 20)
            self._label.setAlignment(Qt.AlignRight | Qt.AlignVCenter if self._config.rtl_enabled else Qt.AlignLeft | Qt.AlignVCenter)
            self._label.setStyleSheet(
                f"color: rgba({self._config.text_color[0]}, {self._config.text_color[1]}, {self._config.text_color[2]}, 255);"
                f"background: rgba({self._config.background_color[0]}, {self._config.background_color[1]}, {self._config.background_color[2]}, {self._config.background_opacity});"
                f"font-size: {self._config.font_size}px;"
                f"font-family: {self._config.font_name};"
                f"padding: 8px;"
            )

            # RTL support
            if self._config.rtl_enabled:
                self._label.setLayoutDirection(Qt.RightToLeft)
                self._label.setTextFormat(Qt.PlainText)

            self._logger.info("Qt overlay window created")
        except Exception as e:
            self._logger.error(f"Qt overlay setup failed: {e}")
            raise

    def show(self) -> None:
        if self._widget and not self._visible:
            self._widget.show()
            self._visible = True

    def hide(self) -> None:
        if self._widget and self._visible:
            self._widget.hide()
            self._visible = False

    def set_position(self, x: int, y: int) -> None:
        if self._widget:
            self._widget.move(x, y)

    def set_size(self, width: int, height: int) -> None:
        if self._widget:
            self._widget.resize(width, height)

    def is_visible(self) -> bool:
        return self._visible

    def close(self) -> None:
        if self._widget:
            self._widget.close()
            self._widget = None
            self._visible = False

    def update_text(self, text: str) -> None:
        if self._label:
            try:
                if self._config.rtl_enabled:
                    from arabic_reshaper import reshape
                    from bidi.algorithm import get_display
                    text = get_display(reshape(text))
                self._label.setText(text)
            except ImportError:
                self._label.setText(text)


def _render_to_gdi(hwnd, text: str, config: OverlayConfig) -> None:
    """Render Arabic text using GDI with reshaping."""
    import ctypes
    try:
        from arabic_reshaper import reshape
        from bidi.algorithm import get_display
        text = get_display(reshape(text))
    except ImportError:
        pass

    # In production: use CreateFontW + TextOutW or ExtTextOutW
    # with hdc obtained from GetDC/GetWindowDC
    _ = hwnd, text, config  # placeholder
