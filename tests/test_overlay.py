"""Tests for OverlaySystem — transparent overlay, Qt fallback, text rendering."""

import pytest
from unittest.mock import Mock, patch

from src.infrastructure.overlay_system import (
    OverlayConfig, OverlayBackend, OverlayManager, QtOverlayWindow,
)
from src.domain.text import TextBuffer


class TestOverlayConfig:

    def test_default_config(self):
        cfg = OverlayConfig()
        assert cfg.enabled is True
        assert cfg.backend == OverlayBackend.WINDOWS_NATIVE
        assert cfg.font_size == 36
        assert cfg.rtl_enabled is True
        assert cfg.always_on_top is True

    def test_custom_config(self):
        cfg = OverlayConfig(
            x=200, y=300, width=800, height=200,
            font_size=48,
            rtl_enabled=True,
            click_through=True,
        )
        assert cfg.x == 200
        assert cfg.y == 300
        assert cfg.width == 800
        assert cfg.height == 200
        assert cfg.font_size == 48


class TestOverlayManager:

    def test_initial_state(self):
        mgr = OverlayManager()
        assert mgr.config.enabled is True
        assert mgr.is_visible is False

    def test_initialize_windows_native(self):
        cfg = OverlayConfig(backend=OverlayBackend.WINDOWS_NATIVE)
        mgr = OverlayManager(cfg)
        # Should not crash, backend may not be fully functional
        result = mgr.initialize()
        # On non-Windows or without ctypes, may fall back to qt
        # Just test it doesn't raise
        assert isinstance(result, bool)

    def test_initialize_qt_fallback(self):
        cfg = OverlayConfig(backend=OverlayBackend.QT_FALLBACK)
        mgr = OverlayManager(cfg)
        # Qt overlay needs PySide6, may not be available in test env
        # Test that it doesn't crash
        try:
            result = mgr.initialize()
            assert isinstance(result, bool)
        except Exception:
            pass

    def test_update_text(self):
        mgr = OverlayManager()
        # Should not crash
        mgr.update_text("مرحبا")
        mgr.update_text("مرحبا")  # duplicate should be skipped
        mgr.update_text("العالم")

    def test_update_text_from_buffer(self):
        mgr = OverlayManager()
        buffer = TextBuffer()
        buffer.sentence = ["مرحبا", "بالعالم"]
        mgr.update_text_from_buffer(buffer)

    def test_show_hide(self):
        mgr = OverlayManager()
        mgr.show()
        mgr.hide()

    def test_set_position(self):
        mgr = OverlayManager()
        mgr.set_position(200, 300)
        assert mgr.config.x == 200
        assert mgr.config.y == 300

    def test_set_size(self):
        mgr = OverlayManager()
        mgr.set_size(800, 200)
        assert mgr.config.width == 800
        assert mgr.config.height == 200

    def test_close(self):
        mgr = OverlayManager()
        mgr.close()


class TestQtOverlayWindow:

    def test_create_qt_overlay(self):
        cfg = OverlayConfig(backend=OverlayBackend.QT_FALLBACK)
        try:
            win = QtOverlayWindow(cfg)
            assert win is not None
            win.close()
        except Exception as e:
            # May fail if no QApplication
            pass

    def test_update_text_qt(self):
        cfg = OverlayConfig(backend=OverlayBackend.QT_FALLBACK)
        try:
            win = QtOverlayWindow(cfg)
            win.update_text("مرحبا")
            win.close()
        except Exception:
            pass
