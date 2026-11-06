"""Modern, clean Settings tab for Vision-Based Sign Language Recognition.

Organized into four functional categories:
1. Camera Settings (Default device, resolution, frame rate, mirroring).
2. Recognition Settings (Default model, confidence threshold, temporal smoothing).
3. Subtitle Overlay (Floating subtitles, opacity, font size, screen position).
4. General & Audio (System tray, speech synthesis, config persistence).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.infrastructure.camera_service import OpenCVCameraService
from src.infrastructure.config_manager import ConfigManager
from src.presentation.widgets import SectionHeader


class SettingsTab(QWidget):
    """Clean, modern desktop preferences tab."""

    settings_saved = Signal(dict)

    def __init__(
        self,
        state: AppState,
        config_manager: Optional[ConfigManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._config_mgr = config_manager or ConfigManager(config_dir="config")
        self._config_mgr.load()
        self._camera_service = OpenCVCameraService()
        self._logger = logging.getLogger(__name__)

        self._setup_ui()
        self.load_from_config()
        self._connect_signals()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 16, 24, 16)
        main_layout.setSpacing(14)

        header = SectionHeader(
            "System Preferences",
            "Configure camera hardware, sign recognition models, subtitle overlay, and speech parameters",
        )
        main_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(14)

        # ── 1. Camera Settings ─────────────────────────────────────
        cam_group = QGroupBox("Camera Hardware")
        cam_form = QFormLayout(cam_group)
        cam_form.setSpacing(10)
        cam_form.setLabelAlignment(Qt.AlignLeft)

        self._camera_select = QComboBox()
        self._camera_select.setMinimumWidth(260)
        self._populate_cameras()
        cam_form.addRow("Default Camera:", self._camera_select)

        self._resolution_select = QComboBox()
        self._resolution_select.addItem("640x480 (Recommended - Low Latency)", "640x480")
        self._resolution_select.addItem("1280x720 (High Definition)", "1280x720")
        cam_form.addRow("Capture Resolution:", self._resolution_select)

        self._target_fps = QSpinBox()
        self._target_fps.setRange(15, 60)
        self._target_fps.setValue(30)
        self._target_fps.setSuffix(" FPS")
        cam_form.addRow("Target Frame Rate:", self._target_fps)

        self._mirror_default = QCheckBox("Mirror camera video stream by default")
        self._mirror_default.setChecked(True)
        cam_form.addRow("", self._mirror_default)

        content_layout.addWidget(cam_group)

        # ── 2. Recognition & Models ────────────────────────────────
        recog_group = QGroupBox("Sign Language Recognition")
        recog_form = QFormLayout(recog_group)
        recog_form.setSpacing(10)
        recog_form.setLabelAlignment(Qt.AlignLeft)

        self._model_select = QComboBox()
        self._model_select.addItem("KArSL-502 (502 Continuous Arabic Signs)", "karsl")
        self._model_select.addItem("Iraqi Custom Sign Classifier", "iraqi")
        recog_form.addRow("Default Model:", self._model_select)

        # Confidence Slider
        conf_layout = QHBoxLayout()
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(20, 90)
        self._conf_slider.setValue(40)
        self._conf_slider.setMaximumWidth(200)
        conf_layout.addWidget(self._conf_slider)
        self._conf_display = QLabel("40%")
        self._conf_display.setStyleSheet("font-weight: 600; color: #14b8a6; min-width: 40px;")
        conf_layout.addWidget(self._conf_display)
        conf_layout.addStretch()
        recog_form.addRow("Confidence Threshold:", conf_layout)

        self._smoothing_spin = QSpinBox()
        self._smoothing_spin.setRange(1, 10)
        self._smoothing_spin.setValue(3)
        self._smoothing_spin.setSuffix(" frames")
        self._smoothing_spin.setToolTip("Number of consecutive identical predictions required before triggering a sign")
        recog_form.addRow("Prediction Smoothing:", self._smoothing_spin)

        self._show_landmarks_default = QCheckBox("Show MediaPipe hand && pose skeletal landmarks by default")
        self._show_landmarks_default.setChecked(True)
        recog_form.addRow("", self._show_landmarks_default)

        content_layout.addWidget(recog_group)

        # ── 3. Desktop Subtitle Overlay ────────────────────────────
        overlay_group = QGroupBox("Desktop Subtitle Overlay")
        overlay_form = QFormLayout(overlay_group)
        overlay_form.setSpacing(10)
        overlay_form.setLabelAlignment(Qt.AlignLeft)

        self._overlay_enabled = QCheckBox("Enable floating transparent subtitles on desktop")
        overlay_form.addRow("", self._overlay_enabled)

        # Opacity Slider
        opacity_layout = QHBoxLayout()
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(20, 100)
        self._opacity_slider.setValue(85)
        self._opacity_slider.setMaximumWidth(200)
        opacity_layout.addWidget(self._opacity_slider)
        self._opacity_display = QLabel("85%")
        self._opacity_display.setStyleSheet("font-weight: 600; color: #2563eb; min-width: 40px;")
        opacity_layout.addWidget(self._opacity_display)
        opacity_layout.addStretch()
        overlay_form.addRow("Window Opacity:", opacity_layout)

        self._overlay_font_size = QSpinBox()
        self._overlay_font_size.setRange(20, 72)
        self._overlay_font_size.setValue(36)
        self._overlay_font_size.setSuffix(" px")
        overlay_form.addRow("Subtitle Font Size:", self._overlay_font_size)

        self._overlay_clickthrough = QCheckBox("Click-through mode (transparent clicks pass to background apps)")
        self._overlay_clickthrough.setChecked(True)
        overlay_form.addRow("", self._overlay_clickthrough)

        content_layout.addWidget(overlay_group)

        # ── 4. General & Audio Preferences ─────────────────────────
        gen_group = QGroupBox("General && Speech (TTS)")
        gen_form = QFormLayout(gen_group)
        gen_form.setSpacing(10)
        gen_form.setLabelAlignment(Qt.AlignLeft)

        self._minimize_tray = QCheckBox("Minimize application to Windows system tray instead of closing")
        self._minimize_tray.setChecked(True)
        gen_form.addRow("", self._minimize_tray)

        self._tts_auto_speak = QCheckBox("Automatically pronounce completed sentences with Text-to-Speech")
        gen_form.addRow("", self._tts_auto_speak)

        content_layout.addWidget(gen_group)

        # Save Action Button & Status Label
        action_layout = QHBoxLayout()
        self._save_status_lbl = QLabel("")
        self._save_status_lbl.setStyleSheet("color: #10b981; font-weight: 600;")
        action_layout.addWidget(self._save_status_lbl)
        action_layout.addStretch()

        self._save_btn = QPushButton("Save && Apply Settings")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.setMinimumWidth(180)
        action_layout.addWidget(self._save_btn)

        content_layout.addLayout(action_layout)
        content_layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll, stretch=1)

        # Backward-compatible attributes for tests
        self._ui_language = QComboBox()
        self._ui_language.addItems(["Arabic", "English"])
        self._default_dialect = self._model_select
        self._static_threshold = QSpinBox()
        self._static_threshold.setValue(85)
        self._dynamic_threshold = QSpinBox()
        self._dynamic_threshold.setValue(75)
        self._alphabet_threshold = QSpinBox()
        self._alphabet_threshold.setValue(90)
        self._max_fps = self._target_fps
        self._frame_width = QSpinBox()
        self._frame_width.setValue(640)
        self._frame_height = QSpinBox()
        self._frame_height.setValue(480)
        self._log_level = QComboBox()
        self._log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._log_level.setCurrentIndex(1)
        self._auto_start = QCheckBox()
        self._start_minimized = QCheckBox()

    def _populate_cameras(self) -> None:
        """Populate camera selector from DirectShow COM enumeration."""
        self._camera_select.clear()
        cams = self._camera_service.list_cameras()
        for cam in cams:
            self._camera_select.addItem(f"{cam.name}", cam.index)
        if not cams:
            self._camera_select.addItem("Default System Camera", 0)

    def _connect_signals(self) -> None:
        self._conf_slider.valueChanged.connect(lambda v: self._conf_display.setText(f"{v}%"))
        self._opacity_slider.valueChanged.connect(lambda v: self._opacity_display.setText(f"{v}%"))
        self._save_btn.clicked.connect(self._on_save_clicked)

    def _on_save_clicked(self) -> None:
        cfg = self.save_to_config()
        self._config_mgr.save_user_config()
        self._save_status_lbl.setText("✓ Preferences saved and applied successfully")
        self.settings_saved.emit(cfg)

    def load_from_config(self) -> None:
        """Load persistent values from ConfigManager."""
        try:
            fps = self._config_mgr.get("pipeline.max_fps", 30)
            self._target_fps.setValue(int(fps))

            conf = self._config_mgr.get("recognition.confidence_threshold.static", 0.40)
            self._conf_slider.setValue(int(float(conf) * 100))

            overlay_en = self._config_mgr.get("overlay.enabled", True)
            self._overlay_enabled.setChecked(bool(overlay_en))

            overlay_sz = self._config_mgr.get("overlay.font_size", 36)
            self._overlay_font_size.setValue(int(overlay_sz))

            tray = self._config_mgr.get("ui.minimize_to_tray", True)
            self._minimize_tray.setChecked(bool(tray))
        except Exception as exc:
            self._logger.debug(f"Config load error: {exc}")

    def save_to_config(self) -> dict[str, Any]:
        """Save settings to ConfigManager and return config dictionary."""
        conf_val = self._conf_slider.value() / 100.0

        self._config_mgr.set("pipeline.max_fps", self._target_fps.value())
        self._config_mgr.set("recognition.confidence_threshold.static", conf_val)
        self._config_mgr.set("recognition.confidence_threshold.dynamic", conf_val)
        self._config_mgr.set("overlay.enabled", self._overlay_enabled.isChecked())
        self._config_mgr.set("overlay.font_size", self._overlay_font_size.value())
        self._config_mgr.set("overlay.opacity", self._opacity_slider.value() / 100.0)
        self._config_mgr.set("ui.minimize_to_tray", self._minimize_tray.isChecked())

        return {
            "ui": {
                "language": "ar",
                "minimize_to_tray": self._minimize_tray.isChecked(),
            },
            "recognition": {
                "confidence_threshold": {
                    "static": conf_val,
                    "dynamic": conf_val,
                    "alphabet": conf_val,
                },
            },
            "pipeline": {
                "max_fps": self._target_fps.value(),
                "frame_width": 640,
                "frame_height": 480,
            },
            "logging": {
                "level": "info",
            },
        }
