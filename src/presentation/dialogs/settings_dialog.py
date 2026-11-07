"""Clean, student-friendly modal preferences dialog for ArSL Studio."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.infrastructure.camera_service import OpenCVCameraService
from src.infrastructure.config_manager import ConfigManager


class SettingsDialog(QDialog):
    """Modal dialog for application preferences and advanced diagnostics."""

    settings_saved = Signal(dict)

    def __init__(
        self,
        state: AppState,
        config_manager: Optional[ConfigManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences — ArSL Studio")
        self.setModal(True)
        self.setMinimumSize(540, 560)
        self.resize(580, 620)

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
        main_layout.setContentsMargins(20, 18, 20, 18)
        main_layout.setSpacing(14)

        # Dialog Title & Description
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_lbl = QLabel("Preferences & Hardware Configuration")
        title_lbl.setStyleSheet("font-size: 17px; font-weight: 700; color: #F3F4F6;")
        title_box.addWidget(title_lbl)

        desc_lbl = QLabel("Configure camera input, recognition sensitivity, speech audio, and advanced model options.")
        desc_lbl.setStyleSheet("font-size: 12px; color: #94A3B8;")
        title_box.addWidget(desc_lbl)
        main_layout.addLayout(title_box)

        # Scrollable Form Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 4, 8, 4)
        content_layout.setSpacing(16)

        # ── 1. Camera Settings ─────────────────────────────────────
        cam_group = QGroupBox("Camera Hardware")
        cam_form = QFormLayout(cam_group)
        cam_form.setSpacing(10)

        self._camera_select = QComboBox()
        self._populate_cameras()
        cam_form.addRow("Default Camera:", self._camera_select)

        self._resolution_select = QComboBox()
        self._resolution_select.addItem("640x480 (Recommended - Fast)", "640x480")
        self._resolution_select.addItem("1280x720 (High Definition)", "1280x720")
        cam_form.addRow("Resolution:", self._resolution_select)

        self._mirror_check = QCheckBox("Mirror camera feed horizontally")
        self._mirror_check.setChecked(True)
        cam_form.addRow("", self._mirror_check)

        content_layout.addWidget(cam_group)

        # ── 2. Recognition Sensitivity ─────────────────────────────
        recog_group = QGroupBox("Recognition Sensitivity")
        recog_form = QFormLayout(recog_group)
        recog_form.setSpacing(10)

        conf_layout = QHBoxLayout()
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(20, 90)
        self._conf_slider.setValue(40)
        conf_layout.addWidget(self._conf_slider, stretch=1)

        self._conf_display = QLabel("40%")
        self._conf_display.setStyleSheet("font-weight: 700; color: #14B8A6; min-width: 45px;")
        conf_layout.addWidget(self._conf_display)
        recog_form.addRow("Confidence Threshold:", conf_layout)

        self._show_landmarks = QCheckBox("Render MediaPipe skeletal tracking landmarks by default")
        self._show_landmarks.setChecked(True)
        recog_form.addRow("", self._show_landmarks)

        content_layout.addWidget(recog_group)

        # ── 3. Audio & Speech ──────────────────────────────────────
        audio_group = QGroupBox("Audio & Speech Output")
        audio_form = QFormLayout(audio_group)
        audio_form.setSpacing(10)

        self._tts_enabled = QCheckBox("Enable Text-to-Speech (TTS)")
        self._tts_enabled.setChecked(True)
        audio_form.addRow("", self._tts_enabled)

        rate_layout = QHBoxLayout()
        self._rate_slider = QSlider(Qt.Horizontal)
        self._rate_slider.setRange(50, 150)
        self._rate_slider.setValue(100)
        rate_layout.addWidget(self._rate_slider, stretch=1)

        self._rate_display = QLabel("1.0x")
        self._rate_display.setStyleSheet("font-weight: 600; color: #94A3B8; min-width: 45px;")
        rate_layout.addWidget(self._rate_display)
        audio_form.addRow("Speech Rate:", rate_layout)

        self._auto_speak = QCheckBox("Automatically speak newly completed sentences")
        self._auto_speak.setChecked(False)
        audio_form.addRow("", self._auto_speak)

        content_layout.addWidget(audio_group)

        # ── 4. Advanced / Research Models (Collapsible) ─────────────
        self._adv_toggle_btn = QPushButton("▸ Show Advanced & Research Options")
        self._adv_toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #3B82F6;
                text-align: left;
                font-size: 13px;
                font-weight: 600;
                padding: 4px 0;
            }
            QPushButton:hover {
                color: #60A5FA;
            }
        """)
        content_layout.addWidget(self._adv_toggle_btn)

        self._adv_container = QFrame()
        self._adv_container.setVisible(False)
        self._adv_container.setStyleSheet("background-color: #151a21; border-radius: 8px; padding: 6px;")
        adv_layout = QFormLayout(self._adv_container)
        adv_layout.setSpacing(10)

        self._model_selector = QComboBox()
        self._model_selector.addItem("Multimodal Iraqi (Two Hands + Face) [Recommended]", "multimodal")
        self._model_selector.addItem("KArSL-502 Pretrained (502 Standard Signs)", "karsl")
        self._model_selector.addItem("Iraqi Custom Classifier (Legacy Single-Hand)", "custom")
        adv_layout.addRow("Active Model Engine:", self._model_selector)

        self._enable_pose = QCheckBox("Track upper-body pose landmarks (Experimental)")
        self._enable_pose.setChecked(False)
        adv_layout.addRow("", self._enable_pose)

        self._show_fps_diagnostics = QCheckBox("Show real-time FPS & latency diagnostics in status bar")
        self._show_fps_diagnostics.setChecked(False)
        adv_layout.addRow("", self._show_fps_diagnostics)

        self._overlay_enabled = QCheckBox("Enable floating desktop subtitle overlay")
        self._overlay_enabled.setChecked(False)
        adv_layout.addRow("", self._overlay_enabled)

        content_layout.addWidget(self._adv_container)

        scroll.setWidget(content)
        main_layout.addWidget(scroll, stretch=1)

        # Bottom Action Buttons
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(10)

        self._reset_btn = QPushButton("Reset to Defaults")
        self._reset_btn.clicked.connect(self._reset_defaults)
        btn_bar.addWidget(self._reset_btn)

        btn_bar.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_bar.addWidget(self._cancel_btn)

        self._save_btn = QPushButton("Save Preferences")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.clicked.connect(self._on_save)
        btn_bar.addWidget(self._save_btn)

        main_layout.addLayout(btn_bar)

    def _populate_cameras(self) -> None:
        self._camera_select.clear()
        try:
            cameras = self._camera_service.list_available_cameras()
            if not cameras:
                self._camera_select.addItem("Default System Webcam (Index 0)", 0)
            else:
                for cam in cameras:
                    self._camera_select.addItem(cam.display_name, cam.device_index)
        except Exception:
            self._camera_select.addItem("Default System Webcam (Index 0)", 0)

    def _connect_signals(self) -> None:
        self._conf_slider.valueChanged.connect(lambda v: self._conf_display.setText(f"{v}%"))
        self._rate_slider.valueChanged.connect(lambda v: self._rate_display.setText(f"{v / 100:.1f}x"))
        self._adv_toggle_btn.clicked.connect(self._toggle_advanced)

    def _toggle_advanced(self) -> None:
        is_vis = not self._adv_container.isVisible()
        self._adv_container.setVisible(is_vis)
        self._adv_toggle_btn.setText("▾ Hide Advanced & Research Options" if is_vis else "▸ Show Advanced & Research Options")

    def load_from_config(self) -> None:
        try:
            dev_idx = getattr(self._state, "selected_camera_index", 0)
            combo_idx = self._camera_select.findData(dev_idx)
            if combo_idx >= 0:
                self._camera_select.setCurrentIndex(combo_idx)

            conf = self._config_mgr.get("recognition.confidence_threshold.static", 0.40)
            self._conf_slider.setValue(int(float(conf) * 100))

            overlay_en = self._config_mgr.get("overlay.enabled", False)
            self._overlay_enabled.setChecked(bool(overlay_en))
        except Exception as exc:
            self._logger.debug(f"Config load error: {exc}")

    def _on_save(self) -> None:
        device_idx = self._camera_select.currentData()
        if device_idx is None:
            device_idx = 0

        res_str = self._resolution_select.currentData() or "640x480"
        parts = res_str.split("x")
        w, h = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (640, 480)

        conf_val = self._conf_slider.value() / 100.0

        data = {
            "camera": {
                "device_index": device_idx,
                "width": w,
                "height": h,
                "mirror": self._mirror_check.isChecked(),
            },
            "recognition": {
                "confidence_threshold": conf_val,
                "show_landmarks": self._show_landmarks.isChecked(),
                "model": self._model_selector.currentData() or "multimodal",
                "enable_pose": self._enable_pose.isChecked(),
            },
            "audio": {
                "enabled": self._tts_enabled.isChecked(),
                "rate": self._rate_slider.value() / 100.0,
                "auto_speak": self._auto_speak.isChecked(),
            },
            "ui": {
                "show_fps": self._show_fps_diagnostics.isChecked(),
                "overlay_enabled": self._overlay_enabled.isChecked(),
            },
        }

        # Apply to state
        self._state.selected_camera_index = device_idx
        self._state.confidence_threshold = conf_val
        self._state.tts_enabled = self._tts_enabled.isChecked()

        try:
            self._config_mgr.set("recognition.confidence_threshold.static", conf_val)
            self._config_mgr.set("overlay.enabled", self._overlay_enabled.isChecked())
            self._config_mgr.save_user_config()
        except Exception as e:
            self._logger.warning(f"Could not persist config to disk: {e}")

        self.settings_saved.emit(data)
        self.accept()

    def _reset_defaults(self) -> None:
        self._resolution_select.setCurrentIndex(0)
        self._mirror_check.setChecked(True)
        self._conf_slider.setValue(40)
        self._show_landmarks.setChecked(True)
        self._tts_enabled.setChecked(True)
        self._rate_slider.setValue(100)
        self._auto_speak.setChecked(False)
        self._model_selector.setCurrentIndex(0)
        self._enable_pose.setChecked(False)
        self._show_fps_diagnostics.setChecked(False)
        self._overlay_enabled.setChecked(False)
