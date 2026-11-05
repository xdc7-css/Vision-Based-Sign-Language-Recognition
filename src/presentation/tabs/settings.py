from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.presentation.widgets import DialectSelector, SectionHeader


class SettingsTab(QWidget):
    """Application settings tab."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        header = SectionHeader(
            "Settings",
            "Configure application behavior and preferences",
        )
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(12)

        # General settings
        general_group = QGroupBox("General")
        general_form = QFormLayout(general_group)

        self._ui_language = QComboBox()
        self._ui_language.addItems(["Arabic (العربية)", "English"])
        self._ui_language.setCurrentIndex(0)
        general_form.addRow("Interface Language:", self._ui_language)

        self._minimize_tray = QCheckBox("Minimize to system tray")
        self._minimize_tray.setChecked(True)
        general_form.addRow("", self._minimize_tray)

        self._start_minimized = QCheckBox("Start minimized")
        general_form.addRow("", self._start_minimized)

        self._auto_start = QCheckBox("Auto-start recognition on launch")
        general_form.addRow("", self._auto_start)

        content_layout.addWidget(general_group)

        # Dialect management
        dialect_group = QGroupBox("Dialect Management")
        dialect_form = QFormLayout(dialect_group)

        self._default_dialect = DialectSelector()
        dialect_form.addRow("Default Dialect:", self._default_dialect)

        dialect_info = QLabel(
            "Dialect packs contain the trained models and gesture vocabulary\n"
            "for a specific sign language dialect. Install packs by placing\n"
            "them in the data/models/ directory."
        )
        dialect_info.setStyleSheet("font-size: 12px; color: #9d9d9d; padding: 8px 0;")
        dialect_form.addRow(dialect_info)

        dialect_btn_row = QHBoxLayout()
        self._install_dialect_btn = QPushButton("📥  Install Dialect Pack")
        dialect_btn_row.addWidget(self._install_dialect_btn)
        self._reload_dialects_btn = QPushButton("🔄  Reload")
        dialect_btn_row.addWidget(self._reload_dialects_btn)
        dialect_btn_row.addStretch()
        dialect_form.addRow(dialect_btn_row)

        content_layout.addWidget(dialect_group)

        # Recognition settings
        recog_group = QGroupBox("Recognition")
        recog_form = QFormLayout(recog_group)

        self._static_threshold = QSpinBox()
        self._static_threshold.setRange(50, 99)
        self._static_threshold.setValue(85)
        self._static_threshold.setSuffix(" %")
        recog_form.addRow("Static Gesture Threshold:", self._static_threshold)

        self._dynamic_threshold = QSpinBox()
        self._dynamic_threshold.setRange(50, 99)
        self._dynamic_threshold.setValue(75)
        self._dynamic_threshold.setSuffix(" %")
        recog_form.addRow("Dynamic Gesture Threshold:", self._dynamic_threshold)

        self._alphabet_threshold = QSpinBox()
        self._alphabet_threshold.setRange(50, 99)
        self._alphabet_threshold.setValue(90)
        self._alphabet_threshold.setSuffix(" %")
        recog_form.addRow("Alphabet Threshold:", self._alphabet_threshold)

        content_layout.addWidget(recog_group)

        # Pipeline settings
        pipeline_group = QGroupBox("Performance")
        pipeline_form = QFormLayout(pipeline_group)

        self._max_fps = QSpinBox()
        self._max_fps.setRange(5, 120)
        self._max_fps.setValue(30)
        self._max_fps.setSuffix(" FPS")
        pipeline_form.addRow("Target Frame Rate:", self._max_fps)

        self._frame_width = QSpinBox()
        self._frame_width.setRange(160, 3840)
        self._frame_width.setValue(640)
        self._frame_width.setSuffix(" px")
        pipeline_form.addRow("Frame Width:", self._frame_width)

        self._frame_height = QSpinBox()
        self._frame_height.setRange(120, 2160)
        self._frame_height.setValue(480)
        self._frame_height.setSuffix(" px")
        pipeline_form.addRow("Frame Height:", self._frame_height)

        content_layout.addWidget(pipeline_group)

        # Logging
        logging_group = QGroupBox("Logging")
        logging_form = QFormLayout(logging_group)

        self._log_level = QComboBox()
        self._log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._log_level.setCurrentIndex(1)
        logging_form.addRow("Log Level:", self._log_level)

        content_layout.addWidget(logging_group)

        # Save button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._save_btn = QPushButton("💾  Save Settings")
        self._save_btn.setObjectName("successButton")
        btn_layout.addWidget(self._save_btn)
        content_layout.addLayout(btn_layout)

        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

    def load_from_config(self) -> None:
        """Load settings from config repository."""
        pass  # populated in a future integration pass

    def save_to_config(self) -> dict:
        """Collect current settings as a config dict."""
        return {
            "ui": {
                "language": "ar" if self._ui_language.currentIndex() == 0 else "en",
                "minimize_to_tray": self._minimize_tray.isChecked(),
            },
            "recognition": {
                "confidence_threshold": {
                    "static": self._static_threshold.value() / 100,
                    "dynamic": self._dynamic_threshold.value() / 100,
                    "alphabet": self._alphabet_threshold.value() / 100,
                },
            },
            "pipeline": {
                "max_fps": self._max_fps.value(),
                "frame_width": self._frame_width.value(),
                "frame_height": self._frame_height.value(),
            },
            "logging": {
                "level": self._log_level.currentText().lower(),
            },
        }
