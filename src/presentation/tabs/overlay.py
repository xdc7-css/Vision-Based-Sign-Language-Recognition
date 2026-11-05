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
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.infrastructure.overlay_system import OverlayManager
from src.presentation.widgets import SectionHeader


class OverlayTab(QWidget):
    """Tab for configuring the transparent caption overlay."""

    def __init__(
        self,
        state: AppState,
        overlay_manager: Optional[OverlayManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._overlay_manager = overlay_manager or OverlayManager()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        header = SectionHeader(
            "Overlay Configuration",
            "Configure the transparent caption overlay appearance and behavior",
        )
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setSpacing(16)

        # Left: Preview
        preview_section = QFrame()
        preview_section.setObjectName("previewSection")
        preview_section.setStyleSheet("""
            QFrame#previewSection {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        preview_layout = QVBoxLayout(preview_section)

        preview_label = QLabel("Overlay Preview")
        preview_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e0e0e0;")
        preview_layout.addWidget(preview_label)

        self._preview = QLabel()
        self._preview.setMinimumSize(400, 150)
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setStyleSheet("""
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #1a1a1a, stop: 1 #2d2d2d
            );
            border: 1px solid #555555;
            border-radius: 6px;
            font-size: 24px;
            font-weight: 700;
            color: #e0e0e0;
            padding: 20px;
        """)
        self._preview.setText("مرحبا بك في نظام الترجمة")
        preview_layout.addWidget(self._preview, stretch=1)

        content_layout.addWidget(preview_section, stretch=1)

        # Right: Controls
        controls_section = QFrame()
        controls_section.setObjectName("controlsSection")
        controls_section.setStyleSheet("""
            QFrame#controlsSection {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        controls_layout = QVBoxLayout(controls_section)

        controls_label = QLabel("Overlay Settings")
        controls_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e0e0e0;")
        controls_layout.addWidget(controls_label)

        # Enable overlay
        self._enable_check = QCheckBox("Enable Overlay")
        self._enable_check.setChecked(self._overlay_manager.config.enabled)
        controls_layout.addWidget(self._enable_check)

        # Position
        pos_group = QGroupBox("Position")
        pos_form = QFormLayout(pos_group)

        self._position_combo = QComboBox()
        self._position_combo.addItems(["Bottom", "Top", "Custom"])
        pos_form.addRow("Location:", self._position_combo)

        self._x_offset = QSpinBox()
        self._x_offset.setRange(0, 9999)
        self._x_offset.setValue(self._overlay_manager.config.x)
        self._x_offset.setSuffix(" px")
        pos_form.addRow("X Offset:", self._x_offset)

        self._y_offset = QSpinBox()
        self._y_offset.setRange(0, 9999)
        self._y_offset.setValue(self._overlay_manager.config.y)
        self._y_offset.setSuffix(" px")
        pos_form.addRow("Y Offset:", self._y_offset)

        controls_layout.addWidget(pos_group)

        # Size
        size_group = QGroupBox("Size")
        size_form = QFormLayout(size_group)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(100, 3840)
        self._width_spin.setValue(self._overlay_manager.config.width)
        self._width_spin.setSuffix(" px")
        size_form.addRow("Width:", self._width_spin)

        self._height_spin = QSpinBox()
        self._height_spin.setRange(30, 1080)
        self._height_spin.setValue(self._overlay_manager.config.height)
        self._height_spin.setSuffix(" px")
        size_form.addRow("Height:", self._height_spin)

        controls_layout.addWidget(size_group)

        # Appearance
        appearance_group = QGroupBox("Appearance")
        appearance_form = QFormLayout(appearance_group)

        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(10, 100)
        self._opacity_slider.setValue(
            100 - (self._overlay_manager.config.background_opacity // 2.55)
        )
        self._opacity_label = QLabel(f"{self._opacity_slider.value()}%")
        appearance_form.addRow("Opacity:", self._opacity_slider)
        appearance_form.addRow("", self._opacity_label)

        self._font_combo = QComboBox()
        self._font_combo.setEditable(True)
        self._font_combo.addItems(["Arial", "Tahoma", "Times New Roman", "Traditional Arabic"])
        self._font_combo.setCurrentText(self._overlay_manager.config.font_name)
        appearance_form.addRow("Font:", self._font_combo)

        self._font_size = QSpinBox()
        self._font_size.setRange(12, 120)
        self._font_size.setValue(self._overlay_manager.config.font_size)
        self._font_size.setSuffix(" pt")
        appearance_form.addRow("Font Size:", self._font_size)

        controls_layout.addWidget(appearance_group)

        # Caption behavior
        behavior_group = QGroupBox("Caption Behavior")
        behavior_form = QFormLayout(behavior_group)

        self._max_lines = QSpinBox()
        self._max_lines.setRange(1, 10)
        self._max_lines.setValue(self._overlay_manager.config.max_lines)
        behavior_form.addRow("Max Lines:", self._max_lines)

        self._duration = QSpinBox()
        self._duration.setRange(1, 60)
        self._duration.setValue(5)
        self._duration.setSuffix(" sec")
        behavior_form.addRow("Line Duration:", self._duration)

        self._show_confidence = QCheckBox("Show Confidence Score")
        self._show_confidence.setChecked(False)
        behavior_form.addRow("", self._show_confidence)

        controls_layout.addWidget(behavior_group)

        # Apply button
        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.setObjectName("successButton")
        controls_layout.addWidget(self._apply_btn)
        controls_layout.addStretch()

        content_layout.addWidget(controls_section)
        content_layout.setStretch(0, 2)
        content_layout.setStretch(1, 1)

        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

    def _connect_signals(self) -> None:
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v}%")
        )
        self._enable_check.toggled.connect(self._on_enable_toggled)
        self._apply_btn.clicked.connect(self._apply_settings)
        self._position_combo.currentTextChanged.connect(self._on_position_changed)

    def _on_enable_toggled(self, checked: bool) -> None:
        self._state.overlay_visible = checked
        if checked:
            self._overlay_manager.show()
        else:
            self._overlay_manager.hide()

    def _on_position_changed(self, position: str) -> None:
        screen = QLabel().screen().geometry() if hasattr(QLabel(), 'screen') else None
        if screen is None:
            return

        if position == "Bottom":
            self._x_offset.setValue(screen.width() // 2 - self._overlay_manager.config.width // 2)
            self._y_offset.setValue(screen.height() - self._overlay_manager.config.height - 50)
        elif position == "Top":
            self._x_offset.setValue(screen.width() // 2 - self._overlay_manager.config.width // 2)
            self._y_offset.setValue(20)
        # "Custom" leaves current values

    def _apply_settings(self) -> None:
        config = self._overlay_manager.config
        config.x = self._x_offset.value()
        config.y = self._y_offset.value()
        config.width = self._width_spin.value()
        config.height = self._height_spin.value()
        config.font_name = self._font_combo.currentText()
        config.font_size = self._font_size.value()
        config.max_lines = self._max_lines.value()

        # Opacity: slider 10-100 → background_opacity 0-255
        opacity_val = self._opacity_slider.value()
        config.background_opacity = int((100 - opacity_val) * 2.55)

        self._overlay_manager.set_position(config.x, config.y)
        self._overlay_manager.set_size(config.width, config.height)
        self._overlay_manager.update_text(self._preview.text())

        self._state.log("info", f"Overlay config applied: {config.width}x{config.height} @ ({config.x}, {config.y})")
