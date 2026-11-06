from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.domain.entities import CameraInfo, DialectInfo, ModelInfo, VideoSourceType


# ─── Camera Selector ───────────────────────────────────────────────────


class CameraSelector(QComboBox):
    """Dropdown for selecting an available camera."""

    camera_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setPlaceholderText("Select camera...")
        self.currentIndexChanged.connect(self._on_selection_changed)

    def populate(self, cameras: list[CameraInfo]) -> None:
        self.blockSignals(True)
        self.clear()
        for cam in cameras:
            label = f"{cam.name}"
            if not cam.is_available:
                label += " (unavailable)"
            self.addItem(label, cam.id)
        self.blockSignals(False)

    def _on_selection_changed(self, index: int) -> None:
        if index >= 0:
            camera_id = self.itemData(index)
            if camera_id:
                self.camera_selected.emit(camera_id)


# ─── Video Source Selector ─────────────────────────────────────────────


class SourceSelector(QWidget):
    """Radio button group for selecting video source type."""

    source_changed = Signal(VideoSourceType)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._group = QButtonGroup(self)
        self._group.idClicked.connect(self._on_selection_changed)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        sources = [
            (VideoSourceType.WEBCAM, "Webcam / USB"),
            (VideoSourceType.IP_CAMERA, "Phone Camera (Wi-Fi)"),
            (VideoSourceType.FILE, "Video File"),
            (VideoSourceType.SCREEN, "Screen Capture"),
            (VideoSourceType.WINDOW, "Window Capture"),
        ]

        for idx, (source_type, label) in enumerate(sources):
            radio = QRadioButton(label)
            radio.setStyleSheet("font-size: 12px; padding: 4px 0;")
            self._group.addButton(radio, idx)
            self._group.buttons()[-1].setProperty("source_type", source_type)
            layout.addWidget(radio)

        layout.addStretch()

    def _on_selection_changed(self, button_id: int) -> None:
        button = self._group.button(button_id)
        if button:
            source_type = button.property("source_type")
            self.source_changed.emit(source_type)

    def selected_source(self) -> VideoSourceType:
        button = self._group.checkedButton()
        if button:
            return button.property("source_type")
        return VideoSourceType.WEBCAM

    def set_source(self, source_type: VideoSourceType) -> None:
        for button in self._group.buttons():
            if button.property("source_type") == source_type:
                button.setChecked(True)
                break


# ─── Dialect Selector ──────────────────────────────────────────────────


class DialectSelector(QComboBox):
    """Dropdown for selecting an installed dialect pack."""

    dialect_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setPlaceholderText("Select dialect...")
        self.currentIndexChanged.connect(self._on_selection_changed)

    def populate(self, dialects: list[DialectInfo]) -> None:
        self.blockSignals(True)
        self.clear()
        for dialect in dialects:
            label = f"{dialect.name} ({dialect.name_ar})"
            self.addItem(label, dialect.dialect_id)
        self.blockSignals(False)

    def _on_selection_changed(self, index: int) -> None:
        if index >= 0:
            dialect_id = self.itemData(index)
            if dialect_id:
                self.dialect_selected.emit(dialect_id)


# ─── Model Selector ────────────────────────────────────────────────────


class ModelSelector(QComboBox):
    """Dropdown for selecting a trained model."""

    model_selected = Signal(str)

    def __init__(self, label_text: str = "Model:", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(180)
        self.setPlaceholderText(label_text)
        self.currentIndexChanged.connect(self._on_selection_changed)

    def populate(self, models: list[ModelInfo]) -> None:
        self.blockSignals(True)
        self.clear()
        for model in models:
            status = "✓" if model.is_loaded else "○"
            acc = f" ({model.accuracy:.1%})" if model.accuracy > 0 else ""
            label = f"{status} {model.model_type.title()}{acc}"
            self.addItem(label, model.model_type)
        self.blockSignals(False)

    def _on_selection_changed(self, index: int) -> None:
        if index >= 0:
            model_type = self.itemData(index)
            if model_type:
                self.model_selected.emit(model_type)


# ─── Stat Card ─────────────────────────────────────────────────────────


class StatCard(QFrame):
    """Styled card widget for displaying a single statistic."""

    def __init__(self, title: str, value: str = "--", icon_text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._icon_text = icon_text
        self._setup_ui()
        self.set_value(value)

    def _setup_ui(self) -> None:
        self.setObjectName("statCard")
        self.setStyleSheet("""
            QFrame#statCard {
                background-color: #181d24;
                border: 1px solid #2b3441;
                border-radius: 8px;
                padding: 12px;
            }
            QFrame#statCard:hover {
                background-color: #1e242d;
                border-color: #3b4656;
            }
        """)
        self.setMinimumSize(140, 80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        if self._icon_text:
            self._icon_label = QLabel(self._icon_text)
            self._icon_label.setStyleSheet("font-size: 16px; color: #2563eb; font-weight: 600;")
            layout.addWidget(self._icon_label)

        self._title_label = QLabel(self._title)
        self._title_label.setStyleSheet("font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;")
        layout.addWidget(self._title_label)

        self._value_label = QLabel("--")
        self._value_label.setObjectName("valueLabel")
        self._value_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #f8fafc;")
        layout.addWidget(self._value_label)

    @Slot(str)
    def set_value(self, value: str) -> None:
        self._value_label.setText(value)

    def set_value_color(self, color: str) -> None:
        self._value_label.setStyleSheet(f"font-size: 28px; font-weight: 700; color: {color};")


# ─── Camera Preview ────────────────────────────────────────────────────


class CameraPreview(QLabel):
    """Widget that displays video frames with aspect-ratio scaling."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #3c3c3c; border-radius: 4px;")

        self._placeholder = self._create_placeholder()
        self.setPixmap(self._placeholder)

    def _create_placeholder(self) -> QPixmap:
        pixmap = QPixmap(640, 480)
        pixmap.fill(Qt.black)
        return pixmap

    @Slot(np.ndarray)
    def display_frame(self, frame: np.ndarray) -> None:
        """Display an RGB numpy array as a scaled image."""
        if frame is None or frame.size == 0:
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled = qimage.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(QPixmap.fromImage(scaled))



# ─── Video File Path Selector ──────────────────────────────────────────


class FilePathSelector(QWidget):
    """File path input with browse button."""

    path_selected = Signal(str)

    def __init__(self, title: str = "Select Video File", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._setup_ui()

    def _setup_ui(self) -> None:
        from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Browse for video file...")
        self._path_edit.setReadOnly(True)
        layout.addWidget(self._path_edit)

        self._browse_btn = QPushButton("Browse")
        self._browse_btn.setFixedWidth(100)
        self._browse_btn.clicked.connect(self._browse)
        layout.addWidget(self._browse_btn)

    def _browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, self._title,
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*.*)",
        )
        if path:
            self._path_edit.setText(path)
            self.path_selected.emit(path)

    @property
    def path(self) -> str:
        return self._path_edit.text()

    @path.setter
    def path(self, value: str) -> None:
        self._path_edit.setText(value)


# ─── Section Header ────────────────────────────────────────────────────


class SectionHeader(QFrame):
    """Reusable section header with title and optional action button."""

    def __init__(self, title: str, subtitle: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("sectionHeader")
        self.setStyleSheet("""
            QFrame#sectionHeader {
                background-color: transparent;
                border: none;
                padding: 4px 0;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(2)

        self._title_label = QLabel(title)
        self._title_label.setObjectName("headingLabel")
        layout.addWidget(self._title_label)

        if subtitle:
            self._subtitle_label = QLabel(subtitle)
            self._subtitle_label.setObjectName("subheadingLabel")
            layout.addWidget(self._subtitle_label)
