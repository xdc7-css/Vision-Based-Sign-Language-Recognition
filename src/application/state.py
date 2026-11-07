from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Signal

from src.domain.entities import (
    CameraInfo,
    DialectInfo,
    ModelInfo,
    SourceProperties,
)


class AppState(QObject):
    """Central application state with Qt signals for reactive UI updates."""

    camera_list_changed = Signal(list)
    active_source_changed = Signal(object)
    dialect_list_changed = Signal(list)
    active_dialect_changed = Signal(object)
    model_list_changed = Signal(list)
    active_model_changed = Signal(object)
    pipeline_status_changed = Signal(str)
    fps_updated = Signal(float)
    frame_received = Signal(object)
    caption_updated = Signal(str)
    gesture_detected = Signal(str, float)
    log_message = Signal(str, str)
    overlay_visibility_changed = Signal(bool)
    error_occurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cameras: list[CameraInfo] = []
        self._active_source: Optional[SourceProperties] = None
        self._dialects: list[DialectInfo] = []
        self._active_dialect: Optional[DialectInfo] = None
        self._models: list[ModelInfo] = []
        self._active_model: Optional[ModelInfo] = None
        self._pipeline_status: str = "idle"
        self._current_fps: float = 0.0
        self._overlay_visible: bool = False
        self._last_caption: str = ""
        self._selected_camera_index: Optional[int] = None

    @property
    def cameras(self) -> list[CameraInfo]:
        return list(self._cameras)

    @cameras.setter
    def cameras(self, value: list[CameraInfo]) -> None:
        self._cameras = list(value)
        self.camera_list_changed.emit(self._cameras)

    @property
    def active_source(self) -> Optional[SourceProperties]:
        return self._active_source

    @active_source.setter
    def active_source(self, value: Optional[SourceProperties]) -> None:
        self._active_source = value
        self.active_source_changed.emit(value)

    @property
    def dialects(self) -> list[DialectInfo]:
        return list(self._dialects)

    @dialects.setter
    def dialects(self, value: list[DialectInfo]) -> None:
        self._dialects = list(value)
        self.dialect_list_changed.emit(self._dialects)

    @property
    def active_dialect(self) -> Optional[DialectInfo]:
        return self._active_dialect

    @active_dialect.setter
    def active_dialect(self, value: Optional[DialectInfo]) -> None:
        self._active_dialect = value
        self.active_dialect_changed.emit(value)

    @property
    def models(self) -> list[ModelInfo]:
        return list(self._models)

    @models.setter
    def models(self, value: list[ModelInfo]) -> None:
        self._models = list(value)
        self.model_list_changed.emit(self._models)

    @property
    def active_model(self) -> Optional[ModelInfo]:
        return self._active_model

    @active_model.setter
    def active_model(self, value: Optional[ModelInfo]) -> None:
        self._active_model = value
        self.active_model_changed.emit(value)

    @property
    def pipeline_status(self) -> str:
        return self._pipeline_status

    @pipeline_status.setter
    def pipeline_status(self, value: str) -> None:
        self._pipeline_status = value
        self.pipeline_status_changed.emit(value)

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @current_fps.setter
    def current_fps(self, value: float) -> None:
        self._current_fps = value
        self.fps_updated.emit(value)

    @property
    def overlay_visible(self) -> bool:
        return self._overlay_visible

    @overlay_visible.setter
    def overlay_visible(self, value: bool) -> None:
        self._overlay_visible = value
        self.overlay_visibility_changed.emit(value)

    @property
    def last_caption(self) -> str:
        return self._last_caption

    @last_caption.setter
    def last_caption(self, value: str) -> None:
        self._last_caption = value
        self.caption_updated.emit(value)

    @property
    def selected_camera_index(self) -> Optional[int]:
        return self._selected_camera_index

    @selected_camera_index.setter
    def selected_camera_index(self, value: Optional[int]) -> None:
        self._selected_camera_index = value

    def log(self, level: str, message: str) -> None:
        self.log_message.emit(level, message)

    def error(self, message: str) -> None:
        self.error_occurred.emit(message)
        self.log("error", message)
