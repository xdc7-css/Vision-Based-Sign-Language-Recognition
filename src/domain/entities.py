from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class VideoSourceType(Enum):
    WEBCAM = auto()
    FILE = auto()
    SCREEN = auto()
    WINDOW = auto()
    IP_CAMERA = auto()

    @property
    def display_name(self) -> str:
        return {
            VideoSourceType.WEBCAM: "Webcam",
            VideoSourceType.FILE: "Video File",
            VideoSourceType.SCREEN: "Screen Capture",
            VideoSourceType.WINDOW: "Window Capture",
            VideoSourceType.IP_CAMERA: "Phone / IP Camera",
        }[self]

    @staticmethod
    def from_string(value: str) -> VideoSourceType:
        mapping = {
            "webcam": VideoSourceType.WEBCAM,
            "file": VideoSourceType.FILE,
            "screen": VideoSourceType.SCREEN,
            "window": VideoSourceType.WINDOW,
            "ip_camera": VideoSourceType.IP_CAMERA,
            "phone": VideoSourceType.IP_CAMERA,
            "stream": VideoSourceType.IP_CAMERA,
            "ip": VideoSourceType.IP_CAMERA,
        }
        result = mapping.get(value.lower())
        if result is None:
            raise ValueError(f"Unknown source type: {value}")
        return result


@dataclass(frozen=True)
class CameraInfo:
    id: str
    name: str
    index: int
    is_available: bool = True
    device_type: str = "webcam"  # "laptop", "usb", "virtual", "ip_camera"
    description: str = ""


@dataclass(frozen=True)
class SourceProperties:
    source_type: VideoSourceType
    uri: str
    display_name: str
    width: int = 640
    height: int = 480
    fps: float = 0.0
    total_frames: int = 0
    duration_seconds: float = 0.0
    is_seekable: bool = False


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    source_type: VideoSourceType
    source_uri: str
    data: Any
    width: int
    height: int
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DialectInfo:
    dialect_id: str
    name: str
    name_ar: str
    version: str = "1.0.0"
    has_static: bool = False
    has_dynamic: bool = False
    has_alphabet: bool = False
    vocabulary_size: int = 0

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.name_ar})"


@dataclass(frozen=True)
class ModelInfo:
    dialect_id: str
    model_type: str
    version: str
    file_path: str
    is_loaded: bool = False
    accuracy: float = 0.0
    input_shape: tuple[int, ...] = ()



class SourceError(Exception):
    def __init__(self, message: str, source_type: VideoSourceType | None = None, recoverable: bool = False) -> None:
        self.source_type = source_type
        self.recoverable = recoverable
        super().__init__(message)


class SourceDisconnectedError(SourceError):
    def __init__(self, message: str = "Video source disconnected", source_type: VideoSourceType | None = None) -> None:
        super().__init__(message, source_type=source_type, recoverable=True)


class SourceNotFoundError(SourceError):
    def __init__(self, uri: str, source_type: VideoSourceType | None = None) -> None:
        self.uri = uri
        super().__init__(f"Video source not found: {uri}", source_type=source_type, recoverable=False)


class SourcePermissionError(SourceError):
    def __init__(self, message: str = "Permission denied for video source", source_type: VideoSourceType | None = None) -> None:
        super().__init__(message, source_type=source_type, recoverable=False)
