from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities import (
    CameraInfo,
    DialectInfo,
    Frame,
    SourceProperties,
    VideoSourceType,
)


class VideoSource(ABC):
    """Unified interface for all video source types (Adapter pattern target)."""

    @abstractmethod
    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read the next frame. Returns None on failure or EOF."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by this source."""
        ...

    @abstractmethod
    def seek(self, position_seconds: float) -> bool:
        """Seek to a position (file sources only). Returns True on success."""
        ...

    @property
    @abstractmethod
    def properties(self) -> SourceProperties:
        """Return the current source properties."""
        ...

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """True if the source is open and ready."""
        ...

    @property
    @abstractmethod
    def current_fps(self) -> float:
        """Return the current streaming FPS (rolling average)."""
        ...


class CameraService(ABC):
    """Abstracts camera hardware discovery and lifecycle."""

    @abstractmethod
    def list_cameras(self) -> list[CameraInfo]:
        """Discover available cameras on the system."""
        ...

    @abstractmethod
    def get_camera(self, index: int) -> Optional[CameraInfo]:
        """Get camera info by device index."""
        ...


class ConfigRepository(ABC):
    """Abstracts application configuration persistence."""

    @abstractmethod
    def load(self) -> dict:
        """Load full configuration dictionary."""
        ...

    @abstractmethod
    def save(self, config: dict) -> None:
        """Persist configuration dictionary."""
        ...

    @abstractmethod
    def get(self, key: str, default=None):
        """Get a single config value by dot-separated key."""
        ...

    @abstractmethod
    def set(self, key: str, value) -> None:
        """Set a single config value by dot-separated key."""
        ...


class DialectRepository(ABC):
    """Abstracts dialect pack discovery and loading."""

    @abstractmethod
    def list_dialects(self) -> list[DialectInfo]:
        """List all available dialect packs."""
        ...

    @abstractmethod
    def get_dialect(self, dialect_id: str) -> Optional[DialectInfo]:
        """Get dialect info by ID."""
        ...
