"""Modular video source engine with Adapter pattern.

Designed for extensibility — new source types implement VideoSource
and register via VideoSourceFactory.register().
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.domain.entities import (
    Frame,
    SourceDisconnectedError,
    SourceError,
    SourceNotFoundError,
    SourceProperties,
    VideoSourceType,
)
from src.domain.interfaces import VideoSource


class FPSMonitor:
    """Rolling-window FPS calculator."""

    __slots__ = ("_timestamps", "_window_size")

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._window_size = window_size

    def tick(self) -> None:
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def average_frame_time_ms(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (elapsed / (len(self._timestamps) - 1)) * 1000 if len(self._timestamps) > 1 else 0.0

    def reset(self) -> None:
        self._timestamps.clear()


# ─── Base Video Source ─────────────────────────────────────────────────


class BaseVideoSource(VideoSource, ABC):
    """Shared base for all video source adapters."""

    def __init__(self, source_type: VideoSourceType, uri: str) -> None:
        self._source_type = source_type
        self._uri = uri
        self._is_open = False
        self._frame_id = 0
        self._fps_monitor = FPSMonitor()
        self._properties = SourceProperties(
            source_type=source_type,
            uri=uri,
            display_name=source_type.display_name,
        )

    @abstractmethod
    def open(self) -> bool:
        ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def seek(self, position_seconds: float) -> bool:
        return False

    @property
    def properties(self) -> SourceProperties:
        return self._properties

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def current_fps(self) -> float:
        return self._fps_monitor.fps

    def _make_frame(self, bgr_frame: np.ndarray, metadata: Optional[dict] = None) -> Frame:
        self._frame_id += 1
        self._fps_monitor.tick()

        if bgr_frame.shape[2] == 3:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = bgr_frame

        return Frame(
            frame_id=self._frame_id,
            timestamp=time.time(),
            source_type=self._source_type,
            source_uri=self._uri,
            data=rgb,
            width=rgb.shape[1],
            height=rgb.shape[0],
            fps=self.current_fps,
            metadata=metadata or {},
        )


# ─── Webcam Adapter ────────────────────────────────────────────────────


class WebcamAdapter(BaseVideoSource):
    """Live camera feed via OpenCV."""

    def __init__(self, uri: str) -> None:
        super().__init__(VideoSourceType.WEBCAM, uri)
        self._cap: Optional[cv2.VideoCapture] = None
        self._backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

    def set_resolution(self, width: int, height: int) -> bool:
        """Dynamically set frame resolution."""
        if self._cap is not None and self._is_open:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._properties = SourceProperties(
                source_type=self._properties.source_type,
                uri=self._properties.uri,
                display_name=self._properties.display_name,
                width=actual_w,
                height=actual_h,
                fps=self._properties.fps,
            )
            return True
        return False

    def open(self) -> bool:
        try:
            index_str = self._uri.replace("webcam:", "")
            index = int(index_str)
        except (ValueError, AttributeError):
            index = 0

        self._cap = cv2.VideoCapture(index, self._backend)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(index)
            if not self._cap.isOpened():
                raise SourceNotFoundError(self._uri, VideoSourceType.WEBCAM)

        # Use buffer size 1 to prevent latency accumulation
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0

        self._properties = SourceProperties(
            source_type=VideoSourceType.WEBCAM,
            uri=self._uri,
            display_name=f"Webcam {index}",
            width=w,
            height=h,
            fps=fps,
        )

        self._is_open = True
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None or not self._is_open:
            return None

        ret, frame = self._cap.read()
        if not ret:
            self._is_open = False
            raise SourceDisconnectedError(source_type=VideoSourceType.WEBCAM)

        return self._make_frame(frame)

    def close(self) -> None:
        self._is_open = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._fps_monitor.reset()


# ─── IP Camera / Smartphone Stream Adapter ─────────────────────────────


class IPCameraAdapter(BaseVideoSource):
    """Network IP Camera and Smartphone Stream Adapter (HTTP, MJPEG, RTSP)."""

    def __init__(self, uri: str) -> None:
        cleaned_uri = uri
        for prefix in ("ip_camera:", "stream:", "phone:"):
            if cleaned_uri.startswith(prefix):
                cleaned_uri = cleaned_uri[len(prefix):]
                break
        super().__init__(VideoSourceType.IP_CAMERA, cleaned_uri)
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0
        self._max_failures = 5

    def open(self) -> bool:
        raw_url = self._uri.strip()
        if not raw_url:
            raise SourceNotFoundError(self._uri, VideoSourceType.IP_CAMERA)

        if not (raw_url.startswith("http://") or raw_url.startswith("https://") or raw_url.startswith("rtsp://")):
            raw_url = "http://" + raw_url

        self._cap = cv2.VideoCapture(raw_url)
        if not self._cap.isOpened():
            raise SourceNotFoundError(raw_url, VideoSourceType.IP_CAMERA)

        # Minimize buffering for lowest live stream delay
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0

        self._properties = SourceProperties(
            source_type=VideoSourceType.IP_CAMERA,
            uri=raw_url,
            display_name=f"IP Camera ({raw_url})",
            width=w,
            height=h,
            fps=fps,
        )

        self._consecutive_failures = 0
        self._is_open = True
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None or not self._is_open:
            return None

        ret, frame = self._cap.read()
        if not ret:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_failures:
                self._is_open = False
                raise SourceDisconnectedError(
                    "Phone camera / IP stream disconnected or lost signal",
                    source_type=VideoSourceType.IP_CAMERA,
                )
            return None

        self._consecutive_failures = 0
        return self._make_frame(frame)

    def close(self) -> None:
        self._is_open = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._fps_monitor.reset()


# ─── Video File Adapter ────────────────────────────────────────────────


class FileAdapter(BaseVideoSource):
    """Video file playback via OpenCV."""

    def __init__(self, uri: str) -> None:
        super().__init__(VideoSourceType.FILE, uri)
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames: int = 0
        self._duration: float = 0.0

    def open(self) -> bool:
        path = Path(self._uri)
        if not path.exists():
            raise SourceNotFoundError(self._uri, VideoSourceType.FILE)

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise SourceError(f"Failed to open video file: {path}", VideoSourceType.FILE)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._total_frames / fps if fps > 0 else 0.0

        self._properties = SourceProperties(
            source_type=VideoSourceType.FILE,
            uri=self._uri,
            display_name=path.name,
            width=w,
            height=h,
            fps=fps,
            total_frames=self._total_frames,
            duration_seconds=self._duration,
            is_seekable=True,
        )

        self._is_open = True
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None or not self._is_open:
            return None

        ret, frame = self._cap.read()
        if not ret:
            self.close()
            return None

        return self._make_frame(frame, {"position": self._cap.get(cv2.CAP_PROP_POS_FRAMES)})

    def seek(self, position_seconds: float) -> bool:
        if self._cap is None:
            return False
        target_frame = int(position_seconds * self._properties.fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self._fps_monitor.reset()
        return True

    def close(self) -> None:
        self._is_open = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._fps_monitor.reset()


# ─── Screen Capture Adapter ────────────────────────────────────────────


class ScreenCaptureAdapter(BaseVideoSource):
    """Screen capture via MSS (cross-platform). Falls back to PIL."""

    def __init__(self, uri: str) -> None:
        super().__init__(VideoSourceType.SCREEN, uri)
        self._mss: Any = None
        self._monitor: Any = None
        self._monitor_index: int = 0

    def open(self) -> bool:
        try:
            import mss
            self._mss = mss.mss()
            monitors = self._mss.monitors

            if self._uri.startswith("screen:"):
                try:
                    self._monitor_index = min(int(self._uri.split(":")[1]), len(monitors) - 1)
                except (ValueError, IndexError):
                    self._monitor_index = 0
            else:
                self._monitor_index = 0

            self._monitor = monitors[self._monitor_index]
            w = self._monitor["width"]
            h = self._monitor["height"]

            self._properties = SourceProperties(
                source_type=VideoSourceType.SCREEN,
                uri=self._uri,
                display_name=f"Screen {self._monitor_index}",
                width=w,
                height=h,
                fps=30.0,
            )

            self._is_open = True
            return True

        except ImportError:
            return self._open_fallback()

    def _open_fallback(self) -> bool:
        try:
            from PIL import ImageGrab

            self._mss = ImageGrab
            w, h = self._mss.grab().size

            self._properties = SourceProperties(
                source_type=VideoSourceType.SCREEN,
                uri=self._uri,
                display_name="Screen (PIL fallback)",
                width=w,
                height=h,
                fps=15.0,
            )

            self._is_open = True
            return True

        except ImportError:

            raise SourceError(
                "Screen capture requires 'mss' or 'Pillow' package. "
                "Install with: pip install mss",
                VideoSourceType.SCREEN,
            )

    def read(self) -> Optional[Frame]:
        if not self._is_open:
            return None

        try:
            if hasattr(self._mss, "grab"):
                screenshot = self._mss.grab(self._monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return self._make_frame(img)
            else:
                pil_img = self._mss.grab()
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return self._make_frame(img)

        except Exception as e:
            raise SourceError(f"Screen capture error: {e}", VideoSourceType.SCREEN, recoverable=True)

    def close(self) -> None:
        self._is_open = False
        if self._mss is not None and hasattr(self._mss, "close"):
            try:
                self._mss.close()
            except Exception:
                pass
        self._mss = None
        self._fps_monitor.reset()


# ─── Window Capture Adapter ────────────────────────────────────────────


class WindowCaptureAdapter(BaseVideoSource):
    """Capture a specific window by title via win32gui + MSS."""

    def __init__(self, uri: str) -> None:
        super().__init__(VideoSourceType.WINDOW, uri)
        self._mss: Any = None
        self._window_title: str = ""
        self._window_handle: Any = None

    def open(self) -> bool:
        self._window_title = self._uri.replace("window:", "", 1).strip()

        try:
            import win32gui

            def enum_callback(hwnd, results):
                if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                    results.append((hwnd, win32gui.GetWindowText(hwnd)))

            windows: list[tuple[Any, str]] = []
            win32gui.EnumWindows(enum_callback, windows)

            for hwnd, title in windows:
                if self._window_title.lower() in title.lower():
                    self._window_handle = hwnd
                    break

            if self._window_handle is None:
                raise SourceNotFoundError(
                    f"No visible window matching '{self._window_title}'",
                    VideoSourceType.WINDOW,
                )

        except ImportError:
            raise SourceError(
                "Window capture requires 'pywin32'. Install with: pip install pywin32",
                VideoSourceType.WINDOW,
            )

        try:
            import mss
            self._mss = mss.mss()
        except ImportError:
            raise SourceError(
                "Window capture requires 'mss'. Install with: pip install mss",
                VideoSourceType.WINDOW,
            )

        import win32gui
        rect = win32gui.GetWindowRect(self._window_handle)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]

        self._properties = SourceProperties(
            source_type=VideoSourceType.WINDOW,
            uri=self._uri,
            display_name=f"Window: {self._window_title}",
            width=w,
            height=h,
            fps=15.0,
        )

        self._is_open = True
        return True

    def read(self) -> Optional[Frame]:
        if not self._is_open:
            return None

        try:
            import win32gui

            if not win32gui.IsWindow(self._window_handle):
                self._is_open = False
                raise SourceDisconnectedError("Window closed", VideoSourceType.WINDOW)

            rect = win32gui.GetWindowRect(self._window_handle)
            monitor = {
                "top": rect[1],
                "left": rect[0],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
            }

            screenshot = self._mss.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            return self._make_frame(img)

        except ImportError:
            raise
        except Exception as e:
            raise SourceError(f"Window capture error: {e}", VideoSourceType.WINDOW, recoverable=True)

    def close(self) -> None:
        self._is_open = False
        self._window_handle = None
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None
        self._fps_monitor.reset()


# ─── Factory / Registry ────────────────────────────────────────────────


class VideoSourceFactory:
    """Registry-based factory for creating VideoSource adapters.

    New source types register themselves via register() and can be
    instantiated by source_type string or VideoSourceType enum.
    """

    _registry: dict[str, type[BaseVideoSource]] = {}

    @classmethod
    def register(cls, source_type: str, adapter_class: type[BaseVideoSource]) -> None:
        cls._registry[source_type.lower()] = adapter_class

    @classmethod
    def create(cls, source_type: VideoSourceType | str, uri: str) -> BaseVideoSource:
        if isinstance(source_type, VideoSourceType):
            type_str = source_type.name.lower()
        else:
            type_str = source_type.lower()

        adapter_class = cls._registry.get(type_str)
        if adapter_class is None:
            raise ValueError(f"No adapter registered for source type: {type_str}. "
                             f"Available: {list(cls._registry.keys())}")

        return adapter_class(uri)

    @classmethod
    def list_available_types(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def list_adapters(cls) -> list[type[BaseVideoSource]]:
        return list(cls._registry.values())


# ─── Register built-in adapters ────────────────────────────────────────

VideoSourceFactory.register("webcam", WebcamAdapter)
VideoSourceFactory.register("file", FileAdapter)
VideoSourceFactory.register("screen", ScreenCaptureAdapter)
VideoSourceFactory.register("window", WindowCaptureAdapter)
VideoSourceFactory.register("ip_camera", IPCameraAdapter)
VideoSourceFactory.register("stream", IPCameraAdapter)
VideoSourceFactory.register("phone", IPCameraAdapter)
