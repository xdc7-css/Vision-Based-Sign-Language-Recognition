from __future__ import annotations

import logging
import sys
from typing import Optional

import cv2

from src.domain.entities import CameraInfo
from src.domain.interfaces import CameraService


class OpenCVCameraService(CameraService):
    """Camera discovery and classification supporting built-in, USB, and virtual phone webcams."""

    MAX_CAMERAS_TO_CHECK = 6

    def __init__(self) -> None:
        self._camera_cache: Optional[list[CameraInfo]] = None
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _classify_device(name: str, index: int) -> str:
        name_lower = name.lower()
        if any(v in name_lower for v in ("ivcam", "droidcam", "iriun", "epoccam", "obs virtual", "virtual", "camo")):
            return "virtual"
        if any(b in name_lower for b in ("integrated", "front", "built-in", "facetime", "internal", "laptop")):
            return "laptop"
        if "usb" in name_lower or "external" in name_lower:
            return "usb"
        if index == 0:
            return "laptop"
        return "usb"

    def list_cameras(self, force_refresh: bool = False) -> list[CameraInfo]:
        if self._camera_cache is not None and not force_refresh:
            return self._camera_cache

        cameras: list[CameraInfo] = []

        # 1. Try QMediaDevices for friendly names
        qt_names: dict[int, str] = {}
        try:
            from PySide6.QtMultimedia import QMediaDevices
            devices = QMediaDevices.videoInputs()
            for idx, dev in enumerate(devices):
                desc = dev.description().strip()
                if desc:
                    qt_names[idx] = desc
        except Exception:
            pass

        # 2. Check availability with OpenCV VideoCapture
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

        max_idx = max(len(qt_names), self.MAX_CAMERAS_TO_CHECK) if qt_names else self.MAX_CAMERAS_TO_CHECK
        for index in range(max_idx):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                backend_name = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                raw_name = qt_names.get(index) or f"Camera {index} ({width}x{height} via {backend_name})"
                dev_type = self._classify_device(raw_name, index)

                cameras.append(CameraInfo(
                    id=f"webcam:{index}",
                    name=raw_name,
                    index=index,
                    is_available=True,
                    device_type=dev_type,
                    description=f"{width}x{height} via {backend_name}",
                ))
                cap.release()
            else:
                cap.release()
                if index in qt_names:
                    raw_name = qt_names[index]
                    dev_type = self._classify_device(raw_name, index)
                    cameras.append(CameraInfo(
                        id=f"webcam:{index}",
                        name=f"{raw_name} (In Use / Disconnected)",
                        index=index,
                        is_available=False,
                        device_type=dev_type,
                        description="Device busy or offline",
                    ))

        if not cameras:
            cameras.append(CameraInfo(
                id="webcam:0",
                name="Default Camera",
                index=0,
                is_available=False,
                device_type="laptop",
                description="No camera detected",
            ))

        self._camera_cache = cameras
        return cameras

    def get_camera(self, index: int) -> Optional[CameraInfo]:
        cameras = self.list_cameras()
        for cam in cameras:
            if cam.index == index:
                return cam

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            info = CameraInfo(
                id=f"webcam:{index}",
                name=f"Camera {index}",
                index=index,
                is_available=True,
                device_type=self._classify_device(f"Camera {index}", index),
                description="Detected on-demand",
            )
            cap.release()
            return info

        return None
