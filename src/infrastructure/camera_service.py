from __future__ import annotations

from typing import Optional

import cv2

from src.domain.entities import CameraInfo
from src.domain.interfaces import CameraService


class OpenCVCameraService(CameraService):
    """Camera discovery and access via OpenCV VideoCapture."""

    MAX_CAMERAS_TO_CHECK = 10

    def __init__(self) -> None:
        self._camera_cache: Optional[list[CameraInfo]] = None

    def list_cameras(self) -> list[CameraInfo]:
        cameras: list[CameraInfo] = []

        for index in range(self.MAX_CAMERAS_TO_CHECK):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                backend_name = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                name = f"Camera {index} ({width}x{height} via {backend_name})"

                cameras.append(CameraInfo(
                    id=f"webcam:{index}",
                    name=name,
                    index=index,
                    is_available=True,
                ))
                cap.release()
            else:
                cap.release()

        if not cameras:
            cameras.append(CameraInfo(
                id="webcam:0",
                name="Default Camera",
                index=0,
                is_available=False,
            ))

        self._camera_cache = cameras
        return cameras

    def get_camera(self, index: int) -> Optional[CameraInfo]:
        cameras = self.list_cameras()
        for cam in cameras:
            if cam.index == index:
                return cam

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            info = CameraInfo(
                id=f"webcam:{index}",
                name=f"Camera {index}",
                index=index,
                is_available=True,
            )
            cap.release()
            return info

        return None
