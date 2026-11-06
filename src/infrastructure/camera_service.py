from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import cv2

from src.domain.entities import CameraInfo
from src.domain.interfaces import CameraService


class OpenCVCameraService(CameraService):
    """Reliable camera discovery and classification supporting built-in, USB, and virtual phone webcams.

    On Windows, uses native DirectShow COM enumeration to guarantee 100% accurate device naming
    and index mapping corresponding to OpenCV's CAP_DSHOW backend.
    """

    MAX_CAMERAS_TO_CHECK = 4

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

    @staticmethod
    def _enumerate_dshow_devices_windows() -> list[dict[str, Any]]:
        """DirectShow COM device enumeration for Windows (exact 1:1 mapping with OpenCV CAP_DSHOW)."""
        devices: list[dict[str, Any]] = []
        if sys.platform != "win32":
            return devices

        try:
            import ctypes
            from ctypes import POINTER, byref, c_void_p, wintypes

            ole32 = ctypes.oledll.ole32
            oleaut32 = ctypes.oledll.oleaut32
            ole32.CoInitialize(None)

            class GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", wintypes.BYTE * 8),
                ]

                def __init__(self, d1, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8):
                    super().__init__(d1, w1, w2, (wintypes.BYTE * 8)(b1, b2, b3, b4, b5, b6, b7, b8))

            CLSID_SystemDeviceEnum = GUID(
                0x62BE5D10, 0x60EB, 0x11D0, 0xBD, 0x3B, 0x00, 0xA0, 0xC9, 0x11, 0xCE, 0x86
            )
            IID_ICreateDevEnum = GUID(
                0x29840822, 0x5B84, 0x11D0, 0xBD, 0x3B, 0x00, 0xA0, 0xC9, 0x11, 0xCE, 0x86
            )
            CLSID_VideoInputDeviceCategory = GUID(
                0x860BB310, 0x5D01, 0x11D0, 0xBD, 0x3B, 0x00, 0xA0, 0xC9, 0x11, 0xCE, 0x86
            )
            IID_IPropertyBag = GUID(
                0x55272A00, 0x42CB, 0x11CE, 0x81, 0x35, 0x00, 0xAA, 0x00, 0x4B, 0xB8, 0x51
            )

            pDevEnum = c_void_p()
            hr = ole32.CoCreateInstance(
                byref(CLSID_SystemDeviceEnum), None, 1, byref(IID_ICreateDevEnum), byref(pDevEnum)
            )
            if hr == 0 and pDevEnum:
                vtbl = ctypes.cast(ctypes.cast(pDevEnum, POINTER(c_void_p)).contents, POINTER(c_void_p))
                CreateClassEnumerator = ctypes.WINFUNCTYPE(
                    ctypes.c_long, c_void_p, POINTER(GUID), POINTER(c_void_p), wintypes.DWORD
                )(vtbl[3])
                pEnum = c_void_p()
                hr2 = CreateClassEnumerator(pDevEnum, byref(CLSID_VideoInputDeviceCategory), byref(pEnum), 0)
                if hr2 == 0 and pEnum:
                    enum_vtbl = ctypes.cast(ctypes.cast(pEnum, POINTER(c_void_p)).contents, POINTER(c_void_p))
                    Next = ctypes.WINFUNCTYPE(
                        ctypes.c_long, c_void_p, wintypes.ULONG, POINTER(c_void_p), POINTER(wintypes.ULONG)
                    )(enum_vtbl[3])

                    idx = 0
                    while True:
                        pMoniker = c_void_p()
                        fetched = wintypes.ULONG(0)
                        if Next(pEnum, 1, byref(pMoniker), byref(fetched)) != 0 or fetched.value == 0:
                            break

                        mon_vtbl = ctypes.cast(ctypes.cast(pMoniker, POINTER(c_void_p)).contents, POINTER(c_void_p))
                        BindToStorage = ctypes.WINFUNCTYPE(
                            ctypes.c_long, c_void_p, c_void_p, c_void_p, POINTER(GUID), POINTER(c_void_p)
                        )(mon_vtbl[9])
                        pPropBag = c_void_p()
                        if BindToStorage(pMoniker, None, None, byref(IID_IPropertyBag), byref(pPropBag)) == 0 and pPropBag:
                            prop_vtbl = ctypes.cast(
                                ctypes.cast(pPropBag, POINTER(c_void_p)).contents, POINTER(c_void_p)
                            )

                            class VARIANT(ctypes.Structure):
                                _fields_ = [
                                    ("vt", wintypes.WORD),
                                    ("wReserved1", wintypes.WORD),
                                    ("wReserved2", wintypes.WORD),
                                    ("wReserved3", wintypes.WORD),
                                    ("bstrVal", wintypes.LPWSTR),
                                    ("dummy", wintypes.BYTE * 8),
                                ]

                            Read = ctypes.WINFUNCTYPE(
                                ctypes.c_long, c_void_p, wintypes.LPCWSTR, POINTER(VARIANT), c_void_p
                            )(prop_vtbl[3])

                            var_name = VARIANT()
                            var_name.vt = 8
                            oleaut32.VariantInit(byref(var_name))
                            name_hr = Read(pPropBag, "FriendlyName", byref(var_name), None)
                            friendly_name = str(var_name.bstrVal) if name_hr == 0 else f"Camera {idx}"
                            oleaut32.VariantClear(byref(var_name))

                            var_path = VARIANT()
                            var_path.vt = 8
                            oleaut32.VariantInit(byref(var_path))
                            path_hr = Read(pPropBag, "DevicePath", byref(var_path), None)
                            device_path = str(var_path.bstrVal) if path_hr == 0 else ""
                            oleaut32.VariantClear(byref(var_path))

                            devices.append({
                                "index": idx,
                                "name": friendly_name,
                                "device_path": device_path,
                            })

                            ReleaseProp = ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)(prop_vtbl[2])
                            ReleaseProp(pPropBag)

                        ReleaseMon = ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)(mon_vtbl[2])
                        ReleaseMon(pMoniker)
                        idx += 1

                    ReleaseEnum = ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)(enum_vtbl[2])
                    ReleaseEnum(pEnum)

                ReleaseDev = ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)(vtbl[2])
                ReleaseDev(pDevEnum)
        except Exception:
            pass

        return devices

    def list_cameras(
        self,
        force_refresh: bool = False,
        active_index: Optional[int] = None,
    ) -> list[CameraInfo]:
        """Discover available cameras on the system without dropping or hiding valid devices."""
        if self._camera_cache is not None and not force_refresh:
            return self._camera_cache

        cameras: list[CameraInfo] = []
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

        # 1. Native DirectShow COM Enumeration (Windows primary)
        dshow_list = self._enumerate_dshow_devices_windows()

        if dshow_list:
            for dev in dshow_list:
                idx = dev["index"]
                raw_name = dev["name"]
                dev_path = dev["device_path"]
                dev_type = self._classify_device(raw_name, idx)

                # If this camera is currently active in the application, keep it available
                if active_index is not None and idx == active_index:
                    cameras.append(CameraInfo(
                        id=f"webcam:{idx}",
                        name=raw_name,
                        index=idx,
                        is_available=True,
                        device_type=dev_type,
                        description="Currently Active",
                        device_path=dev_path,
                    ))
                    continue

                # Verify accessibility
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    bname = cap.getBackendName()
                    cap.release()

                    cameras.append(CameraInfo(
                        id=f"webcam:{idx}",
                        name=raw_name,
                        index=idx,
                        is_available=True,
                        device_type=dev_type,
                        description=f"{w}x{h} via {bname}",
                        device_path=dev_path,
                    ))
                else:
                    cap.release()
                    # Do not hide camera if temporarily busy or in another backend
                    cameras.append(CameraInfo(
                        id=f"webcam:{idx}",
                        name=raw_name,
                        index=idx,
                        is_available=True,
                        device_type=dev_type,
                        description="DirectShow Video Device",
                        device_path=dev_path,
                    ))

            if cameras:
                self._camera_cache = cameras
                return cameras

        # 2. Qt QMediaDevices fallback (cross-platform / WMF)
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

        # 3. OpenCV sequential probing fallback
        max_idx = max(len(qt_names), self.MAX_CAMERAS_TO_CHECK) if qt_names else self.MAX_CAMERAS_TO_CHECK
        for index in range(max_idx):
            if active_index is not None and index == active_index:
                raw_name = qt_names.get(index) or f"Camera {index}"
                dev_type = self._classify_device(raw_name, index)
                cameras.append(CameraInfo(
                    id=f"webcam:{index}",
                    name=raw_name,
                    index=index,
                    is_available=True,
                    device_type=dev_type,
                    description="Currently Active",
                ))
                continue

            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                backend_name = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                raw_name = qt_names.get(index) or f"Camera {index}"
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
                        name=raw_name,
                        index=index,
                        is_available=True,
                        device_type=dev_type,
                        description="Detected Video Device",
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
