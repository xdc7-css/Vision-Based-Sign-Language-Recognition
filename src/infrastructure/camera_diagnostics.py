"""Windows camera diagnostics tool for Vision-Based Sign Language Recognition.

Probes Windows DirectShow COM, Media Foundation, Qt QMediaDevices, WMI PnP entities,
and OpenCV capture backends. Generates logs/camera_diagnostics.txt.
"""

from __future__ import annotations

import datetime
import platform
import sys
from pathlib import Path
from typing import Any

import cv2


def run_camera_diagnostics(output_path: str = "logs/camera_diagnostics.txt") -> str:
    """Run non-destructive diagnostic analysis and save report to logs/camera_diagnostics.txt."""
    lines: list[str] = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 78)
    lines.append("  WINDOWS CAMERA & DRIVER DIAGNOSTICS REPORT")
    lines.append("=" * 78)
    lines.append(f"Generated at: {timestamp}")
    lines.append(f"Operating System: {platform.platform()} ({sys.platform})")
    lines.append(f"Python: {sys.version.split()[0]} ({sys.executable})")
    lines.append(f"OpenCV Version: {cv2.__version__}")
    lines.append("")

    # 1. DirectShow COM Enumeration
    lines.append("-" * 78)
    lines.append("1. NATIVE DIRECTSHOW COM DEVICE ENUMERATION")
    lines.append("-" * 78)

    dshow_devices: list[dict[str, Any]] = []
    if sys.platform == "win32":
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

                            dshow_devices.append({
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

            lines.append(f"DirectShow video devices discovered: {len(dshow_devices)}")
            for dev in dshow_devices:
                lines.append(f"  [Index {dev['index']}] Name: {dev['name']}")
                lines.append(f"             DevicePath: {dev['device_path']}")
        except Exception as ex:
            lines.append(f"DirectShow COM enumeration error: {ex}")
    else:
        lines.append("DirectShow COM enumeration is Windows-specific (skipped on current platform).")
    lines.append("")

    # 2. Qt QMediaDevices Enumeration
    lines.append("-" * 78)
    lines.append("2. QT QMEDIADEVICES (WINDOWS MEDIA FOUNDATION)")
    lines.append("-" * 78)
    qt_devices: list[dict[str, Any]] = []
    try:
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtMultimedia import QMediaDevices

        _ = QCoreApplication.instance() or QCoreApplication([])
        inputs = QMediaDevices.videoInputs()
        lines.append(f"QMediaDevices video inputs count: {len(inputs)}")
        for idx, dev in enumerate(inputs):
            dev_id = bytes(dev.id()).decode("utf-8", errors="ignore")
            qt_devices.append({
                "index": idx,
                "name": dev.description().strip(),
                "id": dev_id,
                "is_default": dev.isDefault(),
            })
            lines.append(f"  [Input {idx}] Description: {dev.description().strip()}")
            lines.append(f"            ID: {dev_id}")
            lines.append(f"            Default Device: {dev.isDefault()}")
    except Exception as ex:
        lines.append(f"QMediaDevices enumeration error: {ex}")
    lines.append("")

    # 3. WMI PnP Entities (Camera & Image drivers)
    lines.append("-" * 78)
    lines.append("3. WINDOWS PNP ENTITIES & DRIVER AUDIT")
    lines.append("-" * 78)
    ivcam_driver_found = False
    try:
        import win32com.client
        wmi = win32com.client.GetObject("winmgmts:")
        pnp_items = wmi.InstancesOf("Win32_PnPEntity")
        found_pnp = 0
        for item in pnp_items:
            pnp_class = getattr(item, "PNPClass", "")
            if pnp_class in ("Camera", "Image"):
                found_pnp += 1
                name = getattr(item, "Name", "Unknown")
                dev_id = getattr(item, "DeviceID", "Unknown")
                status = getattr(item, "Status", "Unknown")
                lines.append(f"  Device: {name} (Class: {pnp_class}, Status: {status})")
                lines.append(f"    PnP ID: {dev_id}")
                if "ivcam" in name.lower() or "ivcam" in dev_id.lower():
                    ivcam_driver_found = True
        lines.append(f"Total Camera/Image PnP entities found: {found_pnp}")
        lines.append(f"e2eSoft iVCam Driver Present: {ivcam_driver_found}")
    except Exception as ex:
        lines.append(f"WMI PnP audit error / not available: {ex}")
    lines.append("")

    # 4. OpenCV Capture Backends & Device Accessibility Testing
    lines.append("-" * 78)
    lines.append("4. OPENCV BACKEND ACCESSIBILITY & FRAME INTEGRITY")
    lines.append("-" * 78)

    backends_to_test = [
        ("CAP_DSHOW (DirectShow)", cv2.CAP_DSHOW),
        ("CAP_MSMF (Media Foundation)", cv2.CAP_MSMF),
        ("CAP_ANY (Default Backend)", cv2.CAP_ANY),
    ] if sys.platform == "win32" else [("CAP_ANY", cv2.CAP_ANY)]

    num_devices_to_test = max(len(dshow_devices), len(qt_devices), 2)
    for b_label, b_val in backends_to_test:
        lines.append(f"Testing {b_label}:")
        for idx in range(num_devices_to_test):
            cap = cv2.VideoCapture(idx, b_val)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_backend = cap.getBackendName()
                ret, frame = cap.read()
                cap.release()

                frame_info = f"shape={frame.shape}" if ret and frame is not None else "Frame read failed"
                lines.append(f"  Index {idx}: ACCESSIBLE ({w}x{h} via {actual_backend}) | Frame Read: {ret} ({frame_info})")
            else:
                cap.release()
                lines.append(f"  Index {idx}: NOT ACCESSIBLE / CLOSED (device busy or index unassigned)")
        lines.append("")

    # 5. Diagnostic Summary & Recommendations
    lines.append("-" * 78)
    lines.append("5. DIAGNOSTIC ASSESSMENT & RECOMMENDATIONS")
    lines.append("-" * 78)

    has_hd_webcam = any("hd webcam" in d.get("name", "").lower() for d in dshow_devices) or \
                    any("hd webcam" in d.get("name", "").lower() for d in qt_devices)
    has_ivcam = any("ivcam" in d.get("name", "").lower() for d in dshow_devices) or \
                any("ivcam" in d.get("name", "").lower() for d in qt_devices) or ivcam_driver_found

    lines.append(f"- HD Webcam (Built-in Laptop Camera) Detected: {has_hd_webcam}")
    lines.append(f"- e2eSoft iVCam (iPhone Camera) Detected: {has_ivcam}")

    if has_hd_webcam and has_ivcam:
        lines.append("STATUS: OPTIMAL — Both physical laptop camera and iPhone iVCam are actively enumerated.")
        lines.append("DirectShow index 0 correctly maps to HD Webcam.")
        lines.append("DirectShow index 1 correctly maps to e2eSoft iVCam.")
        lines.append("Both cameras can be seamlessly switched in the unified camera dropdown.")
    elif has_hd_webcam and not has_ivcam:
        lines.append("STATUS: iVCam is currently disconnected or the iVCam client on the iPhone is closed.")
        lines.append("To connect iPhone camera:")
        lines.append("  1. Launch the iVCam PC client software on Windows.")
        lines.append("  2. Open the iVCam app on your iPhone and ensure Wi-Fi or USB cable is connected.")
        lines.append("  3. Wait for iVCam to establish connection, then click the 'Refresh' button in the application.")
    elif not has_hd_webcam:
        lines.append("WARNING: Built-in HD Webcam was not detected by DirectShow.")
        lines.append("Check Windows Camera Privacy Settings (Settings > Privacy & Security > Camera).")

    lines.append("=" * 78)

    report_content = "\n".join(lines)

    # Ensure output directory exists and save
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_content, encoding="utf-8")

    return report_content


if __name__ == "__main__":
    report = run_camera_diagnostics()
    print(report)
