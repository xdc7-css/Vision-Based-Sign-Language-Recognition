"""Comprehensive tests for unified camera selection, live camera switching, and diagnostics.

Covers Scenarios A through G as specified in the Windows Camera Integration requirements:
- Scenario A: iVCam disconnected, only HD Webcam available.
- Scenario B: Both HD Webcam and e2eSoft iVCam appear simultaneously in ONE dropdown.
- Scenario C: Selecting HD Webcam opens laptop camera.
- Scenario D: Selecting e2eSoft iVCam opens iPhone camera.
- Scenario E: Rapid repeated switching between devices without resource leaks.
- Scenario F: Dynamic disconnect/reconnect and refresh.
- Scenario G: AI inference (Sign Language Hand Detection) on frames from each camera.
- Scenario H: Windows diagnostics generator output verification.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.domain.entities import CameraInfo, VideoSourceType
from src.infrastructure.camera_diagnostics import run_camera_diagnostics
from src.infrastructure.camera_service import OpenCVCameraService
from src.infrastructure.hand_detection.detector import HandDetector
from src.infrastructure.video_source_engine import WebcamAdapter


class TestCameraDisplayLabel:
    """Tests proper friendly name formatting for unified dropdown entries."""

    def test_hd_webcam_display_label(self):
        info = CameraInfo(id="webcam:0", name="HD Webcam", index=0)
        assert info.display_label == "HD Webcam (Built-in Laptop Camera)"

    def test_ivcam_display_label(self):
        info = CameraInfo(id="webcam:1", name="e2eSoft iVCam", index=1, device_type="virtual")
        assert info.display_label == "e2eSoft iVCam (iPhone Camera)"

    def test_external_usb_display_label(self):
        info = CameraInfo(id="webcam:2", name="USB 2.0 HD UVC", index=2, device_type="usb")
        assert info.display_label == "USB 2.0 HD UVC (USB Camera)"

    def test_offline_camera_display_label(self):
        info = CameraInfo(id="webcam:1", name="e2eSoft iVCam", index=1, is_available=False)
        assert "[Offline]" in info.display_label


class TestUnifiedCameraScenarios:
    """Simulated and architectural scenario tests."""

    def test_scenario_a_ivcam_disconnected(self):
        """Scenario A: Only HD Webcam is exposed by Windows."""
        service = OpenCVCameraService()
        mock_dshow = [
            {"index": 0, "name": "HD Webcam", "device_path": "path_0"},
        ]
        with patch.object(service, "_enumerate_dshow_devices_windows", return_value=mock_dshow), \
             patch("cv2.VideoCapture") as mock_cap_cls:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = [640.0, 480.0]
            cap.getBackendName.return_value = "DSHOW"
            mock_cap_cls.return_value = cap

            cams = service.list_cameras(force_refresh=True)
            assert len(cams) == 1
            assert cams[0].name == "HD Webcam"
            assert cams[0].index == 0
            assert cams[0].display_label == "HD Webcam (Built-in Laptop Camera)"

    def test_scenario_b_ivcam_connected_both_in_one_dropdown(self):
        """Scenario B: Both HD Webcam and e2eSoft iVCam are present simultaneously."""
        service = OpenCVCameraService()
        mock_dshow = [
            {"index": 0, "name": "HD Webcam", "device_path": "usb#vid_5986"},
            {"index": 1, "name": "e2eSoft iVCam", "device_path": "root#image#0000"},
        ]
        with patch.object(service, "_enumerate_dshow_devices_windows", return_value=mock_dshow), \
             patch("cv2.VideoCapture") as mock_cap_cls:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = [640.0, 480.0, 640.0, 480.0]
            cap.getBackendName.return_value = "DSHOW"
            mock_cap_cls.return_value = cap

            cams = service.list_cameras(force_refresh=True)
            assert len(cams) == 2
            names = [c.name for c in cams]
            labels = [c.display_label for c in cams]

            assert "HD Webcam" in names
            assert "e2eSoft iVCam" in names
            assert "HD Webcam (Built-in Laptop Camera)" in labels
            assert "e2eSoft iVCam (iPhone Camera)" in labels

    def test_scenario_c_select_hd_webcam(self):
        """Scenario C: Select HD Webcam (index 0) and open video capture."""
        adapter = WebcamAdapter("webcam:0")
        with patch("cv2.VideoCapture") as mock_cap_cls:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = [640.0, 480.0, 30.0]
            mock_cap_cls.return_value = cap

            opened = adapter.open()
            assert opened is True
            assert adapter.is_open is True
            assert adapter.properties.uri == "webcam:0"
            mock_cap_cls.assert_called_with(0, adapter._backend)
            adapter.close()

    def test_scenario_d_select_ivcam(self):
        """Scenario D: Select e2eSoft iVCam (index 1) and open video capture."""
        adapter = WebcamAdapter("webcam:1")
        with patch("cv2.VideoCapture") as mock_cap_cls:
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = [640.0, 480.0, 30.0]
            mock_cap_cls.return_value = cap

            opened = adapter.open()
            assert opened is True
            assert adapter.is_open is True
            assert adapter.properties.uri == "webcam:1"
            mock_cap_cls.assert_called_with(1, adapter._backend)
            adapter.close()

    def test_scenario_e_repeated_camera_switching(self):
        """Scenario E: Rapid switching between camera 0 and camera 1."""
        state = AppState()
        use_cases = VideoSourceUseCases(state)

        with patch("cv2.VideoCapture") as mock_cap_cls:
            cap0 = MagicMock()
            cap0.isOpened.return_value = True
            cap0.get.side_effect = [640.0, 480.0, 30.0] * 5

            cap1 = MagicMock()
            cap1.isOpened.return_value = True
            cap1.get.side_effect = [640.0, 480.0, 30.0] * 5

            mock_cap_cls.side_effect = [cap0, cap1, cap0, cap1]

            # Switch 1: Open camera 0 (HD Webcam)
            assert use_cases.start_source(VideoSourceType.WEBCAM, "webcam:0") is True
            assert use_cases.active_source.properties.uri == "webcam:0"

            # Switch 2: Switch to camera 1 (iVCam)
            assert use_cases.start_source(VideoSourceType.WEBCAM, "webcam:1") is True
            assert use_cases.active_source.properties.uri == "webcam:1"
            cap0.release.assert_called()

            # Switch 3: Switch back to camera 0 (HD Webcam)
            assert use_cases.start_source(VideoSourceType.WEBCAM, "webcam:0") is True
            assert use_cases.active_source.properties.uri == "webcam:0"
            cap1.release.assert_called()

            use_cases.stop_source()

    def test_scenario_f_disconnect_and_reconnect_ivcam(self):
        """Scenario F: Disconnect iVCam -> Refresh -> Reconnect iVCam -> Refresh."""
        service = OpenCVCameraService()

        # Step 1: Disconnected state (only HD Webcam)
        mock_dshow_disconnected = [
            {"index": 0, "name": "HD Webcam", "device_path": "path_0"},
        ]
        with patch.object(service, "_enumerate_dshow_devices_windows", return_value=mock_dshow_disconnected), \
             patch("cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.side_effect = [640.0, 480.0]
            mock_cap.return_value.getBackendName.return_value = "DSHOW"

            cams1 = service.list_cameras(force_refresh=True)
            assert len(cams1) == 1
            assert cams1[0].name == "HD Webcam"

        # Step 2: Reconnected state (both HD Webcam & iVCam)
        mock_dshow_reconnected = [
            {"index": 0, "name": "HD Webcam", "device_path": "path_0"},
            {"index": 1, "name": "e2eSoft iVCam", "device_path": "path_1"},
        ]
        with patch.object(service, "_enumerate_dshow_devices_windows", return_value=mock_dshow_reconnected), \
             patch("cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.side_effect = [640.0, 480.0, 640.0, 480.0]
            mock_cap.return_value.getBackendName.return_value = "DSHOW"

            cams2 = service.list_cameras(force_refresh=True)
            assert len(cams2) == 2
            assert any(c.name == "HD Webcam" for c in cams2)
            assert any(c.name == "e2eSoft iVCam" for c in cams2)

    def test_scenario_g_ai_inference_on_both_cameras(self):
        """Scenario G: Sign language hand tracking pipeline processes frames from either camera source."""
        detector = HandDetector()

        # Test frame from camera 0
        frame_cam0 = np.full((480, 640, 3), 100, dtype=np.uint8)
        res0 = detector.detect(frame_cam0)
        assert res0 is not None
        assert res0.image_width == 640
        assert res0.image_height == 480

        # Test frame from camera 1
        frame_cam1 = np.full((480, 640, 3), 180, dtype=np.uint8)
        res1 = detector.detect(frame_cam1)
        assert res1 is not None
        assert res1.image_width == 640
        assert res1.image_height == 480
        detector.close()


class TestCameraDiagnosticsTool:
    """Verifies that the diagnostic generator produces valid structured outputs."""

    def test_diagnostics_report_generation(self, tmp_path):
        report_path = tmp_path / "camera_diag.txt"
        report_str = run_camera_diagnostics(output_path=str(report_path))

        assert report_path.exists()
        assert "WINDOWS CAMERA & DRIVER DIAGNOSTICS REPORT" in report_str
        assert "DIRECTSHOW COM DEVICE ENUMERATION" in report_str
        assert "OPENCV BACKEND ACCESSIBILITY" in report_str
        assert "DIAGNOSTIC ASSESSMENT & RECOMMENDATIONS" in report_str


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DirectShow COM hardware test")
class TestRealWindowsHardwareEnumeration:
    """Real Windows hardware verification (non-mocked)."""

    def test_real_windows_directshow_enumeration(self):
        service = OpenCVCameraService()
        dshow_devs = service._enumerate_dshow_devices_windows()
        assert isinstance(dshow_devs, list)
        # On this machine, both HD Webcam and e2eSoft iVCam are currently registered
        if len(dshow_devs) >= 2:
            names = [d["name"] for d in dshow_devs]
            assert any("hd webcam" in n.lower() for n in names)
            assert any("ivcam" in n.lower() for n in names)
