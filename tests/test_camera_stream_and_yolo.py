"""Unit and integration tests for Multi-Camera streaming, IP camera, and YOLO object detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.entities import CameraInfo, SourceDisconnectedError, SourceNotFoundError, VideoSourceType
from src.infrastructure.camera_service import OpenCVCameraService
from src.infrastructure.object_detection.yolo_detector import DetectedObject, YOLOConfig, YOLODetector
from src.infrastructure.video_source_engine import IPCameraAdapter, VideoSourceFactory, WebcamAdapter


class TestCameraServiceEnumeration:
    """Test camera enumeration, device categorization, and fallback."""

    def test_classify_device_types(self):
        classify = OpenCVCameraService._classify_device
        assert classify("Integrated Camera", 0) == "laptop"
        assert classify("FaceTime HD Camera", 0) == "laptop"
        assert classify("DroidCam Source 2", 1) == "virtual"
        assert classify("e2eSoft iVCam", 0) == "virtual"
        assert classify("Iriun Webcam", 2) == "virtual"
        assert classify("USB 2.0 Camera", 1) == "usb"
        assert classify("Logitech Webcam C920", 1) == "usb"
        assert classify("Generic Camera", 0) == "laptop"
        assert classify("Generic Camera", 3) == "usb"

    def test_list_cameras_mocked(self):
        service = OpenCVCameraService()
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.side_effect = [True, False, False, False, False, False, False]
            mock_cap.getBackendName.return_value = "DSHOW"
            mock_cap.get.side_effect = [640.0, 480.0]
            mock_cap_cls.return_value = mock_cap

            cams = service.list_cameras(force_refresh=True)
            assert len(cams) >= 1
            first_cam = cams[0]
            assert isinstance(first_cam, CameraInfo)
            assert first_cam.index == 0
            assert first_cam.is_available is True

    def test_get_camera_found(self):
        service = OpenCVCameraService()
        with patch.object(service, "list_cameras") as mock_list:
            mock_list.return_value = [
                CameraInfo(id="webcam:0", name="Front Cam", index=0, is_available=True, device_type="laptop"),
                CameraInfo(id="webcam:1", name="USB Cam", index=1, is_available=True, device_type="usb"),
            ]
            cam = service.get_camera(1)
            assert cam is not None
            assert cam.index == 1
            assert cam.device_type == "usb"

    def test_get_camera_not_found(self):
        service = OpenCVCameraService()
        with patch.object(service, "list_cameras", return_value=[]), patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap
            cam = service.get_camera(99)
            assert cam is None


class TestWebcamAdapter:
    """Test standard and external webcam adapter."""

    def test_webcam_resolution_switching(self):
        adapter = WebcamAdapter("webcam:0")
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = [640.0, 480.0, 30.0, 1280.0, 720.0]
            mock_cap_cls.return_value = mock_cap

            assert adapter.open() is True
            assert adapter.properties.width == 640

            success = adapter.set_resolution(1280, 720)
            assert success is True
            assert adapter.properties.width == 1280
            assert adapter.properties.height == 720
            adapter.close()
            assert adapter.is_open is False


class TestIPCameraAdapter:
    """Test Smartphone IP Camera adapter (Wi-Fi streaming)."""

    def test_ip_camera_url_cleaning_and_open(self):
        adapter = IPCameraAdapter("http://192.168.1.50:8080/video")
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = [1280.0, 720.0, 30.0]
            mock_cap_cls.return_value = mock_cap

            assert adapter.open() is True
            assert adapter.is_open is True
            assert adapter.properties.source_type == VideoSourceType.IP_CAMERA
            assert "192.168.1.50:8080" in adapter.properties.uri
            adapter.close()
            assert adapter.is_open is False

    def test_ip_camera_invalid_url_raises(self):
        adapter = IPCameraAdapter("invalid_stream_url")
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            with pytest.raises(SourceNotFoundError):
                adapter.open()

    def test_ip_camera_empty_url_raises(self):
        adapter = IPCameraAdapter("")
        with pytest.raises(SourceNotFoundError):
            adapter.open()

    def test_ip_camera_read_frame(self):
        adapter = IPCameraAdapter("http://192.168.1.20:8080/video")
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = [640.0, 480.0, 25.0]
            dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, dummy_bgr)
            mock_cap_cls.return_value = mock_cap

            adapter.open()
            frame = adapter.read()
            assert frame is not None
            assert frame.width == 640
            assert frame.height == 480
            assert frame.source_type == VideoSourceType.IP_CAMERA
            adapter.close()

    def test_ip_camera_disconnection_handling(self):
        adapter = IPCameraAdapter("http://192.168.1.20:8080/video")
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = [640.0, 480.0, 25.0]
            mock_cap.read.return_value = (False, None)
            mock_cap_cls.return_value = mock_cap

            adapter.open()
            # 4 failures should return None
            for _ in range(4):
                assert adapter.read() is None

            # 5th failure exceeds threshold and raises SourceDisconnectedError
            with pytest.raises(SourceDisconnectedError):
                adapter.read()

            assert adapter.is_open is False


class TestVideoSourceFactory:
    """Test registry and instantiation of IP camera and other adapters."""

    def test_factory_creates_ip_camera(self):
        source = VideoSourceFactory.create("ip_camera", "http://192.168.1.1:8080/video")
        assert isinstance(source, IPCameraAdapter)

    def test_factory_creates_phone_alias(self):
        source = VideoSourceFactory.create("phone", "http://192.168.1.1:4747/video")
        assert isinstance(source, IPCameraAdapter)

    def test_factory_creates_enum(self):
        source = VideoSourceFactory.create(VideoSourceType.IP_CAMERA, "http://192.168.1.1:8080/video")
        assert isinstance(source, IPCameraAdapter)


class TestYOLODetector:
    """Test lightweight YOLO object detector."""

    def test_yolo_detector_initialization(self):
        detector = YOLODetector()
        assert detector.is_loaded is True
        assert detector.backend in ("ultralytics", "onnx")

    def test_yolo_detection_on_dummy_frame(self):
        detector = YOLODetector()
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(dummy_rgb, confidence_threshold=0.30)
        assert isinstance(detections, list)

    def test_yolo_draw_detections(self):
        detector = YOLODetector()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_detections = [
            DetectedObject(class_id=0, label="person", confidence=0.88, box=(50, 50, 200, 300)),
            DetectedObject(class_id=67, label="cell phone", confidence=0.74, box=(250, 100, 350, 250)),
        ]
        annotated = detector.draw_detections(dummy_frame, mock_detections, show_boxes=True, show_labels=True)
        assert annotated.shape == dummy_frame.shape
        # Pixels should be modified by the drawn boxes and labels
        assert not np.array_equal(annotated, dummy_frame)

    def test_yolo_onnx_fallback_behavior(self):
        cfg = YOLOConfig(model_path="nonexistent.pt", onnx_path="data/models/yolo/yolov8n.onnx")
        detector = YOLODetector(cfg)
        assert detector.is_loaded is True
        assert detector.backend == "onnx"

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(dummy)
        assert isinstance(res, list)
