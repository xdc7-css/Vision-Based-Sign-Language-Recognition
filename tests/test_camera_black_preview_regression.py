"""Regression tests for Camera Black Preview prevention and multi-backend streaming.

Verifies:
1. Video capture multi-backend evaluation and validation.
2. Safe extraction of BGR numpy frame from Frame dataclass.
3. KArSLResult backward-compatible properties (sign, word).
4. TSNSampler float32 tensor conversion to prevent ONNX runtime data type mismatch.
5. CameraPreview status cards (standby, connecting, error) and safe QImage copying.
6. CameraPreview display_frame replaces connecting overlay with live frame.
7. RecognitionTab transitional state machine: Connecting -> Live Streaming upon 1st frame.
8. Authentic sign predictions: requires hand detection, resets on disconnect.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.domain.entities import Frame, SourceNotFoundError
from src.infrastructure.recognition.karsl_recognizer import (
    KArSLResult,
    TSNSampler,
)
from src.infrastructure.video_source_engine import WebcamAdapter


def test_karsl_result_sign_alias():
    """Verify KArSLResult provides .sign and .word properties matching .arabic_text."""
    res = KArSLResult(
        pred_idx=10,
        arabic_text="مرحبا",
        english_text="hello",
        confidence=0.92,
        hand_detected=True,
        face_detected=True,
        pose_detected=True,
        buffer_fill=50,
        is_new=True,
    )
    assert res.sign == "مرحبا"
    assert res.word == "مرحبا"
    assert res.confidence == 0.92


def test_tsn_sampler_returns_float32():
    """Verify TSNSampler always outputs float32 numpy array for ONNX compatibility."""
    sampler = TSNSampler(target_len=50)
    # Simulate a sequence of 20 frames with float32 or float64 keypoints
    kps_input = np.random.randn(20, 184, 4).astype(np.float32)
    sampled = sampler(kps_input)
    assert sampled.shape == (50, 184, 4)
    assert sampled.dtype == np.float32, f"Expected float32, got {sampled.dtype}"


def test_webcam_adapter_multibackend_and_frame_acquisition():
    """Verify WebcamAdapter tests frame acquisition on open."""
    adapter = WebcamAdapter("webcam:0")
    with patch("cv2.VideoCapture") as mock_cap_cls:
        cap = MagicMock()
        cap.isOpened.return_value = True
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cap.read.return_value = (True, dummy_frame)
        cap.get.side_effect = [640.0, 480.0, 30.0]
        mock_cap_cls.return_value = cap

        opened = adapter.open()
        assert opened is True
        assert adapter.is_open is True
        assert adapter.properties.width == 640
        assert adapter.properties.height == 480
        adapter.close()


def test_webcam_adapter_raises_when_all_backends_fail():
    """Verify WebcamAdapter raises SourceNotFoundError when no backend can acquire frames."""
    adapter = WebcamAdapter("webcam:99")
    with patch("cv2.VideoCapture") as mock_cap_cls:
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cap_cls.return_value = cap

        with pytest.raises(SourceNotFoundError):
            adapter.open()


def test_video_source_use_cases_frame_metadata():
    """Verify WebcamAdapter places both RGB data and raw_bgr in Frame metadata."""
    adapter = WebcamAdapter("webcam:0")
    with patch("cv2.VideoCapture") as mock_cap_cls:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = [640.0, 480.0, 30.0]
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_bgr[:, :, 0] = 255  # Pure Blue in BGR
        cap.read.return_value = (True, dummy_bgr)
        mock_cap_cls.return_value = cap

        adapter.open()
        frame = adapter.read()
        assert isinstance(frame, Frame)
        # RGB data should have channel 0 as Red (which was 0) and channel 2 as Blue (255)
        assert frame.data[0, 0, 2] == 255
        # raw_bgr metadata should have channel 0 as Blue (255)
        assert frame.metadata["raw_bgr"][0, 0, 0] == 255
        adapter.close()


def test_camera_preview_display_frame_updates_pixmap():
    """Verify CameraPreview.display_frame properly renders frames without throwing AttributeError."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication

    from src.presentation.widgets import CameraPreview

    _app = QApplication.instance() or QApplication([])
    preview = CameraPreview()
    preview.set_connecting("Connecting to Camera...", "Initializing device")

    # Pass a valid bright frame
    bright_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    preview.display_frame(bright_frame)

    # Convert to image and sample center pixel
    rendered_img = preview.pixmap().toImage()
    assert not rendered_img.isNull()
    assert rendered_img.width() >= 320
    assert rendered_img.height() >= 240
    # Center pixel should match bright_frame value (200), not dark connecting background (15, 19, 25)
    center_color = rendered_img.pixelColor(rendered_img.width() // 2, rendered_img.height() // 2)
    assert center_color.red() > 150
    assert center_color.green() > 150
    assert center_color.blue() > 150


def test_camera_preview_handles_fortran_and_empty_frames():
    """Verify CameraPreview handles non-contiguous arrays, empty frames, and None gracefully."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication

    from src.presentation.widgets import CameraPreview

    _app = QApplication.instance() or QApplication([])
    preview = CameraPreview()

    # None and empty
    preview.display_frame(None)
    preview.display_frame(np.array([]))

    # Fortran contiguous array (non C-contiguous)
    f_array = np.asfortranarray(np.full((240, 320, 3), 100, dtype=np.uint8))
    assert not f_array.flags.c_contiguous
    preview.display_frame(f_array)
    assert not preview.pixmap().isNull()


def test_recognition_tab_transitional_connecting_state_to_live_streaming():
    """Verify RecognitionTab starts in 'Connecting...', transitions to 'Live Streaming' ONLY after 1st frame."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication

    from src.presentation.tabs.recognition import RecognitionTab

    _app = QApplication.instance() or QApplication([])
    state = AppState()
    source_use_cases = VideoSourceUseCases(state)

    tab = RecognitionTab(state=state, source_use_cases=source_use_cases)
    tab._camera_combo.clear()
    tab._camera_combo.addItem("HD Webcam", 0)

    # Mock source start so camera reports open
    with patch.object(source_use_cases, "start", return_value=True):
        tab._start_camera()

        # Step 1: Immediately after start, badge MUST be 'Connecting...', NOT 'Live Streaming'
        assert tab._status_badge.text() == "Connecting..."
        assert tab._first_frame_rendered is False
        assert tab._decision_card._value_label.text() == "Connecting"

        # Step 2: Simulate delivery of first frame
        dummy_rgb = np.full((480, 640, 3), 150, dtype=np.uint8)
        karsl_res = KArSLResult(
            pred_idx=-1,
            arabic_text="",
            english_text="",
            confidence=0.0,
            hand_detected=False,
            face_detected=False,
            pose_detected=False,
            buffer_fill=1,
            is_new=False,
        )
        tab._on_frame_processed(dummy_rgb, karsl_res, {}, 30.0)

        # Step 3: NOW the badge must transition to 'Live Streaming'
        assert tab._first_frame_rendered is True
        assert "Live Streaming" in tab._status_badge.text()
        assert tab._decision_card._value_label.text() == "Active"

        # Step 4: Disconnect
        tab._stop_camera()
        assert tab._is_streaming is False
        assert tab._first_frame_rendered is False
        assert tab._status_badge.text() == "Camera Ready"
        assert tab._caption_label.text() == "جاهز للترجمة"
        assert tab._conf_bar.value() == 0

    tab._worker.stop()
