"""Modern, clean Recognition tab exclusively focused on Arabic Sign Language Recognition.

Provides a balanced two-column desktop interface:
- Left Column: DirectShow camera stream, device switching (HD Webcam & e2eSoft iVCam), landmarks toggle.
- Right Column: Recognized Arabic word, confidence badge, sentence builder, TTS speech, and subtitle overlay.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.domain.entities import VideoSourceType
from src.infrastructure.analytics import AnalyticsEngine
from src.infrastructure.camera_service import OpenCVCameraService
from src.infrastructure.dialect_manager import DialectManager
from src.infrastructure.hand_detection.detector import HandDetector, HandDetectorConfig
from src.infrastructure.hand_detection.renderer import HandRenderer
from src.infrastructure.optimization import PerformanceOptimizer
from src.infrastructure.recognition.hybrid_pipeline import (
    HybridRecognitionPipeline,
)
from src.infrastructure.recognition.karsl_recognizer import (
    KArSLRecognizer,
    KArSLResult,
)
from src.infrastructure.text_builder import ArabicTextBuilder
from src.presentation.dialogs.karsl_vocab_dialog import KArSLVocabularyDialog
from src.presentation.widgets import CameraPreview, SectionHeader, StatCard


class SignRecognitionWorker(QThread):
    """Background worker thread executing real-time MediaPipe tracking and KArSL/Custom sign recognition."""

    karsl_loaded = Signal(bool, str)
    frame_processed = Signal(object, object, dict, float)  # frame, karsl_res, vis_info, elapsed_ms
    error = Signal(str)

    def __init__(
        self,
        detector: Optional[HandDetector] = None,
        renderer: Optional[HandRenderer] = None,
        pipeline: Optional[HybridRecognitionPipeline] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._karsl: Optional[KArSLRecognizer] = None
        self._detector = detector
        self._renderer = renderer
        self._pipeline = pipeline
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)
        self._running = True
        self._lock = threading.Lock()

        # Configurable processing parameters
        self.active_engine = "karsl"  # "karsl" or "custom"
        self.mirror = True
        self.show_landmarks = True
        self.confidence_threshold = 0.35

    @property
    def karsl_recognizer(self) -> Optional[KArSLRecognizer]:
        return self._karsl

    def run(self) -> None:
        # Initialize KArSL Recognizer
        try:
            self._karsl = KArSLRecognizer()
            if self._karsl.is_loaded:
                self.karsl_loaded.emit(True, "KArSL-502 engine loaded successfully")
            else:
                self.karsl_loaded.emit(False, "Failed to load KArSL-502 engine")
        except Exception as e:
            self.karsl_loaded.emit(False, str(e))

        while self._running:
            try:
                frame = self._queue.get(timeout=0.03)
            except queue.Empty:
                continue

            if frame is None:
                continue

            start_t = time.perf_counter()
            karsl_res: Optional[KArSLResult] = None
            vis_info: dict = {
                "hands_detected": 0,
                "pose_detected": False,
                "face_detected": False,
                "keypoint_count": 0,
            }

            try:
                # Sign Language Recognition Path
                if self.active_engine == "karsl" and self._karsl and self._karsl.is_loaded:
                    res, ann_bgr, vis = self._karsl.process_frame(frame)
                    karsl_res = res
                    vis_info = vis
                    frame_to_show = ann_bgr if self.show_landmarks else frame
                    annotated = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
                elif self.active_engine == "custom" and self._detector:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    det_res = self._detector.detect(frame_rgb)
                    vis_info["hands_detected"] = len(det_res.hands)
                    if self.show_landmarks and self._renderer and det_res.hands:
                        ann_bgr = self._renderer.render(frame.copy(), det_res.hands)
                        annotated = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        annotated = frame_rgb

                    if det_res.hands and self._pipeline:
                        hyp_res = self._pipeline.process(det_res.hands[0])
                        if hyp_res and hyp_res.is_valid:
                            karsl_res = KArSLResult(
                                pred_idx=0,
                                arabic_text=hyp_res.arabic_text,
                                english_text="",
                                confidence=hyp_res.final_result.confidence,
                                hand_detected=True,
                                face_detected=False,
                                pose_detected=False,
                                buffer_fill=1,
                                is_new=True,
                                decision_path=hyp_res.decision_path,
                            )
                else:
                    annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            except Exception as e:
                self.error.emit(f"Vision processing error: {e}")
                annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            elapsed = (time.perf_counter() - start_t) * 1000.0
            self.frame_processed.emit(annotated, karsl_res, vis_info, elapsed)

    def process_frame(self, frame: np.ndarray) -> None:
        """Enqueue frame for asynchronous worker processing (drops frame if busy)."""
        if not self._queue.full():
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def reset(self) -> None:
        """Clear queued frames, motion detector, and temporal recognizer state."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
        if self._karsl:
            self._karsl.reset()

    def stop(self) -> None:
        """Gracefully terminate background thread."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self.wait(1000)


# Backward-compatible aliases
UnifiedVisionWorker = SignRecognitionWorker
KArSLWorker = SignRecognitionWorker


class RecognitionTab(QWidget):
    """Modern, clean desktop interface for Real-Time Arabic Sign Language Recognition.
    
    Layout:
    - Left Column: Live camera view, DirectShow source selection, clean start/stop controls.
    - Right Column: Recognized Arabic word, confidence meter, sentence builder, TTS speech.
    """

    def __init__(
        self,
        state: AppState,
        source_use_cases: VideoSourceUseCases,
        pipeline: Optional[HybridRecognitionPipeline] = None,
        text_builder: Optional[ArabicTextBuilder] = None,
        dialect_manager: Optional[DialectManager] = None,
        analytics: Optional[AnalyticsEngine] = None,
        optimizer: Optional[PerformanceOptimizer] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._source_use_cases = source_use_cases
        self._pipeline = pipeline or HybridRecognitionPipeline()
        self._text_builder = text_builder or ArabicTextBuilder()
        self._dialect_manager = dialect_manager
        self._analytics = analytics
        self._optimizer = optimizer
        self._camera_service = OpenCVCameraService()
        self._logger = logging.getLogger(__name__)

        # Camera & Recognition state
        self._is_streaming = False
        self._first_frame_rendered = False
        self._detection_count = 0
        self._last_display_frame: Optional[np.ndarray] = None
        self._current_word: str = ""
        self._accumulated_sentence: list[str] = []

        # Hand Detector for Custom Fallback
        det_conf = self._optimizer.profile.min_detection_confidence if self._optimizer else 0.35
        det_track = self._optimizer.profile.min_tracking_confidence if self._optimizer else 0.35
        self._detector = HandDetector(HandDetectorConfig(
            max_num_hands=2,
            min_detection_confidence=det_conf,
            min_tracking_confidence=det_track,
        ))
        self._renderer = HandRenderer()

        # Text-To-Speech engine
        self._tts = None
        try:
            from PySide6.QtTextToSpeech import QTextToSpeech
            self._tts = QTextToSpeech(self)
        except Exception as exc:
            self._logger.debug(f"QTextToSpeech unavailable: {exc}")

        # Background Vision Worker
        self._worker = SignRecognitionWorker(
            detector=self._detector,
            renderer=self._renderer,
            pipeline=self._pipeline,
            parent=self,
        )
        self._worker.karsl_loaded.connect(self._on_karsl_loaded)
        self._worker.frame_processed.connect(self._on_frame_processed)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

        # Capture polling timer (~30 FPS)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_next_frame)

        self._setup_ui()
        self._refresh_camera_list()
        self._connect_signals()

    # ── UI Construction ──────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 14, 18, 14)
        main_layout.setSpacing(12)

        # Header with clean title
        header = SectionHeader(
            "Arabic Sign Language Recognition",
            "Real-time vision translation with MediaPipe tracking and KArSL-502 continuous sign recognition",
        )
        main_layout.addWidget(header)

        # Main 2-Column Splitter
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)

        # ═════════════════════════════════════════════════════════════════
        # LEFT COLUMN: Camera Preview & Video Controls
        # ═════════════════════════════════════════════════════════════════
        left_card = QFrame()
        left_card.setObjectName("leftCameraCard")
        left_card.setStyleSheet("""
            QFrame#leftCameraCard {
                background-color: #181d24;
                border: 1px solid #2b3441;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Top Camera Bar: Dropdown + Refresh + Start/Stop + Landmarks
        cam_bar = QHBoxLayout()
        cam_bar.setSpacing(8)

        cam_label = QLabel("Camera:")
        cam_label.setStyleSheet("font-weight: 600; color: #94a3b8;")
        cam_bar.addWidget(cam_label)

        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(260)
        self._camera_combo.setToolTip("Select camera (HD Webcam or e2eSoft iVCam)")
        cam_bar.addWidget(self._camera_combo, stretch=1)

        # Alias for backward compatibility with internal scripts/tests
        self._device_combo = self._camera_combo

        self._refresh_cam_btn = QPushButton("Refresh")
        self._refresh_cam_btn.setToolTip("Scan system for newly connected webcams or iVCam")
        cam_bar.addWidget(self._refresh_cam_btn)

        # Primary Start / Stop Button
        self._start_stop_btn = QPushButton("Start Camera")
        self._start_stop_btn.setObjectName("primaryButton")
        self._start_stop_btn.setMinimumWidth(110)
        cam_bar.addWidget(self._start_stop_btn)

        # Compatibility aliases for tests that expect _connect_btn, _disconnect_btn, _start_btn, _stop_btn
        self._connect_btn = self._start_stop_btn
        self._disconnect_btn = self._start_stop_btn
        self._start_btn = self._start_stop_btn
        self._stop_btn = self._start_stop_btn

        left_layout.addLayout(cam_bar)

        # Camera Viewport
        self._preview = CameraPreview()
        self._preview.setMinimumSize(480, 360)
        left_layout.addWidget(self._preview, stretch=1)

        # Camera Bottom Controls Strip
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(12)

        self._status_badge = QLabel("Camera Ready")
        self._status_badge.setStyleSheet("""
            background-color: #14181f;
            color: #94a3b8;
            border: 1px solid #2b3441;
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 600;
        """)
        bottom_bar.addWidget(self._status_badge)

        self._landmarks_check = QCheckBox("Show Landmarks")
        self._landmarks_check.setChecked(True)
        bottom_bar.addWidget(self._landmarks_check)

        self._mirror_check = QCheckBox("Mirror Feed")
        self._mirror_check.setChecked(True)
        bottom_bar.addWidget(self._mirror_check)

        bottom_bar.addStretch()

        self._fps_indicator = QLabel("FPS: --")
        self._fps_indicator.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 500;")
        bottom_bar.addWidget(self._fps_indicator)

        self._resolution_combo = QComboBox()
        self._resolution_combo.addItem("640x480 (Fast)", (640, 480))
        self._resolution_combo.addItem("1280x720 (HD)", (1280, 720))
        self._resolution_combo.setMinimumWidth(135)
        self._resolution_combo.setMaximumWidth(160)
        bottom_bar.addWidget(self._resolution_combo)

        left_layout.addLayout(bottom_bar)
        content_splitter.addWidget(left_card)

        # ═════════════════════════════════════════════════════════════════
        # RIGHT COLUMN: Recognition Results, Sentence Construction, Audio
        # ═════════════════════════════════════════════════════════════════
        right_card = QFrame()
        right_card.setObjectName("rightResultsCard")
        right_card.setStyleSheet("""
            QFrame#rightResultsCard {
                background-color: #181d24;
                border: 1px solid #2b3441;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(12)

        # Model Selector Bar
        model_bar = QHBoxLayout()
        model_bar.setSpacing(8)

        model_label = QLabel("Sign Model:")
        model_label.setStyleSheet("font-weight: 600; color: #94a3b8;")
        model_bar.addWidget(model_label)

        self._engine_selector = QComboBox()
        self._engine_selector.addItem("KArSL-502 (502 Standard Signs)", "karsl")
        self._engine_selector.addItem("Iraqi Custom Classifier", "custom")
        model_bar.addWidget(self._engine_selector, stretch=1)

        self._vocab_btn = QPushButton("502 Signs List")
        self._vocab_btn.setToolTip("View full dictionary of 502 supported Arabic signs")
        model_bar.addWidget(self._vocab_btn)

        right_layout.addLayout(model_bar)

        # Active Sign Detection Display Box
        detection_box = QFrame()
        detection_box.setObjectName("detectionBox")
        detection_box.setStyleSheet("""
            QFrame#detectionBox {
                background-color: #14181f;
                border: 1px solid #2b3441;
                border-radius: 8px;
            }
        """)
        det_vlayout = QVBoxLayout(detection_box)
        det_vlayout.setContentsMargins(16, 12, 16, 14)
        det_vlayout.setSpacing(10)

        det_header = QLabel("DETECTED SIGN")
        det_header.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 700; letter-spacing: 1px; background: transparent; border: none;")
        det_vlayout.addWidget(det_header)

        # Large prominent Arabic text display
        self._caption_label = QLabel("جاهز للترجمة")
        self._caption_label.setAlignment(Qt.AlignCenter)
        self._caption_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: 700;
                color: #14b8a6;
                padding: 4px;
                background: transparent;
                border: none;
            }
        """)
        self._caption_label.setMinimumHeight(55)
        self._caption_label.setWordWrap(True)
        det_vlayout.addWidget(self._caption_label)

        self._word_label = self._caption_label  # compatibility alias

        # Confidence bar & badge row
        conf_row = QHBoxLayout()
        conf_row.setSpacing(10)

        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setValue(0)
        self._conf_bar.setFixedHeight(8)
        conf_row.addWidget(self._conf_bar, stretch=1)

        self._conf_badge = QLabel("Confidence: --")
        self._conf_badge.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600; min-width: 95px; background: transparent; border: none;")
        conf_row.addWidget(self._conf_badge)

        det_vlayout.addLayout(conf_row)
        right_layout.addWidget(detection_box)

        # Sentence Construction Card
        sentence_group = QGroupBox("Constructed Sentence")
        sent_layout = QVBoxLayout(sentence_group)
        sent_layout.setSpacing(8)

        self._sentence_edit = QTextEdit()
        self._sentence_edit.setPlaceholderText("Recognized signs will accumulate here as an Arabic sentence...")
        self._sentence_edit.setMaximumHeight(85)
        self._sentence_edit.setLayoutDirection(Qt.RightToLeft)
        font = self._sentence_edit.font()
        font.setPointSize(15)
        self._sentence_edit.setFont(font)
        sent_layout.addWidget(self._sentence_edit)

        # Sentence Editing Bar: Speak, Space, Delete, Clear
        sent_btn_bar = QHBoxLayout()
        sent_btn_bar.setSpacing(6)

        self._speak_btn = QPushButton("Speak (TTS)")
        self._speak_btn.setToolTip("Read constructed sentence aloud using speech synthesis")
        sent_btn_bar.addWidget(self._speak_btn)

        self._space_btn = QPushButton("Space")
        self._space_btn.setToolTip("Add space between words")
        sent_btn_bar.addWidget(self._space_btn)

        self._delete_btn = QPushButton("Delete Word")
        self._delete_btn.setToolTip("Remove last recognized word")
        sent_btn_bar.addWidget(self._delete_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("dangerButton")
        self._clear_btn.setToolTip("Clear current sentence")
        sent_btn_bar.addWidget(self._clear_btn)

        sent_layout.addLayout(sent_btn_bar)
        right_layout.addWidget(sentence_group)

        # Recent Signs History
        history_group = QGroupBox("Recent Recognitions")
        hist_layout = QVBoxLayout(history_group)
        hist_layout.setContentsMargins(6, 6, 6, 6)

        self._history_list = QListWidget()
        self._history_list.setMaximumHeight(140)
        hist_layout.addWidget(self._history_list)
        right_layout.addWidget(history_group, stretch=1)

        # Quick Overlay Action Bar
        overlay_bar = QHBoxLayout()
        self._toggle_overlay_btn = QPushButton("Toggle Desktop Subtitles")
        self._toggle_overlay_btn.setToolTip("Show/hide transparent desktop subtitle overlay")
        overlay_bar.addWidget(self._toggle_overlay_btn)

        self._screenshot_btn = QPushButton("Screenshot")
        self._screenshot_btn.setToolTip("Save current annotated frame to data/captures/")
        overlay_bar.addWidget(self._screenshot_btn)

        right_layout.addLayout(overlay_bar)
        content_splitter.addWidget(right_card)

        # Set balanced splitter proportions (58% left, 42% right)
        content_splitter.setStretchFactor(0, 58)
        content_splitter.setStretchFactor(1, 42)
        main_layout.addWidget(content_splitter, stretch=1)

        # Stat cards compatibility for internal references
        self._fps_card = StatCard("FPS", "0.0")
        self._latency_card = StatCard("Latency", "-- ms")
        self._detections_card = StatCard("Detections", "0")
        self._decision_card = StatCard("Status", "Ready")

    # ── Signal Connections ───────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._refresh_cam_btn.clicked.connect(self._refresh_camera_list)
        self._start_stop_btn.clicked.connect(self._toggle_camera_stream)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_combo_changed)
        self._resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self._landmarks_check.toggled.connect(self._on_landmarks_toggled)
        self._mirror_check.toggled.connect(self._on_mirror_toggled)
        self._engine_selector.currentIndexChanged.connect(self._on_engine_changed)
        self._vocab_btn.clicked.connect(self._open_vocab_dialog)

        # Sentence builder controls
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._space_btn.clicked.connect(self._on_space_clicked)
        self._speak_btn.clicked.connect(self._on_speak_clicked)
        self._screenshot_btn.clicked.connect(self._take_screenshot)
        self._toggle_overlay_btn.clicked.connect(self._toggle_overlay)

    # ── Camera Operations ────────────────────────────────────────────────

    def _refresh_camera_list(self) -> None:
        """Enumerate system cameras via DirectShow COM and populate dropdown."""
        current_sel = self._camera_combo.currentData()
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()

        active_idx = current_sel if self._is_streaming else None
        cameras = self._camera_service.list_cameras(force_refresh=True, active_index=active_idx)
        if not cameras:
            self._camera_combo.addItem("No cameras found", None)
            self._camera_combo.setEnabled(False)
            self._start_stop_btn.setEnabled(False)
            self._status_badge.setText("No Camera Detected")
            self._status_badge.setStyleSheet("background-color: #3b1d1d; color: #ef4444; border: 1px solid #7f1d1d; border-radius: 4px; padding: 4px 10px;")
        else:
            self._camera_combo.setEnabled(True)
            self._start_stop_btn.setEnabled(True)
            found_idx = -1
            for i, cam in enumerate(cameras):
                label = f"{cam.name}"
                if cam.device_type == "laptop":
                    label += " (Laptop Camera)"
                elif "ivcam" in cam.name.lower() or cam.device_type == "virtual":
                    label += " (iPhone / Virtual)"
                elif cam.device_type == "usb":
                    label += " (USB Camera)"

                self._camera_combo.addItem(label, cam.index)
                if current_sel is not None and cam.index == current_sel:
                    found_idx = i

            if found_idx >= 0:
                self._camera_combo.setCurrentIndex(found_idx)
            else:
                self._camera_combo.setCurrentIndex(0)

            if not self._is_streaming:
                self._status_badge.setText("Ready to Connect")
                self._status_badge.setStyleSheet("background-color: #14181f; color: #94a3b8; border: 1px solid #2b3441; border-radius: 4px; padding: 4px 10px;")

        self._camera_combo.blockSignals(False)

    def _toggle_camera_stream(self) -> None:
        """Toggle live camera streaming on / off."""
        if self._is_streaming:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        """Start capturing from currently selected camera."""
        idx = self._camera_combo.currentData()
        if idx is None:
            return

        self._first_frame_rendered = False
        self._status_badge.setText("Connecting...")
        self._status_badge.setStyleSheet("background-color: #3b2d14; color: #f59e0b; border: 1px solid #78350f; border-radius: 4px; padding: 4px 10px; font-weight: 600;")
        self._preview.set_connecting(title="Connecting to Camera...", subtitle=f"Initializing DirectShow device (Index {idx})")

        target_w, target_h = self._resolution_combo.currentData() or (640, 480)
        self._source_use_cases.select_source(
            VideoSourceType.WEBCAM,
            f"webcam:{idx}",
            device_index=idx,
            resolution=(target_w, target_h),
        )

        res = self._source_use_cases.start()
        if res:
            self._is_streaming = True
            self._start_stop_btn.setText("Stop Camera")
            self._start_stop_btn.setObjectName("dangerButton")
            self._start_stop_btn.setStyleSheet("background-color: #dc2626; border: 1px solid #b91c1c;")
            self._timer.start(30)
            self._decision_card.set_value("Connecting")
        else:
            self._status_badge.setText("Failed to Open")
            self._status_badge.setStyleSheet("background-color: #3b1d1d; color: #ef4444; border: 1px solid #7f1d1d; border-radius: 4px; padding: 4px 10px;")
            self._preview.set_error(title="Camera Error", subtitle=f"Failed to acquire video stream from device {idx}")

    def _stop_camera(self) -> None:
        """Stop capturing and release camera resources cleanly."""
        self._timer.stop()
        self._source_use_cases.stop()
        self._is_streaming = False
        self._first_frame_rendered = False
        self._worker.reset()
        self._start_stop_btn.setText("Start Camera")
        self._start_stop_btn.setObjectName("primaryButton")
        self._start_stop_btn.setStyleSheet("")
        self._status_badge.setText("Camera Ready")
        self._status_badge.setStyleSheet("background-color: #14181f; color: #94a3b8; border: 1px solid #2b3441; border-radius: 4px; padding: 4px 10px;")
        self._fps_indicator.setText("FPS: --")
        self._preview.set_standby()
        self._decision_card.set_value("Ready")
        self._caption_label.setText("جاهز للترجمة")
        self._conf_bar.setValue(0)
        self._conf_badge.setText("Confidence: --")
        self._current_word = ""

    def _on_camera_combo_changed(self, index: int) -> None:
        """Handle immediate camera switching from dropdown."""
        if index < 0:
            return
        if self._is_streaming:
            self._stop_camera()
            QTimer.singleShot(100, self._start_camera)

    def _on_resolution_changed(self, index: int) -> None:
        """Handle resolution change."""
        if self._is_streaming:
            self._stop_camera()
            QTimer.singleShot(100, self._start_camera)

    def _read_next_frame(self) -> None:
        """Read frame from camera adapter and pass to background recognition worker."""
        if not self._is_streaming:
            return

        frame_data = self._source_use_cases.read_frame()
        if frame_data is None:
            return

        # Extract raw BGR numpy array safely from Frame object or direct ndarray
        if hasattr(frame_data, "metadata") and "raw_bgr" in frame_data.metadata:
            raw_bgr = frame_data.metadata["raw_bgr"]
        elif hasattr(frame_data, "data"):
            raw_bgr = frame_data.data
        elif isinstance(frame_data, np.ndarray):
            raw_bgr = frame_data
        else:
            return

        if raw_bgr is None or getattr(raw_bgr, "size", 0) == 0:
            return

        if self._mirror_check.isChecked():
            raw_bgr = cv2.flip(raw_bgr, 1)

        self._last_display_frame = raw_bgr
        self._worker.process_frame(raw_bgr)

    @Slot(object, object, dict, float)
    def _on_frame_processed(
        self,
        annotated_frame: np.ndarray,
        karsl_res: Optional[KArSLResult],
        vis_info: dict,
        elapsed_ms: float,
    ) -> None:
        """Receive processed annotated frame and sign recognition result from worker."""
        if not self._is_streaming:
            return

        if annotated_frame is not None:
            self._preview.display_frame(annotated_frame)
            if not self._first_frame_rendered:
                self._first_frame_rendered = True
                self._status_badge.setText("● Live Streaming")
                self._status_badge.setStyleSheet(
                    "background-color: #064e3b; color: #10b981; border: 1px solid #047857; "
                    "border-radius: 4px; padding: 4px 10px; font-weight: 600;"
                )
                self._decision_card.set_value("Active")

        # Update FPS telemetry
        fps = 1000.0 / max(elapsed_ms, 1.0)
        self._fps_indicator.setText(f"{fps:.1f} FPS · {elapsed_ms:.0f} ms")
        self._fps_card.set_value(f"{fps:.1f}")
        self._latency_card.set_value(f"{elapsed_ms:.0f} ms")

        # Update Sign Language Recognition results only from live valid frames
        if (
            karsl_res
            and karsl_res.hand_detected
            and karsl_res.confidence >= self._worker.confidence_threshold
        ):
            word = (
                getattr(karsl_res, "arabic_text", "")
                or getattr(karsl_res, "sign", "")
            ).strip()
            conf = getattr(karsl_res, "confidence", 0.0)
            conf_pct = int(conf * 100)

            self._conf_bar.setValue(conf_pct)
            self._conf_badge.setText(f"Confidence: {conf_pct}%")

            if word and (karsl_res.is_new or word != self._current_word):
                self._current_word = word
                self._caption_label.setText(word)
                self._detection_count += 1
                self._detections_card.set_value(str(self._detection_count))

                # Update state caption
                self._state.caption = word

                # Append to sentence builder
                self._append_to_sentence(word)

                # Append to history list
                timestamp = time.strftime("%H:%M:%S")
                item_text = f"[{timestamp}]  {word}  ({conf_pct}%)"
                self._history_list.insertItem(0, item_text)
                if self._history_list.count() > 50:
                    self._history_list.takeItem(self._history_list.count() - 1)
        elif karsl_res and not karsl_res.hand_detected:
            self._conf_bar.setValue(0)
            self._conf_badge.setText("Waiting for sign...")

    def _append_to_sentence(self, word: str) -> None:
        """Add newly recognized sign word to the constructed sentence."""
        if not self._accumulated_sentence or self._accumulated_sentence[-1] != word:
            self._accumulated_sentence.append(word)
            full_text = " ".join(self._accumulated_sentence)
            self._sentence_edit.setPlainText(full_text)

    # ── Sentence Controls ────────────────────────────────────────────────

    def _on_clear_clicked(self) -> None:
        self._accumulated_sentence.clear()
        self._current_word = ""
        self._sentence_edit.clear()
        self._caption_label.setText("جاهز للترجمة")
        self._conf_bar.setValue(0)
        self._conf_badge.setText("Confidence: --")

    def _on_delete_clicked(self) -> None:
        if self._accumulated_sentence:
            self._accumulated_sentence.pop()
            full_text = " ".join(self._accumulated_sentence)
            self._sentence_edit.setPlainText(full_text)
            self._current_word = self._accumulated_sentence[-1] if self._accumulated_sentence else ""
            self._caption_label.setText(self._current_word or "جاهز للترجمة")

    def _on_space_clicked(self) -> None:
        cur = self._sentence_edit.toPlainText()
        self._sentence_edit.setPlainText(cur + " ")

    def _on_speak_clicked(self) -> None:
        """Speak constructed Arabic sentence using Text-to-Speech."""
        text = self._sentence_edit.toPlainText().strip()
        if not text:
            text = self._caption_label.text().strip()

        if text and text != "جاهز للترجمة":
            if self._tts:
                self._tts.say(text)
            else:
                self._logger.info(f"TTS requested: {text}")

    def _toggle_overlay(self) -> None:
        """Toggle floating transparent desktop subtitles."""
        vis = not self._state.overlay_visible
        self._state.overlay_visible = vis
        status_str = "ON" if vis else "OFF"
        self._toggle_overlay_btn.setText(f"Desktop Subtitles: {status_str}")

    def _take_screenshot(self) -> None:
        """Save current frame to data/captures/ directory."""
        if self._last_display_frame is not None:
            out_dir = Path("data/captures")
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"capture_{int(time.time())}.jpg"
            cv2.imwrite(str(fname), cv2.cvtColor(self._last_display_frame, cv2.COLOR_RGB2BGR))
            self._status_badge.setText(f"Saved: {fname.name}")

    # ── Toggles & Settings ───────────────────────────────────────────────

    def _on_landmarks_toggled(self, checked: bool) -> None:
        self._worker.show_landmarks = checked

    def _on_mirror_toggled(self, checked: bool) -> None:
        self._worker.mirror = checked

    def _on_engine_changed(self, index: int) -> None:
        self._worker.active_engine = self._engine_selector.currentData()

    def _open_vocab_dialog(self) -> None:
        labels_path = Path("data/models/karsl/labels.json")
        dlg = KArSLVocabularyDialog(labels_path=labels_path, parent=self)
        dlg.exec()

    def _on_karsl_loaded(self, success: bool, msg: str) -> None:
        if success:
            self._logger.info(f"KArSL: {msg}")
        else:
            self._logger.warning(f"KArSL: {msg}")

    def _on_worker_error(self, err_msg: str) -> None:
        self._logger.error(err_msg)

    def closeEvent(self, event) -> None:
        """Cleanup thread and camera on close."""
        self._stop_camera()
        self._worker.stop()
        super().closeEvent(event)
