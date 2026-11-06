from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.domain.dialect import DialectId
from src.domain.entities import VideoSourceType
from src.infrastructure.analytics import AnalyticsEngine
from src.infrastructure.camera_service import OpenCVCameraService
from src.infrastructure.dialect_manager import DialectManager
from src.infrastructure.hand_detection.detector import HandDetector, HandDetectorConfig
from src.infrastructure.hand_detection.renderer import HandRenderer
from src.infrastructure.object_detection.yolo_detector import DetectedObject, YOLODetector
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
from src.presentation.widgets import (
    CameraPreview,
    DialectSelector,
    SectionHeader,
    StatCard,
)


class UnifiedVisionWorker(QThread):
    """Background worker thread executing multimodal KArSL sign recognition and/or YOLO object detection."""

    karsl_loaded = Signal(bool, str)
    yolo_loaded = Signal(bool, str)
    frame_processed = Signal(object, object, list, dict, float)
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._karsl: Optional[KArSLRecognizer] = None
        self._yolo: Optional[YOLODetector] = None
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)
        self._running = True
        self._lock = threading.Lock()

        # Configurable processing parameters
        self.mode = "sign"  # "sign", "yolo", "combined"
        self.active_engine = "karsl"  # "karsl", "custom"
        self.mirror = True
        self.show_landmarks = True
        self.show_boxes = True
        self.confidence_threshold = 0.40

    @property
    def karsl_recognizer(self) -> Optional[KArSLRecognizer]:
        return self._karsl

    @property
    def yolo_detector(self) -> Optional[YOLODetector]:
        return self._yolo

    def run(self) -> None:
        # 1. Initialize KArSL Recognizer
        try:
            self._karsl = KArSLRecognizer()
            if self._karsl.is_loaded:
                self.karsl_loaded.emit(True, "KArSL-502 engine loaded successfully")
            else:
                self.karsl_loaded.emit(False, "Failed to load KArSL-502 engine")
        except Exception as e:
            self.karsl_loaded.emit(False, str(e))

        # 2. Initialize YOLO Object Detector
        try:
            self._yolo = YOLODetector()
            if self._yolo.is_loaded:
                self.yolo_loaded.emit(True, f"YOLO detector ready ({self._yolo.backend})")
            else:
                self.yolo_loaded.emit(False, "YOLO detector unavailable")
        except Exception as e:
            self.yolo_loaded.emit(False, str(e))

        # Main frame loop
        while self._running:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None or not self._running:
                break

            t0 = time.perf_counter()
            try:
                # Mirroring if enabled
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                annotated = frame.copy()
                karsl_res: Optional[KArSLResult] = None
                yolo_detections: list[DetectedObject] = []
                vis_info: dict = {}

                with self._lock:
                    # ── Sign Recognition Path ──
                    if self.mode in ("sign", "combined") and self.active_engine == "karsl":
                        if self._karsl and self._karsl.is_loaded:
                            karsl_res, karsl_ann, vis_info = self._karsl.process_frame(frame)
                            if self.show_landmarks:
                                annotated = karsl_ann

                    # ── YOLO Object Detection Path ──
                    if self.mode in ("yolo", "combined"):
                        if self._yolo and self._yolo.is_loaded:
                            yolo_detections = self._yolo.detect(
                                frame,
                                confidence_threshold=self.confidence_threshold,
                            )
                            if self.show_boxes:
                                annotated = self._yolo.draw_detections(
                                    annotated,
                                    yolo_detections,
                                    show_boxes=self.show_boxes,
                                    show_labels=True,
                                )

                elapsed = (time.perf_counter() - t0) * 1000.0
                self.frame_processed.emit(annotated, karsl_res, yolo_detections, vis_info, elapsed)

            except Exception as ex:
                self.error.emit(str(ex))

    def submit_frame(self, frame: np.ndarray) -> bool:
        """Non-blocking frame submission with bounded backpressure."""
        if not self._running:
            return False
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
                return True
            except queue.Full:
                return False

    def reset_buffer(self) -> None:
        """Clear queued frames and model state."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            if self._karsl:
                self._karsl.reset()

    def stop(self) -> None:
        """Gracefully terminate worker thread."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self.wait(1500)


# Backward-compatible alias for existing tests
KArSLWorker = UnifiedVisionWorker


class RecognitionTab(QWidget):
    """Full-featured Live Camera Recognition tab supporting:
    - Laptop webcam, external USB webcam, Smartphone camera (Wi-Fi & USB virtual).
    - Sign Language Recognition (KArSL-502 & Custom), YOLO Object Detection, and Combined Mode.
    - Screenshot saving, live MP4 video recording, resolution switching, and mirror mode.
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

        # Current camera / source state
        self._source_type = VideoSourceType.WEBCAM
        self._source_uri = "webcam:0"
        self._is_connected = False
        self._is_detecting = False
        self._detection_count = 0

        # Recording state
        self._is_recording = False
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._last_display_frame: Optional[np.ndarray] = None
        self._record_timer = QTimer(self)
        self._record_timer.timeout.connect(self._blink_rec_indicator)
        self._rec_indicator_visible = True

        # Custom Hand Detector & Renderer (for Custom Classifier fallback)
        det_conf = self._optimizer.profile.min_detection_confidence if self._optimizer else 0.30
        det_track = self._optimizer.profile.min_tracking_confidence if self._optimizer else 0.30
        self._detector = HandDetector(HandDetectorConfig(
            max_num_hands=self._optimizer.profile.max_hands if self._optimizer else 2,
            min_detection_confidence=det_conf,
            min_tracking_confidence=det_track,
        ))
        self._renderer = HandRenderer()

        # Background Vision Worker
        self._worker = UnifiedVisionWorker(self)
        self._worker.karsl_loaded.connect(self._on_karsl_loaded)
        self._worker.yolo_loaded.connect(self._on_yolo_loaded)
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
        main_layout.setContentsMargins(18, 12, 18, 12)
        main_layout.setSpacing(12)

        header = SectionHeader(
            "Live Camera & Vision Recognition",
            "Unified real-time camera recognition (HD Webcam & e2eSoft iVCam) with Sign Language & YOLO Detection",
        )
        main_layout.addWidget(header)

        # ─── Panel 1: Camera Source & Connection Panel ───────────────────
        camera_panel = QFrame()
        camera_panel.setObjectName("cameraPanel")
        camera_panel.setStyleSheet("""
            QFrame#cameraPanel {
                background-color: #252526;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        cam_vlayout = QVBoxLayout(camera_panel)
        cam_vlayout.setSpacing(8)

        # Row 1: Unified Camera Dropdown + Refresh Button + Connect / Disconnect + Status Badge
        r1_layout = QHBoxLayout()
        r1_layout.setSpacing(10)

        r1_layout.addWidget(QLabel("<b>Camera:</b>"))
        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(340)
        self._camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 5px 10px;
                color: #ffffff;
                font-weight: 500;
            }
            QComboBox:hover {
                border-color: #007acc;
            }
        """)
        r1_layout.addWidget(self._camera_combo)

        # Compatibility alias for internal references / tests
        self._device_combo = self._camera_combo

        self._refresh_cam_btn = QPushButton("🔄 Refresh")
        self._refresh_cam_btn.setToolTip("Scan system for newly connected webcams or iVCam")
        r1_layout.addWidget(self._refresh_cam_btn)

        r1_layout.addSpacing(8)

        # Connect / Disconnect Buttons
        self._connect_btn = QPushButton("🔗  Connect")
        self._connect_btn.setObjectName("primaryButton")
        r1_layout.addWidget(self._connect_btn)

        self._disconnect_btn = QPushButton("🔌  Disconnect")
        self._disconnect_btn.setEnabled(False)
        r1_layout.addWidget(self._disconnect_btn)

        # Connection Status Badge
        self._status_badge = QLabel("⚪ Disconnected")
        self._status_badge.setStyleSheet("""
            background-color: #1e1e1e;
            color: #9d9d9d;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 4px 10px;
            font-weight: 600;
        """)
        r1_layout.addWidget(self._status_badge)
        r1_layout.addStretch()

        cam_vlayout.addLayout(r1_layout)

        # Row 2: AI Mode + Engine + Detection Controls + Toggles
        r2_layout = QHBoxLayout()
        r2_layout.setSpacing(12)

        r2_layout.addWidget(QLabel("<b>AI Mode:</b>"))
        self._ai_mode_combo = QComboBox()
        self._ai_mode_combo.addItem("🤟  Sign Language Recognition", "sign")
        self._ai_mode_combo.addItem("📦  Object Detection (YOLOv8)", "yolo")
        self._ai_mode_combo.addItem("🔀  Combined Mode (Sign + YOLO)", "combined")
        self._ai_mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #007acc;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
                font-weight: 600;
            }
        """)
        r2_layout.addWidget(self._ai_mode_combo)

        # Engine selector
        self._engine_label = QLabel("Engine:")
        r2_layout.addWidget(self._engine_label)
        self._engine_selector = QComboBox()
        self._engine_selector.addItem("KArSL-502 (502 Signs)", "karsl")
        self._engine_selector.addItem("Custom Classifier", "custom")
        r2_layout.addWidget(self._engine_selector)

        self._vocab_btn = QPushButton("📖 502 Signs")
        self._vocab_btn.setToolTip("View full dictionary of 502 supported Arabic signs")
        r2_layout.addWidget(self._vocab_btn)

        r2_layout.addSpacing(8)

        # Detection Start / Stop
        self._start_btn = QPushButton("▶  Start Detection")
        self._start_btn.setObjectName("successButton")
        self._start_btn.setEnabled(False)
        r2_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop Detection")
        self._stop_btn.setObjectName("dangerButton")
        self._stop_btn.setEnabled(False)
        r2_layout.addWidget(self._stop_btn)

        r2_layout.addStretch()

        # Resolution dropdown
        r2_layout.addWidget(QLabel("Resolution:"))
        self._resolution_combo = QComboBox()
        self._resolution_combo.addItem("640x480 (Fast)", (640, 480))
        self._resolution_combo.addItem("1280x720 (HD)", (1280, 720))
        r2_layout.addWidget(self._resolution_combo)

        cam_vlayout.addLayout(r2_layout)

        # Row 3: Quick Toggles (Mirror, Landmarks, Bounding Boxes, Confidence Slider, Tools)
        r3_layout = QHBoxLayout()
        r3_layout.setSpacing(14)

        self._mirror_check = QCheckBox("🪞 Mirror Camera")
        self._mirror_check.setChecked(True)
        r3_layout.addWidget(self._mirror_check)

        self._landmarks_check = QCheckBox("👁 Hand Landmarks")
        self._landmarks_check.setChecked(True)
        r3_layout.addWidget(self._landmarks_check)

        self._boxes_check = QCheckBox("📦 Bounding Boxes")
        self._boxes_check.setChecked(True)
        r3_layout.addWidget(self._boxes_check)

        # Confidence Slider
        r3_layout.addWidget(QLabel("Confidence:"))
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(20, 90)
        self._conf_slider.setValue(40)
        self._conf_slider.setMaximumWidth(120)
        r3_layout.addWidget(self._conf_slider)
        self._conf_label = QLabel("40%")
        self._conf_label.setMinimumWidth(32)
        r3_layout.addWidget(self._conf_label)

        r3_layout.addStretch()

        # Screenshot & Video Recording buttons
        self._screenshot_btn = QPushButton("📸 Screenshot")
        self._screenshot_btn.setToolTip("Save current annotated frame to data/captures/")
        r3_layout.addWidget(self._screenshot_btn)

        self._record_btn = QPushButton("🔴 Record Video")
        self._record_btn.setToolTip("Record live stream to MP4 in data/captures/")
        r3_layout.addWidget(self._record_btn)

        cam_vlayout.addLayout(r3_layout)
        main_layout.addWidget(camera_panel)

        # ─── Main Content Splitter: Left Preview | Right Recognition / Translation ───
        content_splitter = QSplitter(Qt.Horizontal)

        # Left Column: Preview + Stats
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(8)

        self._preview = CameraPreview()
        left_layout.addWidget(self._preview, stretch=1)

        # Stats Cards
        stats_row = QHBoxLayout()
        self._fps_card = StatCard("FPS", "0.0")
        stats_row.addWidget(self._fps_card)
        self._latency_card = StatCard("Latency", "-- ms")
        stats_row.addWidget(self._latency_card)
        self._detections_card = StatCard("Detections", "0")
        stats_row.addWidget(self._detections_card)
        self._decision_card = StatCard("Status", "Ready")
        stats_row.addWidget(self._decision_card)
        left_layout.addLayout(stats_row)

        content_splitter.addWidget(left_widget)

        # Right Column: Output & Controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(8)

        # Translation / Detection Output Box
        caption_box = QFrame()
        caption_box.setObjectName("captionBox")
        caption_box.setStyleSheet("""
            QFrame#captionBox {
                background-color: #1a1a1a;
                border: 2px solid #007acc;
                border-radius: 8px;
                padding: 14px;
            }
        """)
        caption_layout = QVBoxLayout(caption_box)
        caption_layout.addWidget(QLabel("Live Output & Translation"))

        self._caption_label = QLabel("Select camera and click Connect to start")
        self._caption_label.setStyleSheet("font-size: 22px; font-weight: 700; color: #4ec9b0;")
        self._caption_label.setAlignment(Qt.AlignCenter)
        self._caption_label.setMinimumHeight(55)
        self._caption_label.setWordWrap(True)
        caption_layout.addWidget(self._caption_label)

        word_layout = QHBoxLayout()
        word_layout.addWidget(QLabel("Current Word / Sign:"))
        self._word_label = QLabel("")
        self._word_label.setStyleSheet("font-size: 18px; color: #e0e0e0; font-weight: 600;")
        word_layout.addWidget(self._word_label, stretch=1)
        caption_layout.addLayout(word_layout)

        right_layout.addWidget(caption_box)

        # Text manipulation buttons (Clear, Delete, Space)
        ctrl_bar = QHBoxLayout()
        self._clear_btn = QPushButton("🗑 Clear")
        self._clear_btn.setObjectName("dangerButton")
        ctrl_bar.addWidget(self._clear_btn)
        self._delete_btn = QPushButton("⌫ Delete")
        ctrl_bar.addWidget(self._delete_btn)
        self._space_btn = QPushButton("␣ Space")
        ctrl_bar.addWidget(self._space_btn)
        right_layout.addLayout(ctrl_bar)

        # Recognition & Detection History
        history_box = QGroupBox("Recognition & Detection History")
        hist_layout = QVBoxLayout(history_box)
        hist_layout.setContentsMargins(6, 8, 6, 6)
        self._history_list = QListWidget()
        hist_layout.addWidget(self._history_list)
        right_layout.addWidget(history_box, stretch=1)

        # Dialect selector row
        dialect_row = QHBoxLayout()
        dialect_row.addWidget(QLabel("Dialect:"))
        self._dialect_selector = DialectSelector()
        from src.domain.entities import DialectInfo
        dialects = [
            DialectInfo("iraqi", "Iraqi", "عراقية"),
            DialectInfo("saudi", "Saudi", "سعودية"),
            DialectInfo("egyptian", "Egyptian", "مصرية"),
        ]
        self._dialect_selector.populate(dialects)
        if self._dialect_manager:
            active_id = self._dialect_manager.active_dialect.value
            for idx in range(self._dialect_selector.count()):
                if self._dialect_selector.itemData(idx) == active_id:
                    self._dialect_selector.setCurrentIndex(idx)
                    break
        dialect_row.addWidget(self._dialect_selector, stretch=1)
        right_layout.addLayout(dialect_row)

        content_splitter.addWidget(right_widget)
        content_splitter.setSizes([580, 360])
        main_layout.addWidget(content_splitter, stretch=1)

    # ── Signal Connections ───────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # Camera controls
        self._camera_combo.currentIndexChanged.connect(self._on_camera_selected)
        self._refresh_cam_btn.clicked.connect(self._refresh_camera_list)
        self._connect_btn.clicked.connect(self._connect_camera)
        self._disconnect_btn.clicked.connect(self._disconnect_camera)

        # AI Mode & Engine
        self._ai_mode_combo.currentIndexChanged.connect(self._on_ai_mode_changed)
        self._engine_selector.currentIndexChanged.connect(self._on_engine_changed)
        self._vocab_btn.clicked.connect(self._show_karsl_vocab)

        # Start / Stop
        self._start_btn.clicked.connect(self._start_detection)
        self._stop_btn.clicked.connect(self._stop_detection)

        # Toggles & Adjustments
        self._mirror_check.toggled.connect(self._on_mirror_toggled)
        self._landmarks_check.toggled.connect(self._on_landmarks_toggled)
        self._boxes_check.toggled.connect(self._on_boxes_toggled)
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        self._resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)

        # Tools
        self._screenshot_btn.clicked.connect(self._take_screenshot)
        self._record_btn.clicked.connect(self._toggle_recording)

        # Text operations
        self._clear_btn.clicked.connect(self._clear_text)
        self._delete_btn.clicked.connect(self._delete_last)
        self._space_btn.clicked.connect(self._add_space)

        # Dialect
        self._dialect_selector.dialect_selected.connect(self._on_dialect_selected)

        # App state sync
        self._state.fps_updated.connect(lambda f: self._fps_card.set_value(f"{f:.1f}"))
        self._state.caption_updated.connect(self._caption_label.setText)
        self._state.pipeline_status_changed.connect(self._on_status_changed)

    # ── Camera Enumeration & Source Management ───────────────────────────

    def _refresh_camera_list(self) -> None:
        """Scan system devices and update the unified camera dropdown with all physical and virtual cameras."""
        self._state.log("info", "Enumerating connected video devices...")

        active_idx: Optional[int] = None
        if self._is_connected and self._source_uri.startswith("webcam:"):
            try:
                active_idx = int(self._source_uri.replace("webcam:", ""))
            except ValueError:
                pass

        cameras = self._camera_service.list_cameras(force_refresh=True, active_index=active_idx)

        prev_id = self._camera_combo.currentData()
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()

        selected_index = 0
        for i, cam in enumerate(cameras):
            icon = "📱" if cam.device_type == "virtual" else ("💻" if cam.device_type == "laptop" else "🔌")
            label = f"{icon}  {cam.display_label}"
            self._camera_combo.addItem(label, cam.id)
            if prev_id and cam.id == prev_id:
                selected_index = i

        if self._camera_combo.count() > 0:
            self._camera_combo.setCurrentIndex(selected_index)
            self._source_uri = self._camera_combo.currentData() or "webcam:0"

        self._camera_combo.blockSignals(False)
        self._state.log("info", f"Discovered {len(cameras)} video device(s).")

    def _on_camera_selected(self, index: int) -> None:
        """Handle camera selection change in unified dropdown."""
        if index < 0:
            return
        new_cam_id = self._camera_combo.currentData()
        if not new_cam_id:
            return

        # If not connected yet, update target source URI
        if not self._is_connected:
            self._source_uri = new_cam_id
            return

        # If already streaming from this exact device, ignore
        if new_cam_id == self._source_uri:
            return

        # Perform seamless live switching
        self._switch_to_camera(new_cam_id)

    def _switch_to_camera(self, new_cam_id: str) -> None:
        """Switch camera safely without freezing UI, leaking handles, or crashing."""
        was_detecting = self._is_detecting
        cam_name = self._camera_combo.currentText().strip()

        self._set_status_badge("connecting", "Switching...")
        self._caption_label.setText(f"Switching to {cam_name}...")
        self._state.log("info", f"Switching video source from {self._source_uri} to {new_cam_id}...")

        # 1. Stop UI frame polling timer
        self._timer.stop()

        # 2. Reset worker queue to clear any pending frames
        if self._worker:
            self._worker.reset_buffer()

        # 3. Stop and release previous video source handle
        self._source_use_cases.stop_source()

        # 4. Yield briefly for Windows driver/filter graph cleanup
        QCoreApplication.processEvents()
        time.sleep(0.08)

        # 5. Open new camera device
        self._source_type = VideoSourceType.WEBCAM
        self._source_uri = new_cam_id

        success = self._source_use_cases.start_source(self._source_type, self._source_uri)
        if success:
            self._is_connected = True
            self._connect_btn.setEnabled(False)
            self._disconnect_btn.setEnabled(True)
            self._start_btn.setEnabled(not was_detecting)
            self._stop_btn.setEnabled(was_detecting)
            self._is_detecting = was_detecting
            self._set_status_badge("connected", "Connected")
            self._caption_label.setText(f"Connected to {cam_name}")
            self._state.log("info", f"Switched to {new_cam_id} successfully.")

            # Apply selected resolution
            res_w, res_h = self._resolution_combo.currentData()
            self._source_use_cases.set_resolution(res_w, res_h)

            # Resume frame polling
            self._timer.start(33)
        else:
            self._is_connected = False
            self._is_detecting = False
            self._connect_btn.setEnabled(True)
            self._disconnect_btn.setEnabled(False)
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            self._set_status_badge("error", "Switch Error")
            self._caption_label.setText(f"Failed to open {cam_name}. Check device and click Refresh.")
            self._state.error(f"Failed to switch to {new_cam_id}")

    def _set_status_badge(self, status: str, text: str) -> None:
        colors = {
            "connected": ("#98c379", "🟢"),
            "connecting": ("#e5c07b", "🟡"),
            "disconnected": ("#9d9d9d", "⚪"),
            "error": ("#e06c75", "🔴"),
        }
        color, icon = colors.get(status, ("#9d9d9d", "⚪"))
        self._status_badge.setText(f"{icon} {text}")
        self._status_badge.setStyleSheet(f"""
            background-color: #1e1e1e;
            color: {color};
            border: 1px solid {color};
            border-radius: 4px;
            padding: 4px 10px;
            font-weight: 600;
        """)

    # ── Camera Connection / Disconnection ────────────────────────────────

    def _connect_camera(self) -> None:
        """Establish connection with the selected camera."""
        cam_id = self._camera_combo.currentData() or "webcam:0"
        cam_name = self._camera_combo.currentText().strip()

        self._set_status_badge("connecting", "Connecting...")
        self._caption_label.setText(f"Connecting to {cam_name}...")

        self._source_type = VideoSourceType.WEBCAM
        self._source_uri = cam_id

        success = self._source_use_cases.start_source(self._source_type, self._source_uri)
        if success:
            self._is_connected = True
            self._connect_btn.setEnabled(False)
            self._disconnect_btn.setEnabled(True)
            self._start_btn.setEnabled(True)
            self._set_status_badge("connected", "Connected")
            self._caption_label.setText(f"Connected to {cam_name}. Click 'Start Detection' to begin.")
            self._state.log("info", f"Successfully connected to camera: {self._source_uri}")

            # Apply selected resolution
            res_w, res_h = self._resolution_combo.currentData()
            self._source_use_cases.set_resolution(res_w, res_h)

            # Automatically start frame polling
            self._timer.start(33)
        else:
            self._is_connected = False
            self._set_status_badge("error", "Connection Error")
            self._caption_label.setText(f"Failed to connect to {cam_name}. Check device or click Refresh.")
            self._state.error(f"Failed to connect to {self._source_uri}")

    def _disconnect_camera(self) -> None:
        """Disconnect and release camera resources cleanly."""
        self._stop_detection()
        self._timer.stop()
        self._source_use_cases.stop_source()

        self._is_connected = False
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._set_status_badge("disconnected", "Disconnected")
        self._caption_label.setText("Camera disconnected.")
        self._fps_card.set_value("0.0")
        self._latency_card.set_value("-- ms")
        self._decision_card.set_value("Disconnected")

    # ── Detection Start / Stop ───────────────────────────────────────────

    def _start_detection(self) -> None:
        if not self._is_connected:
            self._connect_camera()
            if not self._is_connected:
                return

        self._is_detecting = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._decision_card.set_value("Running")

        mode = self._ai_mode_combo.currentData()
        if mode == "sign":
            self._caption_label.setText("Sign Language Recognition active — hold hand in view")
        elif mode == "yolo":
            self._caption_label.setText("Object Detection active — detecting objects")
        else:
            self._caption_label.setText("Combined Mode active — detecting signs and objects")

    def _stop_detection(self) -> None:
        self._is_detecting = False
        self._start_btn.setEnabled(self._is_connected)
        self._stop_btn.setEnabled(False)
        if self._worker:
            self._worker.reset_buffer()
        if self._is_recording:
            self._toggle_recording()
        self._decision_card.set_value("Paused")

    # ── Frame Ingestion & Processing ─────────────────────────────────────

    def _read_next_frame(self) -> None:
        """Fetch frame from video source use cases and forward to background worker."""
        if self._analytics:
            self._analytics.record_frame()

        frame_result = self._source_use_cases.read_frame()
        if frame_result is None or frame_result.data is None:
            return

        frame = frame_result.data  # RGB array

        # If detection is not running, just display raw frame directly
        if not self._is_detecting:
            display_frame = cv2.flip(frame, 1) if self._mirror_check.isChecked() else frame
            self._last_display_frame = display_frame
            self._preview.display_frame(display_frame)
            return

        # Frame skipping optimization if enabled
        if self._optimizer and not self._optimizer.should_process_frame():
            display_frame = cv2.flip(frame, 1) if self._mirror_check.isChecked() else frame
            self._preview.display_frame(display_frame)
            return

        # Submit to worker thread for asynchronous inference
        self._worker.submit_frame(frame)

    @Slot(object, object, list, dict, float)
    def _on_frame_processed(
        self,
        annotated_frame: np.ndarray,
        karsl_res: Optional[KArSLResult],
        yolo_detections: list[DetectedObject],
        vis_info: dict,
        elapsed_ms: float,
    ) -> None:
        """Handle annotated frame and inference results emitted by the background worker."""
        mode = self._ai_mode_combo.currentData()

        # Record video frame if recording is active
        if self._is_recording and self._video_writer is not None:
            bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            self._video_writer.write(bgr)

        # Draw recording watermark badge if active
        if self._is_recording and self._rec_indicator_visible:
            cv2.circle(annotated_frame, (28, 28), 8, (255, 50, 50), -1)
            cv2.putText(
                annotated_frame, "REC", (42, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2, cv2.LINE_AA,
            )

        self._last_display_frame = annotated_frame
        self._preview.display_frame(annotated_frame)
        self._latency_card.set_value(f"{elapsed_ms:.0f} ms")

        # ── 1. Sign Language Handling ──
        if mode in ("sign", "combined"):
            hands_cnt = 0
            if vis_info.get("rh"):
                hands_cnt += 1
            if vis_info.get("lh"):
                hands_cnt += 1

            if karsl_res is not None and karsl_res.pred_idx >= 0:
                conf = karsl_res.confidence
                ar = karsl_res.arabic_text
                en = karsl_res.english_text

                if karsl_res.is_new:
                    from src.domain.text import TextCommand
                    self._text_builder.process(TextCommand.word(ar))
                    self._update_caption()
                    self._detection_count += 1

                    self._caption_label.setText(f"{ar} ({en})")
                    self._word_label.setText(ar)
                    self._decision_card.set_value(f"Sign: {ar} ({conf:.0%})")

                    if self._analytics:
                        self._analytics.record_prediction(correct=True, delay_ms=elapsed_ms)
                        self._analytics.record_system_metrics()

                    item = QListWidgetItem(f"🤟  [Sign] {ar} ({en}) — {conf:.1%}")
                    self._history_list.insertItem(0, item)
                else:
                    if conf >= 0.20:
                        self._caption_label.setText(f"[{ar} ({en}) — {conf:.0%}]")
            else:
                buf = karsl_res.buffer_fill if karsl_res else 0
                if hands_cnt > 0 or vis_info.get("has_pose"):
                    self._decision_card.set_value(f"Buffering ({buf}/50)")
                    if not self._word_label.text():
                        self._caption_label.setText(f"[تجميع حركات اليدين {buf}/50]")
                else:
                    self._decision_card.set_value("No Hand")

        # ── 2. Object Detection Handling ──
        if mode in ("yolo", "combined"):
            obj_cnt = len(yolo_detections)
            if mode == "yolo":
                self._detections_card.set_value(f"{obj_cnt} object(s)")
                if obj_cnt > 0:
                    top_obj = max(yolo_detections, key=lambda o: o.confidence)
                    self._caption_label.setText(f"Detected: {top_obj.label} ({top_obj.confidence:.0%})")
                    self._decision_card.set_value(f"YOLO: {top_obj.label}")

                    # Add unique detected objects to history if not recent
                    for det in yolo_detections[:3]:
                        if det.confidence >= 0.50:
                            item = QListWidgetItem(f"📦  [Object] {det.label} ({det.confidence:.0%})")
                            # Avoid flooding history
                            if self._history_list.count() == 0 or self._history_list.item(0).text() != item.text():
                                self._history_list.insertItem(0, item)
                else:
                    self._caption_label.setText("No objects detected above threshold.")
                    self._decision_card.set_value("YOLO: Scanning...")
            else:
                # Combined mode
                if obj_cnt > 0:
                    self._detections_card.set_value(f"Objects: {obj_cnt}")

        if mode == "sign":
            self._detections_card.set_value(str(self._detection_count))

    # ── Controls, Toggles & Settings ─────────────────────────────────────

    def _on_ai_mode_changed(self, index: int) -> None:
        mode = self._ai_mode_combo.currentData()
        self._worker.mode = mode

        is_sign = mode in ("sign", "combined")
        self._engine_label.setVisible(is_sign)
        self._engine_selector.setVisible(is_sign)
        self._vocab_btn.setVisible(is_sign and self._engine_selector.currentData() == "karsl")

        self._state.log("info", f"Switched AI Recognition Mode to: {mode}")

    def _on_engine_changed(self, index: int) -> None:
        engine = self._engine_selector.currentData()
        self._worker.active_engine = engine
        self._vocab_btn.setVisible(engine == "karsl")

    def _on_mirror_toggled(self, checked: bool) -> None:
        self._worker.mirror = checked

    def _on_landmarks_toggled(self, checked: bool) -> None:
        self._worker.show_landmarks = checked

    def _on_boxes_toggled(self, checked: bool) -> None:
        self._worker.show_boxes = checked

    def _on_conf_changed(self, value: int) -> None:
        thresh = value / 100.0
        self._conf_label.setText(f"{value}%")
        self._worker.confidence_threshold = thresh

    def _on_resolution_changed(self, index: int) -> None:
        w, h = self._resolution_combo.currentData()
        self._source_use_cases.set_resolution(w, h)
        self._state.log("info", f"Target camera resolution requested: {w}x{h}")

    def _take_screenshot(self) -> None:
        """Capture and save the current live frame."""
        if self._last_display_frame is None:
            QMessageBox.information(self, "Screenshot", "No live frame available to capture.")
            return

        captures_dir = Path("data/captures")
        captures_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filepath = captures_dir / f"screenshot_{ts}.png"

        bgr = cv2.cvtColor(self._last_display_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), bgr)

        self._state.log("info", f"Screenshot saved to: {filepath}")
        QMessageBox.information(self, "Screenshot Saved", f"Saved successfully to:\n{filepath.resolve()}")

    def _toggle_recording(self) -> None:
        """Start or stop recording live frames to MP4."""
        if not self._is_recording:
            if self._last_display_frame is None:
                QMessageBox.warning(self, "Recording", "Cannot start recording: No camera stream active.")
                return

            captures_dir = Path("data/captures")
            captures_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            filepath = captures_dir / f"recording_{ts}.mp4"

            h, w = self._last_display_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(str(filepath), fourcc, 25.0, (w, h))

            if not self._video_writer.isOpened():
                QMessageBox.critical(self, "Recording Error", "Failed to initialize MP4 video encoder.")
                return

            self._is_recording = True
            self._record_btn.setText("⏹ Stop Recording")
            self._record_btn.setStyleSheet("background-color: #c94f4f; font-weight: 700;")
            self._record_timer.start(500)
            self._state.log("info", f"Started recording to: {filepath}")
        else:
            self._is_recording = False
            self._record_timer.stop()
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None

            self._record_btn.setText("🔴 Record Video")
            self._record_btn.setStyleSheet("")
            self._state.log("info", "Video recording stopped and saved.")
            QMessageBox.information(self, "Recording Saved", "Video recording completed and saved to data/captures/.")

    def _blink_rec_indicator(self) -> None:
        self._rec_indicator_visible = not self._rec_indicator_visible

    # ── Text Builder Helpers ─────────────────────────────────────────────

    def _clear_text(self) -> None:
        if self._text_builder:
            from src.domain.text import TextCommand
            self._text_builder.process(TextCommand.clear())
            self._update_caption()

    def _delete_last(self) -> None:
        if self._text_builder:
            from src.domain.text import TextCommand
            self._text_builder.process(TextCommand.delete())
            self._update_caption()

    def _add_space(self) -> None:
        if self._text_builder:
            from src.domain.text import TextCommand
            self._text_builder.process(TextCommand.space())
            self._update_caption()

    def _update_caption(self) -> None:
        if self._text_builder:
            self._state.last_caption = self._text_builder.full_text
            self._word_label.setText(self._text_builder.current_word)

    def _on_dialect_selected(self, dialect_id: str) -> None:
        if self._dialect_manager:
            try:
                did = DialectId(dialect_id)
                if self._dialect_manager.switch_dialect(did):
                    self._state.log("info", f"Dialect switched to: {dialect_id}")
            except ValueError:
                pass

    def _show_karsl_vocab(self) -> None:
        vocab = []
        if self._worker and self._worker.karsl_recognizer:
            vocab = self._worker.karsl_recognizer.get_vocabulary()
        dlg = KArSLVocabularyDialog(vocabulary=vocab if vocab else None, parent=self)
        dlg.exec()

    # ── Worker Callbacks ─────────────────────────────────────────────────

    def _on_karsl_loaded(self, success: bool, msg: str) -> None:
        if success:
            self._state.log("info", f"KArSL-502: {msg}")
        else:
            self._state.log("warning", f"KArSL-502: {msg}")

    def _on_yolo_loaded(self, success: bool, msg: str) -> None:
        if success:
            self._state.log("info", f"YOLO: {msg}")
        else:
            self._state.log("warning", f"YOLO: {msg}")

    def _on_worker_error(self, err: str) -> None:
        self._state.log("warning", f"Vision worker error: {err}")

    def _on_status_changed(self, status: str) -> None:
        if status == "error":
            self._disconnect_camera()

    def closeEvent(self, event) -> None:
        self._disconnect_camera()
        if self._worker:
            self._worker.stop()
        super().closeEvent(event)
