from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.domain.entities import VideoSourceType
from src.infrastructure.camera_service import OpenCVCameraService
from src.infrastructure.dataset.collector import DatasetCollector
from src.infrastructure.dataset.gesture_registry import GestureRegistry
from src.infrastructure.dataset.organizer import DatasetOrganizer
from src.infrastructure.hand_detection.detector import HandDetector, HandDetectorConfig
from src.infrastructure.hand_detection.renderer import HandRenderer
from src.presentation.dialogs.dataset_dialogs import (
    AddClassDialog,
    DeleteClassDialog,
    EditClassDialog,
    SampleInspectorDialog,
)
from src.presentation.widgets import CameraPreview, SectionHeader, StatCard


class DatasetCollectionTab(QWidget):
    """Modern, robust Dataset Studio for capturing and managing Arabic Sign Language datasets.

    Supports:
    - DirectShow Camera Selection (HD Webcam, e2eSoft iVCam, USB Cameras)
    - Static Sign Sample Capture (single & multi-sample burst with progress and cancellation)
    - Dynamic Temporal Sequence Recording (MediaPipe landmark sequence collection)
    - Comprehensive Class & Sample Management (Add, Edit, Delete, Sample Inspector, Stats)
    - Coordinated Camera Ownership preventing device conflicts
    """

    def __init__(
        self,
        state: AppState,
        source_use_cases: VideoSourceUseCases,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._source_use_cases = source_use_cases
        self._logger = logging.getLogger(__name__)

        # Core services
        self._collector = DatasetCollector()
        self._organizer = DatasetOrganizer()
        self._registry = GestureRegistry()
        self._camera_service = OpenCVCameraService()
        self._detector = HandDetector(HandDetectorConfig(
            min_detection_confidence=0.30,
            min_tracking_confidence=0.30,
        ))
        self._renderer = HandRenderer()

        # Capture & streaming state
        self._is_streaming = False
        self._first_frame_rendered = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_detection = None
        self._last_fps_time = time.perf_counter()
        self._fps_counter = 0

        # Multi-sample static capture state
        self._multi_capture_active = False
        self._target_sample_count = 1
        self._captured_sample_count = 0
        self._multi_capture_timer = QTimer(self)
        self._multi_capture_timer.timeout.connect(self._step_multi_capture)

        # Dynamic sequence recording state
        self._dynamic_recording_active = False
        self._recorded_sequence: list[np.ndarray] = []
        self._sequence_start_time = 0.0

        # Frame reading timer (~30 FPS)
        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._read_camera_frame)

        self._setup_ui()
        self._refresh_camera_list()
        self._connect_signals()
        self._scan_and_populate_classes()

    # ── UI Construction ──────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(12)

        header = SectionHeader(
            "Dataset Studio",
            "Capture, review, and manage training samples for static signs and dynamic sequences",
        )
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ═════════════════════════════════════════════════════════════════
        # LEFT PANEL: Camera Viewport & Capture Controls
        # ═════════════════════════════════════════════════════════════════
        left_panel = QFrame()
        left_panel.setObjectName("leftCapturePanel")
        left_panel.setStyleSheet("""
            QFrame#leftCapturePanel {
                background-color: #181d24;
                border: 1px solid #2b3441;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Camera selection bar
        cam_bar = QHBoxLayout()
        cam_bar.setSpacing(8)

        cam_label = QLabel("Camera:")
        cam_label.setStyleSheet("font-weight: 600; color: #94a3b8;")
        cam_bar.addWidget(cam_label)

        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(220)
        self._camera_combo.setToolTip("Select camera (HD Webcam, e2eSoft iVCam, or USB)")
        cam_bar.addWidget(self._camera_combo, stretch=1)

        self._refresh_cam_btn = QPushButton("Refresh")
        self._refresh_cam_btn.setToolTip("Scan system for newly connected cameras")
        cam_bar.addWidget(self._refresh_cam_btn)

        self._cam_start_btn = QPushButton("Start Camera")
        self._cam_start_btn.setObjectName("primaryButton")
        self._cam_start_btn.setMinimumWidth(100)
        cam_bar.addWidget(self._cam_start_btn)

        self._cam_stop_btn = QPushButton("Stop Camera")
        self._cam_stop_btn.setEnabled(False)
        self._cam_stop_btn.setMinimumWidth(100)
        cam_bar.addWidget(self._cam_stop_btn)

        left_layout.addLayout(cam_bar)

        # Camera Preview
        self._preview = CameraPreview()
        self._preview.setMinimumSize(440, 320)
        left_layout.addWidget(self._preview, stretch=1)

        # Camera Status & Telemetry Row
        cam_status_row = QHBoxLayout()
        self._status_badge = QLabel("Camera Ready")
        self._status_badge.setStyleSheet("""
            background-color: #14181f;
            color: #94a3b8;
            border: 1px solid #2b3441;
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 11px;
            font-weight: 600;
        """)
        cam_status_row.addWidget(self._status_badge)

        self._fps_indicator = QLabel("FPS: --")
        self._fps_indicator.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 500;")
        cam_status_row.addWidget(self._fps_indicator)
        cam_status_row.addStretch()

        left_layout.addLayout(cam_status_row)

        # Hands & Readiness Stat Row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self._hands_card = StatCard("Hands Detected", "0")
        stats_row.addWidget(self._hands_card)
        self._readiness_card = StatCard("Hand Readiness", "Waiting for Camera")
        stats_row.addWidget(self._readiness_card)
        left_layout.addLayout(stats_row)

        # Capture Control Card
        capture_controls = QFrame()
        capture_controls.setObjectName("captureControlsCard")
        capture_controls.setStyleSheet("""
            QFrame#captureControlsCard {
                background-color: #14181f;
                border: 1px solid #2b3441;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        capture_layout = QVBoxLayout(capture_controls)
        capture_layout.setSpacing(8)

        # Capture buttons row
        btn_row = QHBoxLayout()
        self._capture_btn = QPushButton("📸  Capture Static Sign")
        self._capture_btn.setObjectName("successButton")
        self._capture_btn.setStyleSheet("""
            QPushButton#successButton {
                background-color: #059669;
                color: #ffffff;
                font-weight: 600;
                padding: 8px 14px;
                border-radius: 4px;
            }
            QPushButton#successButton:hover {
                background-color: #10b981;
            }
        """)
        btn_row.addWidget(self._capture_btn)

        self._batch_capture_btn = QPushButton("📹  Record Dynamic Sequence")
        self._batch_capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                font-weight: 600;
                padding: 8px 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
        """)
        btn_row.addWidget(self._batch_capture_btn)

        self._cancel_capture_btn = QPushButton("Cancel")
        self._cancel_capture_btn.setObjectName("dangerButton")
        self._cancel_capture_btn.setVisible(False)
        self._cancel_capture_btn.clicked.connect(self._cancel_capture)
        btn_row.addWidget(self._cancel_capture_btn)

        capture_layout.addLayout(btn_row)

        # Sample count selector & Progress
        count_layout = QHBoxLayout()
        count_label = QLabel("Samples to record:")
        count_label.setStyleSheet("color: #94a3b8;")
        count_layout.addWidget(count_label)

        self._samples_spin = QSpinBox()
        self._samples_spin.setRange(1, 500)
        self._samples_spin.setValue(1)
        self._samples_spin.setSuffix(" sample(s)")
        count_layout.addWidget(self._samples_spin)

        self._capture_progress = QProgressBar()
        self._capture_progress.setRange(0, 100)
        self._capture_progress.setValue(0)
        self._capture_progress.setFixedHeight(6)
        self._capture_progress.setVisible(False)
        count_layout.addWidget(self._capture_progress, stretch=1)

        capture_layout.addLayout(count_layout)
        left_layout.addWidget(capture_controls)

        splitter.addWidget(left_panel)

        # ═════════════════════════════════════════════════════════════════
        # RIGHT PANEL: Gesture Classes & Dataset Management
        # ═════════════════════════════════════════════════════════════════
        right_panel = QFrame()
        right_panel.setObjectName("rightClassPanel")
        right_panel.setStyleSheet("""
            QFrame#rightClassPanel {
                background-color: #181d24;
                border: 1px solid #2b3441;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Class list header & action buttons
        class_header = QHBoxLayout()
        self._class_header_label = QLabel("Gesture Classes")
        self._class_header_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #f8fafc;")
        class_header.addWidget(self._class_header_label)
        class_header.addStretch()

        self._add_class_btn = QPushButton("+ Add Class")
        self._add_class_btn.setToolTip("Add new Arabic sign class")
        class_header.addWidget(self._add_class_btn)

        self._edit_class_btn = QPushButton("✏️ Rename / Edit")
        self._edit_class_btn.setToolTip("Rename or edit category of selected class")
        self._edit_class_btn.setEnabled(False)
        class_header.addWidget(self._edit_class_btn)

        self._delete_class_btn = QPushButton("🗑️ Delete / Archive")
        self._delete_class_btn.setToolTip("Delete or archive selected class")
        self._delete_class_btn.setEnabled(False)
        class_header.addWidget(self._delete_class_btn)

        right_layout.addLayout(class_header)

        # Simple Class Organization (Search & Filter bar)
        search_filter_box = QVBoxLayout()
        search_filter_box.setSpacing(6)

        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("🔍  Search classes (Arabic text)...")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setStyleSheet("""
            QLineEdit {
                background-color: #0f1319;
                border: 1px solid #2b3441;
                border-radius: 4px;
                padding: 6px 10px;
                color: #f8fafc;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #0d9488;
            }
        """)
        search_row.addWidget(self._search_edit, stretch=1)

        self._sort_btn = QPushButton("Sort A-Z")
        self._sort_btn.setCheckable(True)
        self._sort_btn.setChecked(True)
        self._sort_btn.setToolTip("Toggle alphabetical sorting")
        self._sort_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #94a3b8;
                padding: 5px 10px;
                font-size: 11px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #0f766e;
                color: #ffffff;
                border-color: #14b8a6;
            }
        """)
        search_row.addWidget(self._sort_btn)
        search_filter_box.addLayout(search_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)

        status_lbl = QLabel("Filter:")
        status_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        filter_row.addWidget(status_lbl)

        self._status_filter_combo = QComboBox()
        self._status_filter_combo.addItem("All Classes", "all")
        self._status_filter_combo.addItem("With Samples Only", "with_samples")
        self._status_filter_combo.addItem("Empty Only", "empty")
        filter_row.addWidget(self._status_filter_combo, stretch=1)

        type_lbl = QLabel("Type:")
        type_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        filter_row.addWidget(type_lbl)

        self._type_filter_combo = QComboBox()
        self._type_filter_combo.addItem("All Types", "all")
        self._type_filter_combo.addItem("Static Signs", "static")
        self._type_filter_combo.addItem("Dynamic Sequences", "dynamic")
        self._type_filter_combo.addItem("Both Types", "both")
        self._type_filter_combo.addItem("Alphabet Letters", "alphabet")
        filter_row.addWidget(self._type_filter_combo, stretch=1)

        search_filter_box.addLayout(filter_row)
        right_layout.addLayout(search_filter_box)

        # Classes List Widget
        self._class_list = QListWidget()
        self._class_list.setStyleSheet("""
            QListWidget {
                background-color: #0f1319;
                border: 1px solid #2b3441;
                border-radius: 6px;
                padding: 6px;
                color: #f8fafc;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 6px 10px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #1e3a5f;
                color: #38bdf8;
                font-weight: 600;
            }
        """)
        right_layout.addWidget(self._class_list, stretch=1)

        # Class Action Strip
        class_action_bar = QHBoxLayout()
        self._inspect_btn = QPushButton("🔍 Inspect Samples")
        self._inspect_btn.setToolTip("View, inspect, and delete individual samples for selected class")
        self._inspect_btn.setEnabled(False)
        class_action_bar.addWidget(self._inspect_btn)

        self._refresh_stats_btn = QPushButton("🔄 Refresh Stats")
        self._refresh_stats_btn.setToolTip("Rescan dataset from disk and update statistics")
        class_action_bar.addWidget(self._refresh_stats_btn)

        right_layout.addLayout(class_action_bar)

        # Dataset Statistics Cards
        stats_grid = QHBoxLayout()
        self._total_card = StatCard("Total Samples", "0")
        stats_grid.addWidget(self._total_card)
        self._classes_card = StatCard("Classes", "0")
        stats_grid.addWidget(self._classes_card)
        self._split_card = StatCard("Static / Dynamic", "0 / 0")
        stats_grid.addWidget(self._split_card)
        self._latest_card = StatCard("Latest Action", "Ready")
        stats_grid.addWidget(self._latest_card)

        right_layout.addLayout(stats_grid)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 56)
        splitter.setStretchFactor(1, 44)

        layout.addWidget(splitter, stretch=1)

    # ── Signal Connections ───────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._refresh_cam_btn.clicked.connect(self._refresh_camera_list)
        self._cam_start_btn.clicked.connect(self._start_camera)
        self._cam_stop_btn.clicked.connect(self._stop_camera)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_combo_changed)

        # Capture actions
        self._capture_btn.clicked.connect(self._on_capture_clicked)
        self._batch_capture_btn.clicked.connect(self._toggle_record_sequence)

        # Class management & search/filter
        self._add_class_btn.clicked.connect(self._add_class)
        self._edit_class_btn.clicked.connect(self._edit_class)
        self._delete_class_btn.clicked.connect(self._delete_class)
        self._inspect_btn.clicked.connect(self._inspect_samples)
        self._refresh_stats_btn.clicked.connect(self._scan_and_populate_classes)
        self._class_list.itemSelectionChanged.connect(self._on_class_selection_changed)
        self._search_edit.textChanged.connect(self._apply_filters_and_populate)
        self._status_filter_combo.currentIndexChanged.connect(self._apply_filters_and_populate)
        self._type_filter_combo.currentIndexChanged.connect(self._apply_filters_and_populate)
        self._sort_btn.toggled.connect(self._apply_filters_and_populate)

    # ── Camera Operations ────────────────────────────────────────────────

    def _refresh_camera_list(self) -> None:
        """Enumerate system cameras via DirectShow COM and populate dropdown."""
        current_sel = self._camera_combo.currentData()
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()

        # Check if state has remembered camera selection
        saved_idx = getattr(self._state, "selected_camera_index", None)
        target_idx = current_sel if current_sel is not None else saved_idx

        cameras = self._camera_service.list_cameras(force_refresh=True, active_index=current_sel if self._is_streaming else None)
        if not cameras:
            self._camera_combo.addItem("No cameras found", None)
            self._camera_combo.setEnabled(False)
            self._cam_start_btn.setEnabled(False)
            self._status_badge.setText("No Camera Detected")
            self._status_badge.setStyleSheet("background-color: #3b1d1d; color: #ef4444; border: 1px solid #7f1d1d; border-radius: 4px; padding: 3px 8px;")
        else:
            self._camera_combo.setEnabled(True)
            self._cam_start_btn.setEnabled(True)
            pick_idx = 0
            for i, cam in enumerate(cameras):
                label = f"{cam.name}"
                if cam.device_type == "laptop":
                    label += " (Laptop Camera)"
                elif "ivcam" in cam.name.lower() or cam.device_type == "virtual":
                    label += " (iPhone / Virtual)"
                elif cam.device_type == "usb":
                    label += " (USB Camera)"

                self._camera_combo.addItem(label, cam.index)
                if target_idx is not None and cam.index == target_idx:
                    pick_idx = i

            self._camera_combo.setCurrentIndex(pick_idx)
            if not self._is_streaming:
                self._status_badge.setText("Camera Ready")
                self._status_badge.setStyleSheet("background-color: #14181f; color: #94a3b8; border: 1px solid #2b3441; border-radius: 4px; padding: 3px 8px;")

        self._camera_combo.blockSignals(False)

    def _start_camera(self) -> None:
        """Start local camera stream for dataset collection."""
        idx = self._camera_combo.currentData()
        if idx is None:
            return

        self._first_frame_rendered = False
        self._status_badge.setText("Connecting...")
        self._status_badge.setStyleSheet("background-color: #3b2d14; color: #f59e0b; border: 1px solid #78350f; border-radius: 4px; padding: 3px 8px;")
        self._preview.set_connecting(title="Connecting to Camera...", subtitle=f"Initializing DirectShow device (Index {idx})")

        # Remember selection
        self._state.selected_camera_index = idx

        # Ensure any active streaming on source is cleanly stopped first
        self._source_use_cases.stop()

        self._source_use_cases.select_source(
            VideoSourceType.WEBCAM,
            f"webcam:{idx}",
            device_index=idx,
            resolution=(640, 480),
        )

        success = self._source_use_cases.start()
        if success:
            self._is_streaming = True
            self._cam_start_btn.setEnabled(False)
            self._cam_stop_btn.setEnabled(True)
            self._cam_timer.start(30)
            self._readiness_card.set_value("Searching for hand...")
            self._fps_counter = 0
            self._last_fps_time = time.perf_counter()
        else:
            self._status_badge.setText("Failed to Open")
            self._status_badge.setStyleSheet("background-color: #3b1d1d; color: #ef4444; border: 1px solid #7f1d1d; border-radius: 4px; padding: 3px 8px;")
            self._preview.set_error(title="Camera Error", subtitle=f"Failed to acquire video stream from device {idx}")

    def _stop_camera(self) -> None:
        """Stop local camera stream and clean up resources."""
        self._cam_timer.stop()
        self._cancel_capture()
        if self._dynamic_recording_active:
            self._dynamic_recording_active = False
            self._batch_capture_btn.setText("📹  Record Dynamic Sequence")
            self._batch_capture_btn.setStyleSheet("")

        self._source_use_cases.stop()
        self._is_streaming = False
        self._first_frame_rendered = False

        self._cam_start_btn.setEnabled(True)
        self._cam_stop_btn.setEnabled(False)
        self._preview.set_standby()
        self._status_badge.setText("Camera Ready")
        self._status_badge.setStyleSheet("background-color: #14181f; color: #94a3b8; border: 1px solid #2b3441; border-radius: 4px; padding: 3px 8px;")
        self._readiness_card.set_value("Camera Stopped")
        self._hands_card.set_value("0")
        self._fps_indicator.setText("FPS: --")

    def _on_camera_combo_changed(self, index: int) -> None:
        """Handle user changing camera selection in dropdown."""
        if index < 0:
            return
        idx = self._camera_combo.currentData()
        if idx is not None:
            self._state.selected_camera_index = idx

        if self._is_streaming:
            self._stop_camera()
            QTimer.singleShot(100, self._start_camera)

    def _read_camera_frame(self) -> None:
        """Timer callback reading frames from active camera."""
        if not self._is_streaming:
            return

        frame_result = self._source_use_cases.read_frame()
        if frame_result is None:
            return

        # Safe extraction of raw BGR frame
        if hasattr(frame_result, "metadata") and "raw_bgr" in frame_result.metadata:
            raw_bgr = frame_result.metadata["raw_bgr"]
        elif hasattr(frame_result, "data"):
            raw_bgr = frame_result.data
        elif isinstance(frame_result, np.ndarray):
            raw_bgr = frame_result
        else:
            return

        if raw_bgr is not None and getattr(raw_bgr, "size", 0) > 0:
            self._process_frame_data(raw_bgr)

    def _process_frame_data(self, frame: np.ndarray) -> None:
        """Run MediaPipe hand detection, render visual skeleton, and update state."""
        self._last_frame = frame

        # Calculate FPS
        self._fps_counter += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            current_fps = self._fps_counter / elapsed
            self._fps_indicator.setText(f"FPS: {current_fps:.1f}")
            self._fps_counter = 0
            self._last_fps_time = now

        # Run hand detection on RGB frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            result = self._detector.detect(frame_rgb)
            self._last_detection = result
        except Exception as e:
            self._logger.error(f"Detection error: {e}")
            self._preview.display_frame(frame_rgb)
            return

        # Render skeleton & landmarks
        annotated_bgr = self._renderer.draw(frame.copy(), result)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        self._preview.display_frame(annotated_rgb)

        # Transition badge to live streaming upon first valid frame
        if not self._first_frame_rendered:
            self._first_frame_rendered = True
            self._status_badge.setText("● Live Streaming")
            self._status_badge.setStyleSheet(
                "background-color: #064e3b; color: #10b981; border: 1px solid #047857; "
                "border-radius: 4px; padding: 3px 8px; font-weight: 600;"
            )

        # Update Hands & Readiness status synchronously
        if result.has_hands:
            hand = result.hands[0]
            self._hands_card.set_value(f"{len(result.hands)} hand(s)")
            self._readiness_card.set_value(f"Ready ({hand.handedness.value})")
        else:
            self._hands_card.set_value("0")
            self._readiness_card.set_value("No hand detected")

        # Accumulate frames during dynamic sequence recording
        if self._dynamic_recording_active and result.has_hands:
            # Extract primary hand landmarks (21, 3)
            primary_hand = None
            for h in result.hands:
                if h.handedness.value == "Right":
                    primary_hand = h
                    break
            if primary_hand is None:
                primary_hand = result.hands[0]

            lm_array = np.array([[p.x, p.y, p.z] for p in primary_hand.landmarks], dtype=np.float32)
            self._recorded_sequence.append(lm_array)
            duration = time.time() - self._sequence_start_time
            self._latest_card.set_value(f"Rec: {len(self._recorded_sequence)} fr ({duration:.1f}s)")

    # ── Static Sign Capture ──────────────────────────────────────────────

    def _get_selected_class_label(self) -> Optional[str]:
        """Extract clean class label from currently selected item in list."""
        item = self._class_list.currentItem()
        if not item:
            return None
        val = item.data(Qt.UserRole)
        if val:
            return str(val).strip()
        raw_text = item.text().strip()
        clean = re.sub(r'\s*\[[^\]]*\]\s*', '', raw_text).strip()
        clean = re.sub(r'\s*\([^)]*\)\s*', '', clean).strip()
        clean = re.sub(r'⚠️\s*', '', clean).strip()
        return clean or None

    @Slot()
    def _on_capture_clicked(self) -> None:
        """Trigger capture of static sign sample(s)."""
        if not self._is_streaming or self._last_frame is None:
            QMessageBox.warning(self, "Camera Required", "Please start the camera before capturing samples.")
            return

        label = self._get_selected_class_label()
        if not label:
            QMessageBox.warning(self, "Select Class", "Please select a gesture class from the list first.")
            return

        # Check hand detection
        if not self._last_detection or not self._last_detection.has_hands:
            QMessageBox.warning(
                self, "No Hand Detected",
                "No hand was detected in the camera frame.\nPlease hold your hand clearly inside the frame."
            )
            return

        count = self._samples_spin.value()
        if count == 1:
            # Single sample capture
            success = self._save_single_static_sample(label)
            if success:
                self._latest_card.set_value(f"Saved: {label}")
                self._update_stats_and_list()
        else:
            # Multi-sample continuous capture
            self._start_multi_capture(label, count)

    def _save_single_static_sample(self, label: str) -> bool:
        """Capture and save one valid static sign sample to disk with verification."""
        if not self._last_detection or not self._last_detection.has_hands:
            return False

        hand = self._last_detection.hands[0]
        lm_array = np.array([[p.x, p.y, p.z] for p in hand.landmarks], dtype=np.float32)
        if lm_array.shape != (21, 3) or np.isnan(lm_array).any():
            return False

        sanitized = self._organizer.get_class_dir("words", label.replace(" ", "_"))
        sample_id = uuid.uuid4().hex[:8]
        sample_dir = sanitized / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save 3D landmarks array (1, 21, 3)
            np.save(sample_dir / "landmarks.npy", lm_array[np.newaxis, :, :])

            # Save metadata JSON (UTF-8)
            meta = {
                "sample_id": sample_id,
                "gesture_type": "static_word",
                "gesture_label": label,
                "dialect": "iraqi",
                "timestamp": time.time(),
                "num_frames": 1,
                "num_hands": len(self._last_detection.hands),
                "handedness": hand.handedness.value,
                "duration_seconds": 0.0,
                "fps": 30.0,
                "has_motion": False,
                "total_distance": 0.0,
                "motion_energy": 0.0,
            }
            import json
            with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            # Re-verify persistence and data integrity
            check = np.load(sample_dir / "landmarks.npy")
            if check.shape != (1, 21, 3):
                import shutil
                shutil.rmtree(sample_dir, ignore_errors=True)
                return False

            self._logger.info(f"Saved static sample for '{label}' to {sample_dir}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save static sample: {e}")
            import shutil
            shutil.rmtree(sample_dir, ignore_errors=True)
            return False

    def _start_multi_capture(self, label: str, count: int) -> None:
        """Start multi-sample continuous capture session."""
        self._multi_capture_active = True
        self._target_sample_count = count
        self._captured_sample_count = 0
        self._current_capture_label = label

        self._capture_btn.setEnabled(False)
        self._batch_capture_btn.setEnabled(False)
        self._cancel_capture_btn.setVisible(True)
        self._capture_progress.setVisible(True)
        self._capture_progress.setRange(0, count)
        self._capture_progress.setValue(0)

        self._latest_card.set_value(f"Capturing: 0/{count}")
        self._multi_capture_timer.start(250)  # Capture every 250ms

    def _step_multi_capture(self) -> None:
        """Timer step for continuous multi-sample recording."""
        if not self._multi_capture_active:
            return

        # Check hand presence
        if self._last_detection and self._last_detection.has_hands:
            success = self._save_single_static_sample(self._current_capture_label)
            if success:
                self._captured_sample_count += 1
                self._capture_progress.setValue(self._captured_sample_count)
                self._latest_card.set_value(
                    f"Capturing: {self._captured_sample_count}/{self._target_sample_count}"
                )

                if self._captured_sample_count >= self._target_sample_count:
                    self._finish_multi_capture(success=True)
        else:
            self._latest_card.set_value(
                f"Waiting for hand... ({self._captured_sample_count}/{self._target_sample_count})"
            )

    def _cancel_capture(self) -> None:
        """Cancel an in-progress multi-sample capture."""
        if self._multi_capture_active:
            self._finish_multi_capture(success=False)

    def _finish_multi_capture(self, success: bool) -> None:
        """Complete multi-sample capture and reset UI state."""
        self._multi_capture_active = False
        self._multi_capture_timer.stop()

        self._capture_btn.setEnabled(True)
        self._batch_capture_btn.setEnabled(True)
        self._cancel_capture_btn.setVisible(False)
        self._capture_progress.setVisible(False)

        self._update_stats_and_list()
        label = getattr(self, "_current_capture_label", "")
        if success:
            self._latest_card.set_value(f"Saved {self._captured_sample_count} samples")
            QMessageBox.information(
                self, "Capture Complete",
                f"Successfully recorded and validated {self._captured_sample_count} samples for «{label}»."
            )
        else:
            self._latest_card.set_value(f"Cancelled: {self._captured_sample_count} saved")

    # ── Dynamic Sequence Recording ───────────────────────────────────────

    @Slot()
    def _toggle_record_sequence(self) -> None:
        """Toggle recording of dynamic gesture sequence (multi-frame motion)."""
        if self._dynamic_recording_active:
            # Stop and save sequence
            self._stop_and_save_sequence()
        else:
            # Start dynamic recording
            self._start_dynamic_recording()

    def _start_dynamic_recording(self) -> None:
        """Initiate dynamic sequence recording."""
        if not self._is_streaming or self._last_frame is None:
            QMessageBox.warning(self, "Camera Required", "Please start the camera before recording a sequence.")
            return

        label = self._get_selected_class_label()
        if not label:
            QMessageBox.warning(self, "Select Class", "Please select a gesture class first.")
            return

        self._dynamic_recording_active = True
        self._recorded_sequence.clear()
        self._sequence_start_time = time.time()
        self._current_sequence_label = label

        self._batch_capture_btn.setText("🛑  Stop Recording")
        self._batch_capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: #ffffff;
                font-weight: 700;
                padding: 8px 14px;
                border-radius: 4px;
            }
        """)
        self._capture_btn.setEnabled(False)
        self._latest_card.set_value("Recording sequence...")

    def _stop_and_save_sequence(self) -> None:
        """Stop dynamic sequence recording, validate quality, and save to disk."""
        self._dynamic_recording_active = False
        self._batch_capture_btn.setText("📹  Record Dynamic Sequence")
        self._batch_capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                font-weight: 600;
                padding: 8px 14px;
                border-radius: 4px;
            }
        """)
        self._capture_btn.setEnabled(True)

        frames_count = len(self._recorded_sequence)
        duration = time.time() - self._sequence_start_time
        label = getattr(self, "_current_sequence_label", "gesture")

        # Dynamic model quality requirement: minimum 5 frames
        if frames_count < 5:
            self._latest_card.set_value(f"Rejected: {frames_count} fr (<5)")
            QMessageBox.warning(
                self, "Sequence Rejected",
                f"Sequence rejected: Only {frames_count} valid hand frames recorded.\n"
                f"Dynamic sign models require at least 5 frames of continuous hand movement."
            )
            return

        # Stack into 3D array (T, 21, 3)
        try:
            seq_array = np.stack(self._recorded_sequence, axis=0)
            if seq_array.shape[1:] != (21, 3) or np.isnan(seq_array).any():
                self._latest_card.set_value("Rejected: Invalid data")
                return

            sanitized_dir = self._organizer.get_class_dir("dynamic", label.replace(" ", "_"))
            sample_id = uuid.uuid4().hex[:8]
            sample_dir = sanitized_dir / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            np.save(sample_dir / "landmarks.npy", seq_array)

            # Metadata JSON
            meta = {
                "sample_id": sample_id,
                "gesture_type": "dynamic_gesture",
                "gesture_label": label,
                "dialect": "iraqi",
                "timestamp": time.time(),
                "num_frames": frames_count,
                "num_hands": 1,
                "handedness": "Right",
                "duration_seconds": duration,
                "fps": frames_count / max(duration, 0.001),
                "has_motion": True,
                "total_distance": 0.0,
                "motion_energy": 0.0,
            }
            import json
            with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            # Re-verify disk write
            check = np.load(sample_dir / "landmarks.npy")
            if check.shape[0] != frames_count:
                import shutil
                shutil.rmtree(sample_dir, ignore_errors=True)
                self._latest_card.set_value("Save Error")
                return

            self._latest_card.set_value(f"Saved seq ({frames_count} fr)")
            self._update_stats_and_list()
            QMessageBox.information(
                self, "Sequence Saved",
                f"Saved dynamic sequence ({frames_count} frames, {duration:.1f}s) for «{label}»."
            )
        except Exception as e:
            self._logger.error(f"Failed to save dynamic sequence: {e}")
            self._latest_card.set_value("Save Error")

    # ── Class & Dataset Management ───────────────────────────────────────

    def _scan_and_populate_classes(self) -> None:
        """Reload registry and populate list with real-time filters and sample counts."""
        self._registry.load_or_initialize()
        self._apply_filters_and_populate()

    def _update_stats_and_list(self) -> None:
        """Update sample counts and active list items."""
        self._apply_filters_and_populate()

    def _apply_filters_and_populate(self) -> None:
        """Filter, sort, and display active gesture classes in the list widget."""
        current_selection = self._get_selected_class_label()
        self._class_list.clear()

        query = self._search_edit.text().strip() if hasattr(self, "_search_edit") else ""
        status_filter = self._status_filter_combo.currentData() if hasattr(self, "_status_filter_combo") else "all"
        type_filter = self._type_filter_combo.currentData() if hasattr(self, "_type_filter_combo") else "all"
        sort_az = self._sort_btn.isChecked() if hasattr(self, "_sort_btn") else True

        filtered_classes = self._registry.filter_classes(
            query=query,
            status_filter=status_filter or "all",
            category_filter=type_filter or "all",
            sort_alphabetical=sort_az,
        )

        select_row = 0
        row = 0
        total_samples = 0
        total_static = 0
        total_dynamic = 0

        # Global totals across all active classes
        all_active = self._registry.list_active_classes()
        for c in all_active:
            counts = self._registry.get_sample_counts(c.name)
            total_samples += counts["total"]
            total_static += counts["static"]
            total_dynamic += counts["dynamic"]

        for cls_obj, counts in filtered_classes:
            static_cnt = counts["static"]
            dynamic_cnt = counts["dynamic"]
            cls_total = counts["total"]

            status_note = " ⚠️ (<2)" if cls_total < 2 else ""
            cat_display = cls_obj.category.capitalize()
            item_text = f"  {cls_obj.name}   [{cat_display}]   ({static_cnt} static, {dynamic_cnt} dynamic){status_note}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, cls_obj.name)
            item.setData(Qt.UserRole + 1, cls_obj.id)
            self._class_list.addItem(item)

            if current_selection and cls_obj.name == current_selection:
                select_row = row
            row += 1

        if self._class_list.count() > 0:
            self._class_list.setCurrentRow(select_row)

        self._on_class_selection_changed()

        self._total_card.set_value(str(total_samples))
        self._classes_card.set_value(str(len(all_active)))
        self._split_card.set_value(f"{total_static} / {total_dynamic}")
        if hasattr(self, "_class_header_label"):
            self._class_header_label.setText(f"Gesture Classes ({len(filtered_classes)} of {len(all_active)})")

    def _on_class_selection_changed(self) -> None:
        """Handle class selection change, updating action buttons state."""
        has_sel = self._class_list.currentItem() is not None
        self._edit_class_btn.setEnabled(has_sel)
        self._delete_class_btn.setEnabled(has_sel)
        self._inspect_btn.setEnabled(has_sel)

        label = self._get_selected_class_label()
        if label:
            self._logger.debug(f"Selected gesture class: {label}")

    @Slot()
    def _add_class(self) -> None:
        """Open AddClassDialog and create new gesture class via GestureRegistry."""
        active_names = {c.name for c in self._registry.list_active_classes()}
        dlg = AddClassDialog(existing_names=active_names, parent=self)
        if dlg.exec():
            name, cat_type, desc = dlg.get_full_data()
            try:
                cls_obj = self._registry.add_class(name, category=cat_type, description=desc)
                self._scan_and_populate_classes()

                # Select newly added class
                for idx in range(self._class_list.count()):
                    item = self._class_list.item(idx)
                    if item.data(Qt.UserRole) == cls_obj.name:
                        self._class_list.setCurrentRow(idx)
                        break

                self._latest_card.set_value(f"Added: {name}")
                QMessageBox.information(
                    self,
                    "Class Added",
                    f"Gesture class «{name}» ({cat_type}) created successfully.",
                )
            except Exception as e:
                self._logger.error(f"Failed to add class '{name}': {e}")
                QMessageBox.critical(self, "Error Adding Class", str(e))

    @Slot()
    def _edit_class(self) -> None:
        """Open EditClassDialog to rename or change category of the selected class."""
        current_name = self._get_selected_class_label()
        if not current_name:
            QMessageBox.information(self, "Select Class", "Please select a class to edit.")
            return

        cls_obj = self._registry.get_class(current_name)
        if not cls_obj:
            QMessageBox.warning(self, "Class Not Found", f"Class «{current_name}» not found in registry.")
            return

        counts = self._registry.get_sample_counts(current_name)
        active_names = {c.name for c in self._registry.list_active_classes() if c.id != cls_obj.id}

        dlg = EditClassDialog(
            current_name=current_name,
            current_category=cls_obj.category,
            sample_counts=counts,
            existing_names=active_names,
            parent=self,
        )

        if dlg.exec():
            new_name, new_cat = dlg.get_data()
            try:
                renamed = False
                cat_updated = False
                target_name = current_name

                if new_name != current_name:
                    cls_obj = self._registry.rename_class(current_name, new_name)
                    target_name = new_name
                    renamed = True

                if new_cat != cls_obj.category:
                    cls_obj = self._registry.update_category(target_name, new_cat)
                    cat_updated = True

                self._scan_and_populate_classes()

                # Reselect renamed item
                for idx in range(self._class_list.count()):
                    item = self._class_list.item(idx)
                    if item.data(Qt.UserRole) == target_name:
                        self._class_list.setCurrentRow(idx)
                        break

                msg_parts = []
                if renamed:
                    msg_parts.append(f"Renamed to «{target_name}»")
                if cat_updated:
                    msg_parts.append(f"Category changed to {new_cat}")
                msg = "\n".join(msg_parts) if msg_parts else "No changes made."

                self._latest_card.set_value(f"Edited: {target_name}")
                QMessageBox.information(self, "Class Updated", f"Class «{target_name}» updated successfully.\n\n{msg}")
            except Exception as e:
                self._logger.error(f"Failed to edit class '{current_name}': {e}")
                QMessageBox.critical(self, "Error Editing Class", str(e))

    @Slot()
    def _delete_class(self) -> None:
        """Delete or archive selected class with explicit confirmation workflow."""
        current_name = self._get_selected_class_label()
        if not current_name:
            QMessageBox.information(self, "Select Class", "Please select a class to delete or archive.")
            return

        cls_obj = self._registry.get_class(current_name)
        if not cls_obj:
            QMessageBox.warning(self, "Class Not Found", f"Class «{current_name}» not found in registry.")
            return

        counts = self._registry.get_sample_counts(current_name)
        dlg = DeleteClassDialog(class_name=current_name, sample_counts=counts, parent=self)

        if dlg.exec():
            is_perm = dlg.is_permanent_delete()
            try:
                action, affected_samples = self._registry.delete_class(
                    current_name, permanent_delete_samples=is_perm
                )
                self._scan_and_populate_classes()

                if action == "deleted":
                    self._latest_card.set_value(f"Removed: {current_name}")
                    QMessageBox.information(
                        self,
                        "Class Removed",
                        f"Empty class «{current_name}» removed from active vocabulary.",
                    )
                elif action == "archived":
                    self._latest_card.set_value(f"Archived: {current_name}")
                    QMessageBox.information(
                        self,
                        "Class Archived",
                        f"Class «{current_name}» archived from active vocabulary.\n"
                        f"All {affected_samples} sample files were safely preserved on disk.",
                    )
                elif action == "permanently_deleted":
                    self._latest_card.set_value(f"Deleted: {current_name}")
                    QMessageBox.information(
                        self,
                        "Class Deleted",
                        f"Class «{current_name}» and its {affected_samples} samples were permanently deleted.",
                    )
            except Exception as e:
                self._logger.error(f"Failed to delete class '{current_name}': {e}")
                QMessageBox.critical(self, "Error Deleting Class", str(e))

    @Slot()
    def _inspect_samples(self) -> None:
        """Open SampleInspectorDialog to view and delete individual samples."""
        current_name = self._get_selected_class_label()
        if not current_name:
            QMessageBox.information(self, "Select Class", "Please select a class first.")
            return

        dlg = SampleInspectorDialog(current_name, base_dir=self._registry.dataset_dir, parent=self)
        dlg.exec()
        self._update_stats_and_list()

    def closeEvent(self, event) -> None:
        """Clean shutdown of timers and camera when tab closes."""
        self._stop_camera()
        super().closeEvent(event)
