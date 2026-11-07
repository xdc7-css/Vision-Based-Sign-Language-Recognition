from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
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
from src.domain.hand import HandDetectionResult
from src.domain.multimodal import (
    MULTIMODAL_SCHEMA_VERSION,
    ModalityConfig,
    ModalityMode,
    MultimodalFrame,
)
from src.infrastructure.camera_service import OpenCVCameraService
from src.infrastructure.dataset.collector import DatasetCollector
from src.infrastructure.dataset.gesture_registry import GestureRegistry
from src.infrastructure.dataset.organizer import DatasetOrganizer
from src.infrastructure.hand_detection.detector import HandDetector, HandDetectorConfig
from src.infrastructure.vision.feature_extractor import MultimodalFeatureExtractor
from src.infrastructure.vision.multimodal_detector import MultimodalDetector
from src.infrastructure.vision.multimodal_renderer import MultimodalRenderer
from src.presentation.dialogs.dataset_dialogs import (
    AddClassDialog,
    DeleteClassDialog,
    EditClassDialog,
    SampleInspectorDialog,
)
from src.presentation.widgets import CameraPreview, StatCard


class DatasetCollectionTab(QWidget):
    """Modern, robust 'Teach Signs' interface for capturing and managing Arabic Sign Language datasets.

    Supports:
    - Guided 5-Step Workflow (Select -> Camera -> Record -> Review -> Train)
    - DirectShow Camera Selection (HD Webcam, e2eSoft iVCam, USB Cameras)
    - Multimodal Dual-Hand, Face, and Upper-Body Pose tracking and capture
    - Static Sign Sample Capture (single & multi-sample burst with Space shortcut)
    - Dynamic Temporal Sequence Recording
    - Comprehensive Class & Sample Management (Add, Edit, Delete, Sample Inspector, Stats)
    - Seamless navigation to 'Train Model'
    """

    navigate_to_train = Signal()

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

        # Multimodal Tracking & Feature Services
        self._mm_config = ModalityConfig(
            mode=ModalityMode.TWO_HANDS_FACE,
            enable_left_hand=True,
            enable_right_hand=True,
            enable_face=True,
            enable_pose=False,
            mirror_camera=True,
        )
        self._detector = MultimodalDetector(self._mm_config)
        self._renderer = MultimodalRenderer()
        self._extractor = MultimodalFeatureExtractor(ModalityMode.TWO_HANDS_FACE)

        # Legacy Hand detector for single-hand compatibility
        self._legacy_detector = HandDetector(HandDetectorConfig(
            min_detection_confidence=0.30,
            min_tracking_confidence=0.30,
        ))

        # Capture & streaming state
        self._is_streaming = False
        self._first_frame_rendered = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_detection = None
        self._last_mm_frame: Optional[MultimodalFrame] = None
        self._last_fps_time = time.perf_counter()
        self._fps_counter = 0
        self._session_samples_count = 0

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
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        # ── Guided Workflow Banner ───────────────────────────────────────
        step_banner = QFrame()
        step_banner.setObjectName("stepBanner")
        step_banner.setStyleSheet("""
            QFrame#stepBanner {
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
                padding: 6px 12px;
            }
        """)
        step_layout = QHBoxLayout(step_banner)
        step_layout.setContentsMargins(10, 4, 10, 4)
        step_layout.setSpacing(8)

        steps = [
            ("1", "Select or Add Sign"),
            ("2", "Check Camera & Hands"),
            ("3", "Hold Sign & Record"),
            ("4", "Review Samples"),
            ("5", "Train Model"),
        ]
        for num, text in steps:
            step_box = QHBoxLayout()
            step_box.setSpacing(6)
            num_lbl = QLabel(num)
            num_lbl.setStyleSheet("""
                background-color: #2563EB;
                color: #FFFFFF;
                font-weight: 700;
                font-size: 11px;
                border-radius: 9px;
                min-width: 18px;
                max-width: 18px;
                min-height: 18px;
                max-height: 18px;
                qproperty-alignment: AlignCenter;
            """)
            step_box.addWidget(num_lbl)

            txt_lbl = QLabel(text)
            txt_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #F3F4F6;")
            step_box.addWidget(txt_lbl)
            step_layout.addLayout(step_box)

            if num != "5":
                arrow_lbl = QLabel("→")
                arrow_lbl.setStyleSheet("color: #64748B; font-weight: 700; font-size: 13px;")
                step_layout.addWidget(arrow_lbl)

        step_layout.addStretch()
        layout.addWidget(step_banner)

        # Main Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ═════════════════════════════════════════════════════════════════
        # LEFT PANEL: Camera Viewport & Capture Controls
        # ═════════════════════════════════════════════════════════════════
        left_panel = QFrame()
        left_panel.setObjectName("leftCapturePanel")
        left_panel.setStyleSheet("""
            QFrame#leftCapturePanel {
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        # Camera selection bar
        cam_bar = QHBoxLayout()
        cam_bar.setSpacing(8)

        cam_label = QLabel("Camera:")
        cam_label.setStyleSheet("font-weight: 600; color: #94A3B8;")
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
            background-color: #101419;
            color: #94A3B8;
            border: 1px solid #2B3442;
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 11px;
            font-weight: 600;
        """)
        cam_status_row.addWidget(self._status_badge)

        self._session_counter_lbl = QLabel("Session: 0 samples")
        self._session_counter_lbl.setStyleSheet("color: #14B8A6; font-size: 12px; font-weight: 600; padding-left: 6px;")
        cam_status_row.addWidget(self._session_counter_lbl)

        cam_status_row.addStretch()

        self._fps_indicator = QLabel("FPS: --")
        self._fps_indicator.setStyleSheet("color: #64748B; font-size: 12px; font-weight: 500;")
        cam_status_row.addWidget(self._fps_indicator)

        left_layout.addLayout(cam_status_row)

        # Multimodal Tracking Telemetry Row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(6)
        self._lh_card = StatCard("Left Hand", "Missing")
        stats_row.addWidget(self._lh_card)
        self._rh_card = StatCard("Right Hand", "Missing")
        stats_row.addWidget(self._rh_card)
        self._face_card = StatCard("Face", "Missing")
        stats_row.addWidget(self._face_card)
        self._pose_card = StatCard("Pose", "Disabled")
        stats_row.addWidget(self._pose_card)
        left_layout.addLayout(stats_row)

        # Capture Control Card
        capture_controls = QFrame()
        capture_controls.setObjectName("captureControlsCard")
        capture_controls.setStyleSheet("""
            QFrame#captureControlsCard {
                background-color: #101419;
                border: 1px solid #2B3442;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        capture_layout = QVBoxLayout(capture_controls)
        capture_layout.setSpacing(8)

        # Modality selector and visual toggles row
        modality_bar = QHBoxLayout()
        modality_bar.setSpacing(8)

        mod_lbl = QLabel("Modality:")
        mod_lbl.setStyleSheet("font-weight: 600; color: #94A3B8; font-size: 12px;")
        modality_bar.addWidget(mod_lbl)

        self._modality_combo = QComboBox()
        self._modality_combo.addItem("Two Hands + Face (Recommended)", "two_hands_face")
        self._modality_combo.addItem("Two Hands Only", "two_hands")
        self._modality_combo.addItem("Two Hands + Face + Pose", "two_hands_face_pose")
        self._modality_combo.addItem("One Hand (Legacy v1)", "one_hand")
        self._modality_combo.currentIndexChanged.connect(self._on_modality_changed)
        modality_bar.addWidget(self._modality_combo, stretch=1)

        # Landmark display toggles
        self._chk_show_hands = QCheckBox("Hands")
        self._chk_show_hands.setChecked(True)
        self._chk_show_hands.toggled.connect(self._on_render_toggles_changed)
        modality_bar.addWidget(self._chk_show_hands)

        self._chk_show_face = QCheckBox("Face")
        self._chk_show_face.setChecked(True)
        self._chk_show_face.toggled.connect(self._on_render_toggles_changed)
        modality_bar.addWidget(self._chk_show_face)

        self._chk_show_pose = QCheckBox("Pose")
        self._chk_show_pose.setChecked(False)
        self._chk_show_pose.toggled.connect(self._on_render_toggles_changed)
        modality_bar.addWidget(self._chk_show_pose)

        capture_layout.addLayout(modality_bar)

        # Prominent Capture buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._capture_btn = QPushButton("●  Record Sample (Space)")
        self._capture_btn.setObjectName("actionButton")
        self._capture_btn.setMinimumHeight(40)
        self._capture_btn.setToolTip("Record 1 static sign sample (or press Space key)")
        btn_row.addWidget(self._capture_btn, stretch=2)

        self._burst_btn = QPushButton("⚡ Burst (5)")
        self._burst_btn.setObjectName("successButton")
        self._burst_btn.setMinimumHeight(40)
        self._burst_btn.setToolTip("Rapidly record 5 consecutive samples with 0.4s interval")
        self._burst_btn.clicked.connect(self._on_burst_clicked)
        btn_row.addWidget(self._burst_btn, stretch=1)

        self._batch_capture_btn = QPushButton("📹 Dynamic")
        self._batch_capture_btn.setMinimumHeight(40)
        self._batch_capture_btn.setToolTip("Record continuous motion sequence")
        btn_row.addWidget(self._batch_capture_btn, stretch=1)

        self._cancel_capture_btn = QPushButton("Cancel")
        self._cancel_capture_btn.setObjectName("dangerButton")
        self._cancel_capture_btn.setVisible(False)
        self._cancel_capture_btn.clicked.connect(self._cancel_capture)
        btn_row.addWidget(self._cancel_capture_btn)

        capture_layout.addLayout(btn_row)

        # Sample count selector & Progress
        count_layout = QHBoxLayout()
        count_label = QLabel("Recording Quantity:")
        count_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
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

        # Instant Visual Feedback Alert
        self._save_feedback_lbl = QLabel("")
        self._save_feedback_lbl.setStyleSheet("""
            color: #10B981;
            font-weight: 700;
            font-size: 13px;
            qproperty-alignment: AlignCenter;
            padding: 2px;
        """)
        self._save_feedback_lbl.setVisible(False)
        capture_layout.addWidget(self._save_feedback_lbl)

        left_layout.addWidget(capture_controls)
        splitter.addWidget(left_panel)

        # ═════════════════════════════════════════════════════════════════
        # RIGHT PANEL: Gesture Classes & Dataset Management
        # ═════════════════════════════════════════════════════════════════
        right_panel = QFrame()
        right_panel.setObjectName("rightClassPanel")
        right_panel.setStyleSheet("""
            QFrame#rightClassPanel {
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        # Class list header & action buttons
        class_header = QHBoxLayout()
        self._class_header_label = QLabel("Gesture Signs")
        self._class_header_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #F3F4F6;")
        class_header.addWidget(self._class_header_label)
        class_header.addStretch()

        self._add_class_btn = QPushButton("+ Add Sign")
        self._add_class_btn.setObjectName("primaryButton")
        self._add_class_btn.setToolTip("Add new Arabic sign class")
        class_header.addWidget(self._add_class_btn)

        self._edit_class_btn = QPushButton("✎ Rename")
        self._edit_class_btn.setToolTip("Rename or edit category of selected sign")
        self._edit_class_btn.setEnabled(False)
        class_header.addWidget(self._edit_class_btn)

        self._delete_class_btn = QPushButton("🗑 Delete")
        self._delete_class_btn.setToolTip("Delete or archive selected sign")
        self._delete_class_btn.setEnabled(False)
        class_header.addWidget(self._delete_class_btn)

        right_layout.addLayout(class_header)

        # Search & Filter box
        search_filter_box = QVBoxLayout()
        search_filter_box.setSpacing(6)

        search_row = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("🔍  Search sign gestures (Arabic text)...")
        self._search_edit.setClearButtonEnabled(True)
        search_row.addWidget(self._search_edit, stretch=1)

        self._sort_btn = QPushButton("Sort A-Z")
        self._sort_btn.setCheckable(True)
        self._sort_btn.setChecked(True)
        self._sort_btn.setToolTip("Toggle alphabetical sorting")
        search_row.addWidget(self._sort_btn)
        search_filter_box.addLayout(search_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)

        self._with_samples_first_check = QCheckBox("Signs with samples first")
        self._with_samples_first_check.setChecked(True)
        self._with_samples_first_check.toggled.connect(self._apply_filters_and_populate)
        filter_row.addWidget(self._with_samples_first_check)

        filter_row.addStretch()

        type_lbl = QLabel("Type:")
        type_lbl.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 600;")
        filter_row.addWidget(type_lbl)

        self._type_filter_combo = QComboBox()
        self._type_filter_combo.addItem("All Types", "all")
        self._type_filter_combo.addItem("Static Signs", "static")
        self._type_filter_combo.addItem("Dynamic Sequences", "dynamic")
        self._type_filter_combo.addItem("Both Types", "both")
        self._type_filter_combo.addItem("Alphabet Letters", "alphabet")
        filter_row.addWidget(self._type_filter_combo)

        self._status_filter_combo = QComboBox()
        self._status_filter_combo.addItem("All Classes", "all")
        self._status_filter_combo.addItem("With Samples", "with_samples")
        self._status_filter_combo.addItem("Empty", "empty")
        filter_row.addWidget(self._status_filter_combo)

        search_filter_box.addLayout(filter_row)
        right_layout.addLayout(search_filter_box)

        # Classes List Widget
        self._class_list = QListWidget()
        self._class_list.setStyleSheet("""
            QListWidget {
                background-color: #101419;
                border: 1px solid #2B3442;
                border-radius: 6px;
                padding: 6px;
                color: #F3F4F6;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-radius: 4px;
                margin-bottom: 2px;
            }
            QListWidget::item:selected {
                background-color: #2563EB;
                color: #FFFFFF;
                font-weight: 600;
            }
        """)
        right_layout.addWidget(self._class_list, stretch=1)

        # Class Action Strip
        class_action_bar = QHBoxLayout()
        self._inspect_btn = QPushButton("👁 Review / Inspect Samples")
        self._inspect_btn.setToolTip("View, inspect, and delete individual samples for selected sign")
        self._inspect_btn.setEnabled(False)
        class_action_bar.addWidget(self._inspect_btn)

        self._refresh_stats_btn = QPushButton("🔄 Refresh")
        self._refresh_stats_btn.setToolTip("Rescan dataset from disk and update statistics")
        class_action_bar.addWidget(self._refresh_stats_btn)

        right_layout.addLayout(class_action_bar)

        # Dataset Statistics Cards
        stats_grid = QHBoxLayout()
        self._total_card = StatCard("Total Samples", "0")
        stats_grid.addWidget(self._total_card)
        self._classes_card = StatCard("Signs", "0")
        stats_grid.addWidget(self._classes_card)
        self._split_card = StatCard("Static / Dyn", "0 / 0")
        stats_grid.addWidget(self._split_card)
        self._latest_card = StatCard("Latest Action", "Ready")
        stats_grid.addWidget(self._latest_card)

        right_layout.addLayout(stats_grid)

        # ── Step 5 CTA Card: Train Model Now ─────────────────────────────
        cta_card = QFrame()
        cta_card.setObjectName("ctaCard")
        cta_card.setStyleSheet("""
            QFrame#ctaCard {
                background-color: #101419;
                border: 1px solid #2563EB;
                border-radius: 8px;
                padding: 10px 14px;
            }
        """)
        cta_layout = QHBoxLayout(cta_card)
        cta_layout.setContentsMargins(8, 6, 8, 6)

        self._ready_to_train_lbl = QLabel("Ready to train? Add samples to at least 2 signs.")
        self._ready_to_train_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #F3F4F6;")
        cta_layout.addWidget(self._ready_to_train_lbl, stretch=1)

        self._train_now_btn = QPushButton("🚀 Train Model Now →")
        self._train_now_btn.setObjectName("actionButton")
        self._train_now_btn.setToolTip("Navigate to Train Model tab to train neural network")
        self._train_now_btn.clicked.connect(self._on_train_now_clicked)
        cta_layout.addWidget(self._train_now_btn)

        right_layout.addWidget(cta_card)

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

    # ── Keyboard Shortcuts ───────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        """Allow pressing Space to capture sample when camera is active."""
        if event.key() == Qt.Key_Space:
            if self._is_streaming and not self._multi_capture_active and not self._search_edit.hasFocus():
                self._on_capture_clicked()
                event.accept()
                return
        super().keyPressEvent(event)

    # ── Camera Operations ────────────────────────────────────────────────

    def _refresh_camera_list(self) -> None:
        """Enumerate system cameras via DirectShow COM and populate dropdown."""
        current_sel = self._camera_combo.currentData()
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()

        saved_idx = getattr(self._state, "selected_camera_index", None)
        if current_sel is None and saved_idx is not None:
            current_sel = saved_idx

        active_idx = current_sel if self._is_streaming else None
        cameras = self._camera_service.list_cameras(force_refresh=True, active_index=active_idx)

        if not cameras:
            self._camera_combo.addItem("No Camera Detected", None)
            self._camera_combo.setEnabled(False)
            self._cam_start_btn.setEnabled(False)
            self._status_badge.setText("No Camera Detected")
            self._status_badge.setStyleSheet(
                "background-color: #3b1d1d; color: #ef4444; border: 1px solid #7f1d1d; "
                "border-radius: 4px; padding: 3px 8px;"
            )
        else:
            self._camera_combo.setEnabled(True)
            self._cam_start_btn.setEnabled(True)
            sel_idx = 0
            for i, cam in enumerate(cameras):
                label = cam.name
                if cam.device_type == "laptop":
                    label += " (Laptop Camera)"
                elif "ivcam" in cam.name.lower() or cam.device_type == "virtual":
                    label += " (iPhone / Virtual)"
                elif cam.device_type == "usb":
                    label += " (USB Camera)"

                self._camera_combo.addItem(label, cam.index)
                if current_sel is not None and cam.index == current_sel:
                    sel_idx = i

            self._camera_combo.setCurrentIndex(sel_idx)
            if not self._is_streaming:
                self._status_badge.setText("Camera Ready")
                self._status_badge.setStyleSheet(
                    "background-color: #101419; color: #94A3B8; border: 1px solid #2B3442; "
                    "border-radius: 4px; padding: 3px 8px;"
                )

        self._camera_combo.blockSignals(False)

    @Slot()
    def _start_camera(self) -> None:
        """Start capturing from currently selected camera."""
        idx = self._camera_combo.currentData()
        if idx is None:
            return

        self._first_frame_rendered = False
        self._status_badge.setText("Connecting...")
        self._status_badge.setStyleSheet(
            "background-color: #3b2d14; color: #f59e0b; border: 1px solid #78350f; "
            "border-radius: 4px; padding: 3px 8px; font-weight: 600;"
        )
        self._preview.set_connecting(title="Connecting to Camera...", subtitle=f"DirectShow device Index {idx}")

        self._source_use_cases.select_source(
            VideoSourceType.WEBCAM,
            f"webcam:{idx}",
            device_index=idx,
            resolution=(640, 480),
        )

        ok = self._source_use_cases.start()
        if ok:
            self._is_streaming = True
            self._cam_start_btn.setEnabled(False)
            self._cam_stop_btn.setEnabled(True)
            self._cam_timer.start(30)
            self._latest_card.set_value("Camera Live")
        else:
            self._status_badge.setText("Failed to Open")
            self._status_badge.setStyleSheet(
                "background-color: #3b1d1d; color: #ef4444; border: 1px solid #7f1d1d; "
                "border-radius: 4px; padding: 3px 8px;"
            )
            self._preview.set_error(title="Camera Error", subtitle=f"Device {idx} could not be acquired")

    @Slot()
    def _stop_camera(self) -> None:
        """Stop capturing and release camera cleanly."""
        self._cam_timer.stop()
        if self._multi_capture_active:
            self._cancel_capture()
        if self._dynamic_recording_active:
            self._dynamic_recording_active = False

        self._source_use_cases.stop()
        self._is_streaming = False
        self._first_frame_rendered = False
        self._cam_start_btn.setEnabled(True)
        self._cam_stop_btn.setEnabled(False)
        self._status_badge.setText("Camera Ready")
        self._status_badge.setStyleSheet(
            "background-color: #101419; color: #94A3B8; border: 1px solid #2B3442; "
            "border-radius: 4px; padding: 3px 8px;"
        )
        self._fps_indicator.setText("FPS: --")
        self._preview.set_standby()

        self._lh_card.set_value("Missing")
        self._rh_card.set_value("Missing")
        self._face_card.set_value("Missing")
        self._pose_card.set_value("Disabled")

    def _on_camera_combo_changed(self, index: int) -> None:
        if index < 0:
            return
        idx = self._camera_combo.currentData()
        if idx is not None and hasattr(self._state, "selected_camera_index"):
            self._state.selected_camera_index = idx

        if self._is_streaming:
            self._stop_camera()
            QTimer.singleShot(100, self._start_camera)

    def _on_modality_changed(self, index: int) -> None:
        mode_val = self._modality_combo.currentData()
        try:
            self._mm_config.mode = ModalityMode(mode_val)
        except Exception:
            pass
        enable_pose = (mode_val == "two_hands_face_pose")
        self._chk_show_pose.setChecked(enable_pose)
        self._mm_config.enable_pose = enable_pose
        if hasattr(self._detector, "set_config"):
            self._detector.set_config(self._mm_config)
        else:
            self._detector._config = self._mm_config

    def _on_render_toggles_changed(self) -> None:
        self._renderer.show_hands = self._chk_show_hands.isChecked()
        self._renderer.show_face = self._chk_show_face.isChecked()
        self._renderer.show_pose = self._chk_show_pose.isChecked()

    def _read_camera_frame(self) -> None:
        if not self._is_streaming:
            return

        frame_result = self._source_use_cases.read_frame()
        if frame_result is None:
            return

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
        """Run multimodal detection, render visual skeleton, and update state."""
        self._last_frame = frame

        # FPS
        self._fps_counter += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            current_fps = self._fps_counter / elapsed
            self._fps_indicator.setText(f"FPS: {current_fps:.1f}")
            self._fps_counter = 0
            self._last_fps_time = now

        try:
            mm_frame = self._detector.detect(frame)
            self._last_mm_frame = mm_frame

            hands_list = []
            if mm_frame.left_hand:
                hands_list.append(mm_frame.left_hand)
            if mm_frame.right_hand:
                hands_list.append(mm_frame.right_hand)
            self._last_detection = HandDetectionResult(
                hands=hands_list,
                image_width=frame.shape[1],
                image_height=frame.shape[0],
                processing_time_ms=mm_frame.processing_time_ms,
            )
        except Exception as e:
            self._logger.error(f"Detection error: {e}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._preview.display_frame(frame_rgb)
            return

        annotated_bgr = self._renderer.draw(frame.copy(), mm_frame)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        self._preview.display_frame(annotated_rgb)

        if not self._first_frame_rendered:
            self._first_frame_rendered = True
            self._status_badge.setText("● Live Streaming")
            self._status_badge.setStyleSheet(
                "background-color: #064e3b; color: #10B981; border: 1px solid #047857; "
                "border-radius: 4px; padding: 3px 8px; font-weight: 600;"
            )

        # Update Multimodal Telemetry cards
        self._lh_card.set_value("● Detected" if mm_frame.has_left_hand else "○ Missing")
        self._rh_card.set_value("● Detected" if mm_frame.has_right_hand else "○ Missing")
        self._face_card.set_value("● Detected" if mm_frame.has_face else "○ Missing")
        if not self._chk_show_pose.isChecked():
            self._pose_card.set_value("⊘ Disabled")
        else:
            self._pose_card.set_value("● Detected" if mm_frame.has_pose else "○ Missing")

        if self._dynamic_recording_active:
            mode_val = self._modality_combo.currentData()
            if mode_val == "one_hand":
                if self._last_detection and self._last_detection.has_hands:
                    hand = self._last_detection.hands[0]
                    lm_array = np.array([[p.x, p.y, p.z] for p in hand.landmarks], dtype=np.float32)
                    self._recorded_sequence.append(lm_array)
            else:
                if mm_frame.has_left_hand or mm_frame.has_right_hand:
                    try:
                        mm_mode = ModalityMode(mode_val)
                    except Exception:
                        mm_mode = ModalityMode.TWO_HANDS_FACE
                    vec = self._extractor.extract(mm_frame, mm_mode)
                    self._recorded_sequence.append(vec)

            duration = time.time() - self._sequence_start_time
            self._latest_card.set_value(f"Rec: {len(self._recorded_sequence)} fr ({duration:.1f}s)")

    # ── Static Sign Capture ──────────────────────────────────────────────

    def _get_selected_class_label(self) -> Optional[str]:
        item = self._class_list.currentItem()
        if not item:
            return None
        data = item.data(Qt.UserRole)
        if data:
            return str(data).strip()
        raw_text = item.text().strip()
        clean = re.sub(r'\s*\[[^\]]*\]\s*', '', raw_text).strip()
        clean = re.sub(r'\s*\([^)]*\)\s*', '', clean).strip()
        clean = re.sub(r'⚠️\s*', '', clean).strip()
        return clean or None

    @Slot()
    def _on_burst_clicked(self) -> None:
        """Trigger burst capture of 5 samples."""
        self._samples_spin.setValue(5)
        self._on_capture_clicked()

    @Slot()
    def _on_capture_clicked(self) -> None:
        """Trigger capture of static sign sample(s)."""
        if not self._is_streaming or self._last_frame is None:
            QMessageBox.warning(self, "Camera Required", "Please start the camera before capturing samples.")
            return

        label = self._get_selected_class_label()
        if not label:
            QMessageBox.warning(self, "Select Sign", "Please select a sign from the list first.")
            return

        mode_val = self._modality_combo.currentData()

        if mode_val == "one_hand":
            if not self._last_detection or not self._last_detection.has_hands:
                QMessageBox.warning(
                    self, "No Hand Detected",
                    "No hand was detected in the camera frame.\nPlease hold your hand clearly inside the frame."
                )
                return
        else:
            if not self._last_mm_frame or (not self._last_mm_frame.has_left_hand and not self._last_mm_frame.has_right_hand):
                QMessageBox.warning(
                    self, "No Hand Detected",
                    "At least one hand must be detected in the camera frame.\nPlease hold your hand(s) clearly in view."
                )
                return

            if mode_val in ("two_hands_face", "two_hands_face_pose") and not self._last_mm_frame.has_face:
                QMessageBox.warning(
                    self, "Face Not Detected",
                    "Face detection is required for multimodal sign recording.\nPlease look towards the camera."
                )
                return

            if mode_val == "two_hands_face_pose" and not self._last_mm_frame.has_pose:
                QMessageBox.warning(
                    self, "Pose Not Detected",
                    "Upper-body pose is required for this modality.\nPlease step back slightly."
                )
                return

        count = self._samples_spin.value()
        if count == 1:
            success = self._save_single_static_sample(label)
            if success:
                self._session_samples_count += 1
                self._session_counter_lbl.setText(f"Session: {self._session_samples_count} samples")
                self._latest_card.set_value(f"Saved: {label}")
                self._show_feedback(f"✓ Saved sample for «{label}»")
                self._update_stats_and_list()
            else:
                QMessageBox.warning(self, "Save Error", "Failed to capture sample. Please hold steady and try again.")
        else:
            self._start_multi_capture(label, count)

    def _show_feedback(self, text: str) -> None:
        self._save_feedback_lbl.setText(text)
        self._save_feedback_lbl.setVisible(True)
        QTimer.singleShot(2500, lambda: self._save_feedback_lbl.setVisible(False))

    def _save_single_static_sample(self, label: str) -> bool:
        mode_val = self._modality_combo.currentData()

        if mode_val == "one_hand":
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
                np.save(sample_dir / "landmarks.npy", lm_array[np.newaxis, :, :])
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
                    "schema_version": "v1_single_hand",
                }
                import json
                with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                check = np.load(sample_dir / "landmarks.npy")
                if check.shape != (1, 21, 3) or np.isnan(check).any():
                    import shutil
                    shutil.rmtree(sample_dir, ignore_errors=True)
                    return False

                return True
            except Exception as e:
                self._logger.error(f"Failed to save static sample: {e}")
                import shutil
                shutil.rmtree(sample_dir, ignore_errors=True)
                return False

        else:
            if not self._last_mm_frame:
                return False

            try:
                mm_mode = ModalityMode(mode_val)
            except Exception:
                mm_mode = ModalityMode.TWO_HANDS_FACE

            if not self._last_mm_frame.has_left_hand and not self._last_mm_frame.has_right_hand:
                return False
            if mm_mode in (ModalityMode.TWO_HANDS_FACE, ModalityMode.TWO_HANDS_FACE_POSE) and not self._last_mm_frame.has_face:
                return False
            if mm_mode == ModalityMode.TWO_HANDS_FACE_POSE and not self._last_mm_frame.has_pose:
                return False

            feat_vec = self._extractor.extract(self._last_mm_frame, mm_mode)
            expected_dim = self._extractor.get_feature_dim(mm_mode)
            if feat_vec.shape[0] != expected_dim or np.isnan(feat_vec).any():
                return False

            sanitized = self._organizer.get_class_dir("words", label.replace(" ", "_"))
            sample_id = uuid.uuid4().hex[:8]
            sample_dir = sanitized / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            try:
                np.save(sample_dir / "landmarks.npy", feat_vec[np.newaxis, :])
                meta = {
                    "sample_id": sample_id,
                    "gesture_type": "static_word",
                    "gesture_label": label,
                    "dialect": "iraqi",
                    "timestamp": time.time(),
                    "num_frames": 1,
                    "num_hands": self._last_mm_frame.total_hands_detected,
                    "has_left_hand": self._last_mm_frame.has_left_hand,
                    "has_right_hand": self._last_mm_frame.has_right_hand,
                    "has_face": self._last_mm_frame.has_face,
                    "has_pose": self._last_mm_frame.has_pose,
                    "duration_seconds": 0.0,
                    "fps": 30.0,
                    "has_motion": False,
                    "schema_version": MULTIMODAL_SCHEMA_VERSION,
                    "modality_mode": mm_mode.value,
                    "feature_dim": int(feat_vec.shape[0]),
                }
                import json
                with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                check = np.load(sample_dir / "landmarks.npy")
                if check.shape != (1, expected_dim) or np.isnan(check).any():
                    import shutil
                    shutil.rmtree(sample_dir, ignore_errors=True)
                    return False

                return True
            except Exception as e:
                self._logger.error(f"Failed to save multimodal static sample: {e}")
                import shutil
                shutil.rmtree(sample_dir, ignore_errors=True)
                return False

    def _start_multi_capture(self, label: str, total_count: int) -> None:
        self._multi_capture_active = True
        self._target_sample_count = total_count
        self._captured_sample_count = 0
        self._capture_btn.setEnabled(False)
        self._burst_btn.setEnabled(False)
        self._batch_capture_btn.setEnabled(False)
        self._cancel_capture_btn.setVisible(True)
        self._capture_progress.setVisible(True)
        self._capture_progress.setValue(0)
        self._multi_capture_timer.start(400)

    def _step_multi_capture(self) -> None:
        if not self._multi_capture_active:
            return

        label = self._get_selected_class_label()
        if not label:
            self._cancel_capture()
            return

        success = self._save_single_static_sample(label)
        if success:
            self._captured_sample_count += 1
            self._session_samples_count += 1
            self._session_counter_lbl.setText(f"Session: {self._session_samples_count} samples")
            pct = int((self._captured_sample_count / self._target_sample_count) * 100)
            self._capture_progress.setValue(pct)
            self._latest_card.set_value(f"{label}: {self._captured_sample_count}/{self._target_sample_count}")
            self._show_feedback(f"✓ Recording {self._captured_sample_count}/{self._target_sample_count} for «{label}»")

        if self._captured_sample_count >= self._target_sample_count:
            self._finish_multi_capture()

    def _finish_multi_capture(self) -> None:
        self._multi_capture_timer.stop()
        self._multi_capture_active = False
        self._capture_btn.setEnabled(True)
        self._burst_btn.setEnabled(True)
        self._batch_capture_btn.setEnabled(True)
        self._cancel_capture_btn.setVisible(False)
        self._capture_progress.setVisible(False)
        self._update_stats_and_list()

    def _cancel_capture(self) -> None:
        self._multi_capture_timer.stop()
        self._multi_capture_active = False
        self._capture_btn.setEnabled(True)
        self._burst_btn.setEnabled(True)
        self._batch_capture_btn.setEnabled(True)
        self._cancel_capture_btn.setVisible(False)
        self._capture_progress.setVisible(False)
        self._latest_card.set_value("Capture Cancelled")
        self._update_stats_and_list()

    # ── Dynamic Sequence Recording ───────────────────────────────────────

    @Slot()
    def _toggle_record_sequence(self) -> None:
        if not self._dynamic_recording_active:
            self._start_record_sequence()
        else:
            self._stop_record_sequence()

    def _start_record_sequence(self) -> None:
        if not self._is_streaming or self._last_frame is None:
            QMessageBox.warning(self, "Camera Required", "Please start camera before recording.")
            return

        label = self._get_selected_class_label()
        if not label:
            QMessageBox.warning(self, "Select Sign", "Please select a sign from the list first.")
            return

        self._dynamic_recording_active = True
        self._recorded_sequence.clear()
        self._sequence_start_time = time.time()
        self._batch_capture_btn.setText("⏹  Stop Recording")
        self._batch_capture_btn.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: 700; padding: 8px 14px; border-radius: 4px;")
        self._capture_btn.setEnabled(False)
        self._burst_btn.setEnabled(False)
        self._latest_card.set_value(f"Recording: {label}")

    def _stop_record_sequence(self) -> None:
        self._dynamic_recording_active = False
        self._batch_capture_btn.setText("📹 Dynamic")
        self._batch_capture_btn.setStyleSheet("")
        self._capture_btn.setEnabled(True)
        self._burst_btn.setEnabled(True)

        label = self._get_selected_class_label()
        if not label or len(self._recorded_sequence) < 15:
            QMessageBox.warning(
                self, "Recording Too Short",
                f"Recorded sequence contains only {len(self._recorded_sequence)} frames (minimum 15 required)."
            )
            self._recorded_sequence.clear()
            return

        mode_val = self._modality_combo.currentData()
        sanitized = self._organizer.get_class_dir("dynamic", label.replace(" ", "_"))
        sample_id = uuid.uuid4().hex[:8]
        sample_dir = sanitized / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            seq_arr = np.array(self._recorded_sequence, dtype=np.float32)
            np.save(sample_dir / "landmarks.npy", seq_arr)
            duration = time.time() - self._sequence_start_time
            meta = {
                "sample_id": sample_id,
                "gesture_type": "dynamic_phrase",
                "gesture_label": label,
                "dialect": "iraqi",
                "timestamp": time.time(),
                "num_frames": int(seq_arr.shape[0]),
                "duration_seconds": duration,
                "fps": float(seq_arr.shape[0] / max(duration, 0.1)),
                "has_motion": True,
                "schema_version": MULTIMODAL_SCHEMA_VERSION if mode_val != "one_hand" else "v1_single_hand",
                "modality_mode": mode_val,
            }
            import json
            with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            self._session_samples_count += 1
            self._session_counter_lbl.setText(f"Session: {self._session_samples_count} samples")
            self._latest_card.set_value(f"Saved: {label} ({seq_arr.shape[0]} fr)")
            self._show_feedback(f"✓ Saved dynamic sequence ({seq_arr.shape[0]} frames) for «{label}»")
            self._update_stats_and_list()
        except Exception as e:
            self._logger.error(f"Failed to save dynamic sequence: {e}")
            import shutil
            shutil.rmtree(sample_dir, ignore_errors=True)
            QMessageBox.critical(self, "Save Error", f"Error saving sequence: {e}")

        self._recorded_sequence.clear()

    # ── Classes Management ───────────────────────────────────────────────

    def _scan_and_populate_classes(self) -> None:
        self._apply_filters_and_populate()

    def _update_stats_and_list(self) -> None:
        self._apply_filters_and_populate()

    def _apply_filters_and_populate(self) -> None:
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

        # Optional "signs with samples first" sorting
        if hasattr(self, "_with_samples_first_check") and self._with_samples_first_check.isChecked():
            filtered_classes.sort(key=lambda item: (item[1]["total"] == 0, item[0].name))

        select_row = 0
        row = 0
        total_samples = 0
        total_static = 0
        total_dynamic = 0
        classes_with_samples = 0

        all_active = self._registry.list_active_classes()
        for c in all_active:
            counts = self._registry.get_sample_counts(c.name)
            total_samples += counts["total"]
            total_static += counts["static"]
            total_dynamic += counts["dynamic"]
            if counts["total"] >= 2:
                classes_with_samples += 1

        for cls_obj, counts in filtered_classes:
            static_cnt = counts["static"]
            dynamic_cnt = counts["dynamic"]
            cls_total = counts["total"]

            status_note = " ⚠️ (need ≥2)" if cls_total < 2 else f" ✓ ({cls_total} samples)"
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
            self._class_header_label.setText(f"Gesture Signs ({len(filtered_classes)} of {len(all_active)})")

        # Step 5 CTA banner update
        if hasattr(self, "_ready_to_train_lbl"):
            if classes_with_samples >= 2:
                self._ready_to_train_lbl.setText(f"Ready to Train! {classes_with_samples} signs have ≥2 valid samples.")
                self._ready_to_train_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #10B981;")
                self._train_now_btn.setEnabled(True)
            else:
                self._ready_to_train_lbl.setText(f"Need ≥2 signs with samples to train. ({classes_with_samples} ready)")
                self._ready_to_train_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #F59E0B;")
                self._train_now_btn.setEnabled(True)

    def _on_class_selection_changed(self) -> None:
        has_sel = self._class_list.currentItem() is not None
        self._edit_class_btn.setEnabled(has_sel)
        self._delete_class_btn.setEnabled(has_sel)
        self._inspect_btn.setEnabled(has_sel)

    @Slot()
    def _on_train_now_clicked(self) -> None:
        self.navigate_to_train.emit()

    @Slot()
    def _add_class(self) -> None:
        active_names = {c.name for c in self._registry.list_active_classes()}
        dlg = AddClassDialog(existing_names=active_names, parent=self)
        if dlg.exec():
            name, cat_type, desc = dlg.get_full_data()
            try:
                cls_obj = self._registry.add_class(name, category=cat_type, description=desc)
                self._scan_and_populate_classes()

                for idx in range(self._class_list.count()):
                    item = self._class_list.item(idx)
                    if item.data(Qt.UserRole) == cls_obj.name:
                        self._class_list.setCurrentRow(idx)
                        break

                self._latest_card.set_value(f"Added: {name}")
                self._show_feedback(f"✓ Added sign «{name}»")
            except Exception as e:
                self._logger.error(f"Failed to add class '{name}': {e}")
                QMessageBox.critical(self, "Error Adding Class", str(e))

    @Slot()
    def _edit_class(self) -> None:
        current_name = self._get_selected_class_label()
        if not current_name:
            QMessageBox.information(self, "Select Sign", "Please select a sign to edit.")
            return

        cls_obj = self._registry.get_class(current_name)
        if not cls_obj:
            QMessageBox.warning(self, "Not Found", f"Sign «{current_name}» not found.")
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
                target_name = current_name
                if new_name != current_name:
                    cls_obj = self._registry.rename_class(current_name, new_name)
                    target_name = new_name

                if new_cat != cls_obj.category:
                    cls_obj = self._registry.update_category(target_name, new_cat)

                self._scan_and_populate_classes()

                for idx in range(self._class_list.count()):
                    item = self._class_list.item(idx)
                    if item.data(Qt.UserRole) == target_name:
                        self._class_list.setCurrentRow(idx)
                        break

                self._latest_card.set_value(f"Updated: {target_name}")
                self._show_feedback(f"✓ Updated «{target_name}»")
            except Exception as e:
                self._logger.error(f"Failed to edit class: {e}")
                QMessageBox.critical(self, "Error Editing Class", str(e))

    @Slot()
    def _delete_class(self) -> None:
        current_name = self._get_selected_class_label()
        if not current_name:
            QMessageBox.information(self, "Select Sign", "Please select a sign to delete.")
            return

        cls_obj = self._registry.get_class(current_name)
        if not cls_obj:
            QMessageBox.warning(self, "Not Found", f"Sign «{current_name}» not found.")
            return

        counts = self._registry.get_sample_counts(current_name)
        has_samples = counts["total"] > 0

        dlg = DeleteClassDialog(
            class_name=current_name,
            sample_counts=counts,
            has_samples=has_samples,
            parent=self,
        )

        if dlg.exec():
            action = dlg.get_action()
            try:
                if action == "archive":
                    self._registry.archive_class(current_name)
                    self._latest_card.set_value(f"Archived: {current_name}")
                    self._show_feedback(f"✓ Archived «{current_name}»")
                elif action == "delete":
                    self._registry.delete_class(current_name, delete_sample_files=True)
                    self._latest_card.set_value(f"Deleted: {current_name}")
                    self._show_feedback(f"✓ Deleted «{current_name}»")

                self._scan_and_populate_classes()
            except Exception as e:
                self._logger.error(f"Failed to delete class: {e}")
                QMessageBox.critical(self, "Error Deleting Class", str(e))

    @Slot()
    def _inspect_samples(self) -> None:
        current_name = self._get_selected_class_label()
        if not current_name:
            QMessageBox.information(self, "Select Sign", "Please select a sign to inspect.")
            return

        counts = self._registry.get_sample_counts(current_name)
        if counts["total"] == 0:
            QMessageBox.information(self, "No Samples", f"No saved samples exist for «{current_name}».")
            return

        dlg = SampleInspectorDialog(class_name=current_name, organizer=self._organizer, parent=self)
        dlg.exec()
        self._update_stats_and_list()

    def closeEvent(self, event) -> None:
        self._stop_camera()
        super().closeEvent(event)
