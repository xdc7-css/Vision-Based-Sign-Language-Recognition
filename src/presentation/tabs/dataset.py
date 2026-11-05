from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.presentation.widgets import CameraPreview, SectionHeader, StatCard


class DatasetCollectionTab(QWidget):
    """Tab for capturing and managing training dataset."""

    def __init__(
        self,
        state: AppState,
        source_use_cases: VideoSourceUseCases,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._source_use_cases = source_use_cases
        
        # Initialize collectors, detectors, and renderers
        from src.infrastructure.dataset.collector import DatasetCollector
        from src.infrastructure.hand_detection.detector import HandDetector, HandDetectorConfig
        from src.infrastructure.hand_detection.renderer import HandRenderer
        from PySide6.QtCore import QTimer
        
        self._collector = DatasetCollector()
        self._detector = HandDetector(HandDetectorConfig(
            min_detection_confidence=0.30,
            min_tracking_confidence=0.30,
        ))
        self._renderer = HandRenderer()
        self._last_frame = None
        self._last_detection = None

        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._read_camera_frame)

        self._setup_ui()
        self._connect_signals()
        self._update_stats_and_list()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        header = SectionHeader(
            "Dataset Collection",
            "Capture and manage sign language samples for training",
        )
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Camera preview + capture controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(12)

        self._preview = CameraPreview()
        left_layout.addWidget(self._preview, stretch=1)

        # Hands & Readiness status row
        preview_stats = QHBoxLayout()
        self._hands_card = StatCard("Hands Detected", "0")
        preview_stats.addWidget(self._hands_card)
        self._readiness_card = StatCard("Hand Readiness", "Waiting for Camera")
        preview_stats.addWidget(self._readiness_card)
        left_layout.addLayout(preview_stats)

        capture_controls = QFrame()
        capture_controls.setObjectName("captureControls")
        capture_controls.setStyleSheet("""
            QFrame#captureControls {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        capture_layout = QVBoxLayout(capture_controls)

        capture_label = QLabel("Capture Controls")
        capture_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e0e0e0;")
        capture_layout.addWidget(capture_label)

        # Camera Start/Stop buttons
        cam_btn_row = QHBoxLayout()
        self._cam_start_btn = QPushButton("▶  Start Camera")
        self._cam_start_btn.setObjectName("successButton")
        self._cam_stop_btn = QPushButton("■  Stop Camera")
        self._cam_stop_btn.setObjectName("dangerButton")
        self._cam_stop_btn.setEnabled(False)
        cam_btn_row.addWidget(self._cam_start_btn)
        cam_btn_row.addWidget(self._cam_stop_btn)
        capture_layout.addLayout(cam_btn_row)

        capture_btn_row = QHBoxLayout()
        self._capture_btn = QPushButton("📸  Capture Sample")
        self._capture_btn.setObjectName("successButton")
        capture_btn_row.addWidget(self._capture_btn)

        self._batch_capture_btn = QPushButton("📹  Record Sequence")
        capture_btn_row.addWidget(self._batch_capture_btn)

        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Samples per class:"))

        self._samples_spin = QSpinBox()
        self._samples_spin.setRange(1, 1000)
        self._samples_spin.setValue(50)
        self._samples_spin.setSuffix(" samples")
        count_layout.addWidget(self._samples_spin)
        count_layout.addStretch()

        capture_layout.addLayout(capture_btn_row)
        capture_layout.addLayout(count_layout)
        left_layout.addWidget(capture_controls)

        # Right panel: Class list + stats
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(12)

        # Class list
        class_section = QFrame()
        class_section.setObjectName("classSection")
        class_section.setStyleSheet("""
            QFrame#classSection {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        class_layout = QVBoxLayout(class_section)

        class_header = QHBoxLayout()
        class_label = QLabel("Gesture Classes")
        class_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e0e0e0;")
        class_header.addWidget(class_label)
        class_header.addStretch()

        self._add_class_btn = QPushButton("+ Add Class")
        self._add_class_btn.setFixedWidth(100)
        class_header.addWidget(self._add_class_btn)
        class_layout.addLayout(class_header)

        self._class_list = QListWidget()
        items = ["السلام عليكم", "شكرا", "كيف حالك", "نعم", "لا", "مع السلامة"]
        for item_text in items:
            item = QListWidgetItem(f"  {item_text}  (0 samples)")
            self._class_list.addItem(item)
        class_layout.addWidget(self._class_list)

        right_layout.addWidget(class_section, stretch=1)

        # Stats cards
        stats_row = QHBoxLayout()
        self._total_card = StatCard("Total Samples", "0")
        stats_row.addWidget(self._total_card)
        self._classes_card = StatCard("Classes", str(self._class_list.count()))
        stats_row.addWidget(self._classes_card)
        self._latest_card = StatCard("Latest Capture", "--")
        stats_row.addWidget(self._latest_card)
        right_layout.addLayout(stats_row)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 350])

        layout.addWidget(splitter, stretch=1)

    def _connect_signals(self) -> None:
        self._state.frame_received.connect(self._on_frame_received)
        self._cam_start_btn.clicked.connect(self._start_camera)
        self._cam_stop_btn.clicked.connect(self._stop_camera)
        self._capture_btn.clicked.connect(self._capture_sample)
        self._batch_capture_btn.clicked.connect(self._toggle_record_sequence)
        self._add_class_btn.clicked.connect(self._add_class)
        self._class_list.itemClicked.connect(self._on_class_selected)

    def _start_camera(self) -> None:
        """Start local camera stream for dataset collection."""
        from src.domain.entities import VideoSourceType
        success = self._source_use_cases.start_source(VideoSourceType.WEBCAM, "webcam:0")
        if success:
            self._cam_start_btn.setEnabled(False)
            self._cam_stop_btn.setEnabled(True)
            self._cam_timer.start(33)
            self._readiness_card.set_value("Searching for hand...")
        else:
            self._state.error("Failed to start camera. Check device connection.")

    def _stop_camera(self) -> None:
        """Stop local camera stream."""
        self._cam_timer.stop()
        self._source_use_cases.stop_source()
        self._cam_start_btn.setEnabled(True)
        self._cam_stop_btn.setEnabled(False)
        self._readiness_card.set_value("Camera Stopped")
        self._hands_card.set_value("0")

    def _read_camera_frame(self) -> None:
        """Timer callback to read frame from active source when started locally."""
        frame_result = self._source_use_cases.read_frame()
        if frame_result is not None and frame_result.data is not None:
            self._process_frame_data(frame_result.data)

    @Slot(object)
    def _on_frame_received(self, frame) -> None:
        """Receive frame from external streaming source (e.g. Recognition tab)."""
        if self._cam_timer.isActive():
            # Already reading directly via local timer
            return
        if frame is not None:
            self._process_frame_data(frame)

    def _process_frame_data(self, frame) -> None:
        """Run detection, render skeleton/landmarks, and update session."""
        if frame is None:
            return
        self._last_frame = frame

        try:
            result = self._detector.detect(frame)
            self._last_detection = result
        except Exception as e:
            self._state.error(f"Detection error: {e}")
            self._preview.display_frame(frame)
            return

        # Render skeleton, landmarks, and bounding box
        annotated = self._renderer.draw(frame, result)
        self._preview.display_frame(annotated)

        # Update readiness & hands cards
        if result.has_hands:
            hand = result.hands[0]
            self._hands_card.set_value(f"{len(result.hands)} hand(s)")
            self._readiness_card.set_value(f"Ready ({hand.handedness.value})")

            # Auto-clear old 'No hand detected' error card if it has recovered
            if "No hand" in self._latest_card.text:
                self._latest_card.set_value("Ready")
        else:
            self._hands_card.set_value("0")
            self._readiness_card.set_value("No hand detected")

        # Accumulate frames during active sequence recording
        if self._collector.is_recording and result.has_hands:
            hand = result.hands[0]
            w = hand.landmarks[0]
            m1 = hand.landmarks[5]
            m2 = hand.landmarks[17]
            palm_center = ((w.x + m1.x + m2.x) / 3.0, (w.y + m1.y + m2.y) / 3.0)
            self._collector.current_session.add_frame(result, palm_center)
            frames_count = len(self._collector.current_session.frames)
            self._latest_card.set_value(f"Recording: {frames_count} fr")

    @Slot()
    def _capture_sample(self) -> None:
        """Capture a single static sample (1 frame)."""
        if self._last_frame is None:
            self._state.error("No frame received yet. Make sure the camera is running.")
            return

        selected_item = self._class_list.currentItem()
        if not selected_item:
            self._state.error("Please select a gesture class first.")
            return

        raw_text = selected_item.text().strip()
        clean_label = re.sub(r'\s*\(\d+\s+samples\)\s*', '', raw_text).strip()

        # Find category mapping
        from src.domain.recognition import GestureCategory, DEFAULT_VOCABULARY
        category = GestureCategory.STATIC_WORD
        for entry in DEFAULT_VOCABULARY:
            if entry.arabic_text == clean_label:
                category = entry.category
                break

        # Check hand detection on current frame
        result = self._detector.detect(self._last_frame)
        if not result.has_hands:
            self._state.error("No hand detected. Please hold your hand clearly in front of the camera.")
            self._latest_card.set_value("No hand detected")
            return

        # Active dialect
        dialect_str = "iraqi"
        if hasattr(self._state, "active_dialect") and self._state.active_dialect:
            dialect_str = getattr(self._state.active_dialect.id, "value", "iraqi")

        try:
            self._collector.start_session(gesture_type=category, gesture_label=clean_label, dialect=dialect_str)
            hand = result.hands[0]
            w = hand.landmarks[0]
            m1 = hand.landmarks[5]
            m2 = hand.landmarks[17]
            palm_center = ((w.x + m1.x + m2.x) / 3.0, (w.y + m1.y + m2.y) / 3.0)

            self._collector.current_session.add_frame(result, palm_center)
            meta = self._collector.stop_session()
            if meta:
                path = self._collector.save_sample()
                if path and path.exists() and (path / "landmarks.npy").exists():
                    self._state.log("info", f"Saved sample for '{clean_label}' to {path}")
                    self._update_stats_and_list()
                    self._latest_card.set_value(f"Saved: {clean_label}")
                else:
                    self._state.error("Failed to save sample.")
        except Exception as e:
            self._state.error(f"Failed to capture sample: {e}")

    @Slot()
    def _toggle_record_sequence(self) -> None:
        """Toggle recording of dynamic gesture sequence (multiple frames)."""
        if self._collector.is_recording:
            # Stop and save sequence
            try:
                meta = self._collector.stop_session()
                # Check sufficient valid frames according to dynamic model requirements (>= 5 frames)
                if meta and meta.num_frames >= 5:
                    path = self._collector.save_sample()
                    if path and path.exists() and (path / "landmarks.npy").exists():
                        self._state.log("info", f"Saved sequence for '{meta.gesture_label}' to {path}")
                        self._update_stats_and_list()
                        self._latest_card.set_value(f"Sequence Saved ({meta.num_frames} fr)")
                    else:
                        self._state.error("Failed to save sequence sample.")
                elif meta and meta.num_frames > 0:
                    self._state.error(
                        f"Sequence rejected: Only {meta.num_frames} valid hand frames recorded. "
                        f"Dynamic models require at least 5 frames."
                    )
                    self._latest_card.set_value(f"Rejected: {meta.num_frames} fr (<5)")
                else:
                    self._state.error("No valid hand frames were recorded during sequence.")
                    self._latest_card.set_value("No hand frames")
            except Exception as e:
                self._state.error(f"Error saving sequence: {e}")
            finally:
                self._batch_capture_btn.setText("📹  Record Sequence")
                self._batch_capture_btn.setStyleSheet("")
                self._capture_btn.setEnabled(True)
        else:
            # Start sequence recording
            if self._last_frame is None:
                self._state.error("No frame received yet. Make sure the camera is running.")
                return

            selected_item = self._class_list.currentItem()
            if not selected_item:
                self._state.error("Please select a gesture class first.")
                return

            raw_text = selected_item.text().strip()
            clean_label = re.sub(r'\s*\(\d+\s+samples\)\s*', '', raw_text).strip()

            from src.domain.recognition import GestureCategory, DEFAULT_VOCABULARY
            category = GestureCategory.DYNAMIC_GESTURE
            for entry in DEFAULT_VOCABULARY:
                if entry.arabic_text == clean_label:
                    category = entry.category
                    break

            dialect_str = "iraqi"
            if hasattr(self._state, "active_dialect") and self._state.active_dialect:
                dialect_str = getattr(self._state.active_dialect.id, "value", "iraqi")

            try:
                self._collector.start_session(gesture_type=category, gesture_label=clean_label, dialect=dialect_str)
                self._batch_capture_btn.setText("🛑 Stop Recording")
                self._batch_capture_btn.setStyleSheet("background-color: #f14c4c; color: white;")
                self._capture_btn.setEnabled(False)
                self._latest_card.set_value("Recording...")
            except Exception as e:
                self._state.error(f"Failed to start recording: {e}")
                self._state.error(f"Failed to start recording: {e}")

    @Slot()
    def _add_class(self) -> None:
        """Add a new gesture class to the list."""
        text, ok = QInputDialog.getText(self, "Add Class", "Enter new gesture/word:")
        if ok and text.strip():
            clean_text = text.strip()
            # Check if class already exists in the list widget
            exists = False
            for idx in range(self._class_list.count()):
                item = self._class_list.item(idx)
                item_lbl = re.sub(r'\s*\(\d+\s+samples\)\s*', '', item.text()).strip()
                if item_lbl == clean_text:
                    exists = True
                    break
            
            if not exists:
                item = QListWidgetItem(f"  {clean_text}  (0 samples)")
                self._class_list.addItem(item)
                self._class_list.setCurrentItem(item)
                self._update_stats_and_list()
            else:
                self._state.error(f"Class '{clean_text}' already exists.")

    @Slot(object)
    def _on_class_selected(self, item: QListWidgetItem) -> None:
        pass

    def _update_stats_and_list(self) -> None:
        """Scan the dataset directories, update counts for each list item, and update stats cards."""
        total_samples = 0
        
        # Iterate over all items in the list widget and update their sample counts
        for idx in range(self._class_list.count()):
            item = self._class_list.item(idx)
            raw_text = item.text().strip()
            clean_label = re.sub(r'\s*\(\d+\s+samples\)\s*', '', raw_text).strip()
            
            # Count directories on disk
            count = 0
            sanitized_label = self._collector._sanitize_label(clean_label)
            for category_dir in ["words", "dynamic", "alphabet", "other"]:
                path = self._collector._base_dir / category_dir / sanitized_label / "samples"
                if path.exists() and path.is_dir():
                    count += len([d for d in path.iterdir() if d.is_dir()])
            
            item.setText(f"  {clean_label}  ({count} samples)")
            total_samples += count

        self._total_card.set_value(str(total_samples))
        self._classes_card.set_value(str(self._class_list.count()))
