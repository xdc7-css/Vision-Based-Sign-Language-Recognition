from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.domain.dialect import DialectId
from src.domain.entities import VideoSourceType
from src.infrastructure.analytics import AnalyticsEngine
from src.infrastructure.dialect_manager import DialectManager
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
    FilePathSelector,
    SectionHeader,
    SourceSelector,
    StatCard,
)


class KArSLWorker(QThread):
    """Background worker thread executing multimodal extraction and ST-Transformer inference."""

    loaded = Signal(bool, str)
    result_ready = Signal(object, object, dict, float)  # (KArSLResult, annotated_frame, vis_info, elapsed_ms)
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._recognizer: Optional[KArSLRecognizer] = None
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)
        self._running = True
        self._lock = threading.Lock()

    @property
    def recognizer(self) -> Optional[KArSLRecognizer]:
        return self._recognizer

    def run(self) -> None:
        try:
            self._recognizer = KArSLRecognizer()
            if self._recognizer.is_loaded:
                self.loaded.emit(True, "KArSL-502 engine loaded successfully")
            else:
                self.loaded.emit(False, "Failed to load KArSL-502 engine")
        except Exception as e:
            self.loaded.emit(False, str(e))
            self.error.emit(str(e))

        while self._running:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None or not self._running:
                break

            t0 = time.perf_counter()
            try:
                with self._lock:
                    if self._recognizer and self._recognizer.is_loaded:
                        res, ann_frame, vis_info = self._recognizer.process_frame(frame)
                    else:
                        res, ann_frame, vis_info = None, frame, {}
                elapsed = (time.perf_counter() - t0) * 1000.0
                self.result_ready.emit(res, ann_frame, vis_info, elapsed)
            except Exception as ex:
                self.error.emit(str(ex))

    def submit_frame(self, frame: np.ndarray) -> bool:
        """Submit frame without blocking. Drop old frame if worker is busy."""
        if not self._running or self._recognizer is None or not self._recognizer.is_loaded:
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
        """Clear queued frames and recognizer temporal state."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            if self._recognizer:
                self._recognizer.reset()

    def stop(self) -> None:
        """Gracefully stop worker thread."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self.wait(1500)


class RecognitionTab(QWidget):
    """Live recognition tab supporting KArSL-502 Pretrained Model and Custom Classifier."""

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

        self._source_type = VideoSourceType.WEBCAM
        self._source_uri = "webcam:0"
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_frame)
        self._detection_count = 0

        # Custom Hand Detector & Renderer
        from src.infrastructure.hand_detection.detector import HandDetector, HandDetectorConfig
        from src.infrastructure.hand_detection.renderer import HandRenderer

        det_conf = self._optimizer.profile.min_detection_confidence if self._optimizer else 0.30
        det_track = self._optimizer.profile.min_tracking_confidence if self._optimizer else 0.30
        self._detector = HandDetector(HandDetectorConfig(
            max_num_hands=self._optimizer.profile.max_hands if self._optimizer else 2,
            min_detection_confidence=det_conf,
            min_tracking_confidence=det_track,
        ))
        self._renderer = HandRenderer()

        # Recognition Engine configuration
        self._active_engine = "karsl"
        self._karsl_worker = KArSLWorker(self)
        self._karsl_worker.loaded.connect(self._on_karsl_loaded)
        self._karsl_worker.result_ready.connect(self._on_karsl_result)
        self._karsl_worker.error.connect(self._on_karsl_error)
        self._karsl_worker.start()

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        header = SectionHeader(
            "Real-Time Recognition",
            "Live sign language recognition — KArSL-502 Pretrained Engine & Custom Classifier",
        )
        layout.addWidget(header)

        # Source & Engine controls
        source_bar = QFrame()
        source_bar.setObjectName("sourceBar")
        source_bar.setStyleSheet("""
            QFrame#sourceBar {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        source_layout = QHBoxLayout(source_bar)
        source_layout.setSpacing(14)

        source_layout.addWidget(QLabel("Source:"))
        self._source_selector = SourceSelector()
        source_layout.addWidget(self._source_selector)

        source_layout.addWidget(QLabel("Device:"))
        self._file_selector = FilePathSelector()
        self._file_selector.setVisible(False)
        source_layout.addWidget(self._file_selector)

        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setObjectName("successButton")
        source_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setObjectName("dangerButton")
        self._stop_btn.setEnabled(False)
        source_layout.addWidget(self._stop_btn)

        # Engine selector
        source_layout.addSpacing(12)
        source_layout.addWidget(QLabel("Engine:"))
        self._engine_selector = QComboBox()
        self._engine_selector.addItem("KArSL-502 — Pretrained (502 Signs)", "karsl")
        self._engine_selector.addItem("Custom Sign Classifier", "custom")
        self._engine_selector.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #007acc;
                border-radius: 4px;
                padding: 4px 10px;
                color: #ffffff;
                font-weight: 600;
            }
        """)
        source_layout.addWidget(self._engine_selector)

        self._vocab_btn = QPushButton("📖 502 Signs")
        self._vocab_btn.setToolTip("View full dictionary of 502 supported Arabic signs")
        self._vocab_btn.setStyleSheet("""
            QPushButton {
                background-color: #264f78;
                border: 1px solid #007acc;
                border-radius: 4px;
                padding: 5px 12px;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3168a3;
            }
        """)
        source_layout.addWidget(self._vocab_btn)

        layout.addWidget(source_bar)

        # Main content: preview + results
        content_splitter = QSplitter(Qt.Horizontal)

        # Left: Camera preview
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        self._preview = CameraPreview()
        left_layout.addWidget(self._preview, stretch=1)

        # Stats row
        stats_row = QHBoxLayout()
        self._fps_card = StatCard("FPS", "0")
        stats_row.addWidget(self._fps_card)
        self._latency_card = StatCard("Latency", "-- ms")
        stats_row.addWidget(self._latency_card)
        self._detections_card = StatCard("Detections", "0")
        stats_row.addWidget(self._detections_card)
        self._decision_card = StatCard("Decision Path", "Initializing...")
        stats_row.addWidget(self._decision_card)
        left_layout.addLayout(stats_row)

        content_splitter.addWidget(left_widget)

        # Right: Recognition results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        # Caption display
        caption_box = QFrame()
        caption_box.setObjectName("captionBox")
        caption_box.setStyleSheet("""
            QFrame#captionBox {
                background-color: #1a1a1a;
                border: 2px solid #007acc;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        caption_layout = QVBoxLayout(caption_box)
        caption_layout.addWidget(QLabel("Translation Output"))
        self._caption_label = QLabel("Initializing KArSL-502 engine...")
        self._caption_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #4ec9b0;")
        self._caption_label.setAlignment(Qt.AlignCenter)
        self._caption_label.setMinimumHeight(60)
        caption_layout.addWidget(self._caption_label)

        # Current word
        word_layout = QHBoxLayout()
        word_layout.addWidget(QLabel("Current Word:"))
        self._word_label = QLabel("")
        self._word_label.setStyleSheet("font-size: 20px; color: #e0e0e0;")
        word_layout.addWidget(self._word_label, stretch=1)
        caption_layout.addLayout(word_layout)

        right_layout.addWidget(caption_box)

        # Control buttons (clear, delete, space)
        ctrl_bar = QHBoxLayout()
        self._clear_btn = QPushButton("🗑 Clear")
        self._clear_btn.setObjectName("dangerButton")
        ctrl_bar.addWidget(self._clear_btn)
        self._delete_btn = QPushButton("⌫ Delete")
        ctrl_bar.addWidget(self._delete_btn)
        self._space_btn = QPushButton("␣ Space")
        ctrl_bar.addWidget(self._space_btn)
        right_layout.addLayout(ctrl_bar)

        # Gesture history
        history_label = QLabel("Recognition History")
        history_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #e0e0e0;")
        right_layout.addWidget(history_label)

        self._history_list = QListWidget()
        right_layout.addWidget(self._history_list, stretch=1)

        # Dialect selector
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
        content_splitter.setSizes([550, 350])

        layout.addWidget(content_splitter, stretch=1)

    def _connect_signals(self) -> None:
        self._source_selector.source_changed.connect(self._on_source_type_changed)
        self._file_selector.path_selected.connect(self._on_file_selected)
        self._start_btn.clicked.connect(self._start_recognition)
        self._stop_btn.clicked.connect(self._stop_recognition)
        self._dialect_selector.dialect_selected.connect(self._on_dialect_selected)
        self._engine_selector.currentIndexChanged.connect(self._on_engine_changed)
        self._vocab_btn.clicked.connect(self._show_karsl_vocab)

        self._clear_btn.clicked.connect(self._clear_text)
        self._delete_btn.clicked.connect(self._delete_last)
        self._space_btn.clicked.connect(self._add_space)

        self._state.frame_received.connect(self._preview.display_frame)
        self._state.fps_updated.connect(lambda f: self._fps_card.set_value(f"{f:.1f}"))
        self._state.caption_updated.connect(self._caption_label.setText)
        self._state.gesture_detected.connect(self._on_gesture_detected)
        self._state.pipeline_status_changed.connect(self._on_status_changed)
        if hasattr(self._state, "active_dialect_changed"):
            self._state.active_dialect_changed.connect(self._sync_dialect_selector)

    def _sync_dialect_selector(self, dialect_info: Optional[object]) -> None:
        if dialect_info is None:
            return
        target_id = getattr(dialect_info, "dialect_id", "") or (dialect_info.id.value if hasattr(dialect_info, "id") else "")
        self._dialect_selector.blockSignals(True)
        for idx in range(self._dialect_selector.count()):
            if self._dialect_selector.itemData(idx) == target_id:
                self._dialect_selector.setCurrentIndex(idx)
                break
        self._dialect_selector.blockSignals(False)

    def _on_source_type_changed(self, source_type: VideoSourceType) -> None:
        self._source_type = source_type
        is_file = source_type == VideoSourceType.FILE
        self._file_selector.setVisible(is_file)
        if self._karsl_worker:
            self._karsl_worker.reset_buffer()
        self._stop_recognition()

    def _on_file_selected(self, path: str) -> None:
        self._source_uri = path
        if self._karsl_worker:
            self._karsl_worker.reset_buffer()

    def _on_engine_changed(self, index: int) -> None:
        engine_key = self._engine_selector.currentData()
        self._active_engine = engine_key
        if self._karsl_worker:
            self._karsl_worker.reset_buffer()
        self._vocab_btn.setVisible(engine_key == "karsl")

        if engine_key == "karsl":
            if self._karsl_worker and self._karsl_worker.recognizer and self._karsl_worker.recognizer.is_loaded:
                self._decision_card.set_value("KArSL-502 Ready")
                self._caption_label.setText("[KArSL-502 Ready — 502 Signs Available]")
            else:
                self._decision_card.set_value("Loading...")
                self._caption_label.setText("[Loading KArSL-502 models...]")
            self._state.log("info", "Switched recognition engine to KArSL-502 Pretrained.")
        else:
            self._decision_card.set_value("Custom Ready")
            self._caption_label.setText("...")
            self._state.log("info", "Switched recognition engine to Custom Sign Classifier.")

    def _show_karsl_vocab(self) -> None:
        vocab = []
        if self._karsl_worker and self._karsl_worker.recognizer:
            vocab = self._karsl_worker.recognizer.get_vocabulary()
        dlg = KArSLVocabularyDialog(vocabulary=vocab if vocab else None, parent=self)
        dlg.exec()

    def _on_karsl_loaded(self, success: bool, msg: str) -> None:
        if success:
            self._state.log("info", f"KArSL-502 Engine: {msg}")
            if self._active_engine == "karsl":
                self._decision_card.set_value("KArSL-502 Ready")
                self._caption_label.setText("[KArSL-502 Ready — 502 Signs Available]")
        else:
            self._state.log("error", f"KArSL-502 Engine error: {msg}")
            if self._active_engine == "karsl":
                self._decision_card.set_value("Load Error")
                self._caption_label.setText(f"[KArSL Load Error: {msg}]")

    def _on_karsl_error(self, err: str) -> None:
        self._state.log("warning", f"KArSL Worker error: {err}")

    def _on_karsl_result(
        self,
        res: Optional[KArSLResult],
        annotated_frame: np.ndarray,
        vis_info: dict,
        elapsed_ms: float,
    ) -> None:
        if self._active_engine != "karsl":
            return

        self._state.frame_received.emit(annotated_frame)
        self._latency_card.set_value(f"{elapsed_ms:.0f} ms")

        # Multimodal and Hand presence metrics
        hands_cnt = 0
        if vis_info.get("rh"):
            hands_cnt += 1
        if vis_info.get("lh"):
            hands_cnt += 1

        has_pose = vis_info.get("has_pose", False)
        has_face = vis_info.get("has_face", False)

        if hands_cnt > 0:
            self._detections_card.set_value(f"{hands_cnt} hand(s)")
        elif has_pose:
            self._detections_card.set_value("Pose detected")
        elif has_face:
            self._detections_card.set_value("Face detected")
        else:
            self._detections_card.set_value("0")

        if res is not None and res.pred_idx >= 0:
            conf = res.confidence
            ar = res.arabic_text
            en = res.english_text
            buf = res.buffer_fill

            if res.is_new:
                from src.domain.text import TextCommand
                self._text_builder.process(TextCommand.word(ar))
                self._update_caption()
                self._detection_count += 1

                self._caption_label.setText(f"{ar} ({en})")
                self._word_label.setText(ar)
                self._decision_card.set_value(f"KArSL-502 ({conf:.1%})")

                if self._analytics:
                    self._analytics.record_prediction(correct=True, delay_ms=elapsed_ms)
                    self._analytics.record_system_metrics()

                item = QListWidgetItem(f"  [KArSL-502] {ar} ({en}) ({conf:.1%})")
                self._history_list.insertItem(0, item)
                self._detections_card.set_value(str(self._detection_count))
            else:
                if conf >= 0.20:
                    self._decision_card.set_value(f"eval ({conf:.1%})")
                    self._caption_label.setText(f"[{ar} ({en}) — {conf:.1%}]")
                else:
                    self._decision_card.set_value(f"buffer ({buf}/50)")
                    if not self._word_label.text():
                        self._caption_label.setText(f"[تجميع الإطارات {buf}/50]")
        else:
            buf = res.buffer_fill if res else 0
            if hands_cnt > 0 or has_pose:
                self._decision_card.set_value(f"buffer ({buf}/50)")
                if not self._word_label.text():
                    self._caption_label.setText(f"[تجميع الحركات {buf}/50]")
            else:
                self._decision_card.set_value("no hand")
                if not self._word_label.text():
                    self._caption_label.setText("No hand detected. Please hold your hand in front of the camera.")

    def _on_dialect_selected(self, dialect_id: str) -> None:
        if self._dialect_manager:
            try:
                did = DialectId(dialect_id)
                if self._dialect_manager.switch_dialect(did):
                    self._state.log("info", f"Dialect switched to: {dialect_id}")
                    if self._active_engine != "karsl":
                        self._decision_card.set_value(did.value)
                    for d in getattr(self._state, "dialects", []):
                        if getattr(d, "dialect_id", "") == dialect_id:
                            self._state.active_dialect = d
                            break
            except ValueError:
                pass

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

    def _start_recognition(self) -> None:
        if self._source_type == VideoSourceType.FILE:
            uri = self._file_selector.path
            if not uri:
                self._state.error("Please select a video file first")
                return
        elif self._source_type == VideoSourceType.WEBCAM:
            uri = "webcam:0"
        elif self._source_type == VideoSourceType.SCREEN:
            uri = "screen:0"
        elif self._source_type == VideoSourceType.WINDOW:
            uri = "window:ArSL"
        else:
            uri = "webcam:0"

        if self._karsl_worker:
            self._karsl_worker.reset_buffer()

        success = self._source_use_cases.start_source(self._source_type, uri)
        if success:
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._timer.start(33)  # ~30 FPS

    def _stop_recognition(self) -> None:
        self._timer.stop()
        self._source_use_cases.stop_source()
        if self._karsl_worker:
            self._karsl_worker.reset_buffer()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _process_frame(self) -> None:
        if self._analytics:
            self._analytics.record_frame()

        frame_result = self._source_use_cases.read_frame()

        if frame_result is None or frame_result.data is None:
            return

        frame = frame_result.data
        start_time = time.perf_counter()

        # Frame skipping
        if self._optimizer and not self._optimizer.should_process_frame():
            self._state.frame_received.emit(frame)
            return

        # Resize for performance if requested by profile
        if self._optimizer:
            frame, _ = self._optimizer.resize_frame(frame)

        # Route according to active engine
        if self._active_engine == "karsl" and self._karsl_worker is not None:
            submitted = self._karsl_worker.submit_frame(frame)
            if not submitted:
                self._state.frame_received.emit(frame)
            return

        # ─── Custom Sign Classifier Route ───
        try:
            detection_result = self._detector.detect(frame)
        except Exception as e:
            self._decision_card.set_value("detector error")
            self._state.error(f"Hand detection error: {e}")
            self._state.frame_received.emit(frame)
            return

        # Render landmarks and skeleton on frame
        annotated_frame = self._renderer.draw(frame, detection_result)
        self._state.frame_received.emit(annotated_frame)

        elapsed = (time.perf_counter() - start_time) * 1000.0
        self._latency_card.set_value(f"{elapsed:.0f} ms")

        # Hand processing
        if detection_result.has_hands:
            hand = detection_result.hands[0]
            self._detections_card.set_value(f"{len(detection_result.hands)} hand(s)")

            p = self._pipeline
            has_loaded_models = (
                (p._word_recognizer is not None and getattr(p._word_recognizer, 'is_loaded', False))
                or (p._static_recognizer is not None and getattr(p._static_recognizer, 'is_loaded', False))
                or (p._dynamic_recognizer is not None and getattr(p._dynamic_recognizer, 'is_loaded', False))
            )

            track = None
            hybrid_result, text_cmd = self._pipeline.process_to_command(hand=hand, track=track)

            if hybrid_result.is_valid and text_cmd is not None:
                self._text_builder.process(text_cmd)
                self._update_caption()
                self._detection_count += 1

                conf = hybrid_result.final_result.confidence
                self._decision_card.set_value(f"{hybrid_result.decision_path} ({conf:.1%})")
                self._caption_label.setText(hybrid_result.arabic_text)
                self._word_label.setText(hybrid_result.arabic_text)

                if self._analytics:
                    self._analytics.record_prediction(correct=True, delay_ms=elapsed)
                    self._analytics.record_system_metrics()

                item = QListWidgetItem(
                    f"  [{hybrid_result.decision_path}] "
                    f"{hybrid_result.arabic_text} "
                    f"({conf:.1%})"
                )
                self._history_list.insertItem(0, item)
                self._detections_card.set_value(str(self._detection_count))
            else:
                if not has_loaded_models:
                    self._decision_card.set_value(f"hand ({hand.handedness.value}) - untrained")
                    self._caption_label.setText(f"[يد مكتشفة ({hand.handedness.value}) — لا يوجد نموذج مدرّب]")
                else:
                    conf = hybrid_result.final_result.confidence if hybrid_result.final_result else 0.0
                    pred_ar = hybrid_result.final_result.arabic_text if hybrid_result.final_result else ""
                    self._decision_card.set_value(f"rejected ({conf:.1%})")
                    if pred_ar:
                        self._caption_label.setText(f"[توقع غير مؤكد: {pred_ar} ({conf:.1%})]")
                    else:
                        self._caption_label.setText(f"[إشارة غير مؤكدة: {conf:.1%}]")
        else:
            self._decision_card.set_value("no hand")
            self._detections_card.set_value("0")

    def _on_gesture_detected(self, gesture_id: str, confidence: float) -> None:
        item = QListWidgetItem(f"  [{gesture_id}]  confidence: {confidence:.2%}")
        self._history_list.insertItem(0, item)
        self._detections_card.set_value(str(self._history_list.count()))

    def _on_status_changed(self, status: str) -> None:
        if status == "error":
            self._stop_recognition()

    def closeEvent(self, event) -> None:
        if self._karsl_worker:
            self._karsl_worker.stop()
        super().closeEvent(event)
