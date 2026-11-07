"""Modern, clean Recognition tab exclusively focused on Arabic Sign Language Recognition.

Provides a balanced two-column desktop interface:
- Left Column: DirectShow camera stream, device switching (HD Webcam & e2eSoft iVCam), landmarks toggle.
- Right Column: Recognized Arabic word, confidence badge, sentence builder, TTS speech, and subtitle overlay.
"""

from __future__ import annotations

import json
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
from src.domain.entities import DialectInfo, VideoSourceType
from src.domain.multimodal import ModalityConfig, ModalityMode
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
from src.infrastructure.recognition.multimodal_recognizer import MultimodalSignRecognizer
from src.infrastructure.text_builder import ArabicTextBuilder
from src.infrastructure.vision.multimodal_detector import MultimodalDetector
from src.infrastructure.vision.multimodal_renderer import MultimodalRenderer
from src.presentation.dialogs.karsl_vocab_dialog import KArSLVocabularyDialog
from src.presentation.widgets import CameraPreview, StatCard


class SignRecognitionWorker(QThread):
    """Background worker thread executing real-time MediaPipe tracking and KArSL/Custom/Multimodal sign recognition."""

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

        # Multimodal Components (Dual Hands + Face)
        self._multimodal_detector = MultimodalDetector(
            ModalityConfig(mode=ModalityMode.TWO_HANDS_FACE, enable_pose=False)
        )
        self._multimodal_renderer = MultimodalRenderer()
        self._multimodal_recognizer = MultimodalSignRecognizer()

        # Configurable processing parameters
        self.active_engine = "karsl"  # "karsl", "custom", or "multimodal"
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
                    det_res = self._detector.detect(frame)
                    vis_info["hands_detected"] = len(det_res.hands)
                    if self.show_landmarks and self._renderer and det_res.has_hands:
                        ann_bgr = self._renderer.draw(frame.copy(), det_res)
                        annotated = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Check if custom model is trained and available
                    has_custom_model = (
                        self._pipeline is not None
                        and (
                            (getattr(self._pipeline, "_static_recognizer", None) is not None and self._pipeline._static_recognizer.is_loaded)
                            or (getattr(self._pipeline, "_word_recognizer", None) is not None and getattr(self._pipeline._word_recognizer, "_static_recognizer", None) is not None and self._pipeline._word_recognizer._static_recognizer.is_loaded)
                        )
                    )
                    vis_info["custom_model_loaded"] = has_custom_model

                    if not has_custom_model:
                        vis_info["custom_status"] = "not_trained"
                        karsl_res = KArSLResult(
                            pred_idx=-1,
                            arabic_text="",
                            english_text="",
                            confidence=0.0,
                            hand_detected=det_res.has_hands,
                            face_detected=False,
                            pose_detected=False,
                            buffer_fill=0,
                            is_new=False,
                            decision_path="not_trained",
                        )
                    elif det_res.has_hands and self._pipeline:
                        # Synchronize pipeline thresholds with worker confidence_threshold
                        self._pipeline.config.word_threshold = min(self._pipeline.config.word_threshold, self.confidence_threshold)
                        self._pipeline.config.alphabet_threshold = min(self._pipeline.config.alphabet_threshold, self.confidence_threshold)
                        if hasattr(self._pipeline, "_word_recognizer") and self._pipeline._word_recognizer:
                            self._pipeline._word_recognizer._config.min_confidence_static = min(
                                self._pipeline._word_recognizer._config.min_confidence_static,
                                self.confidence_threshold,
                            )

                        hand = det_res.hands[0]
                        hyp_res = self._pipeline.process(hand)
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
                            # Hand detected, but score below threshold or in cooldown
                            raw_sr = getattr(self._pipeline, "_static_recognizer", None) or getattr(getattr(self._pipeline, "_word_recognizer", None), "_static_recognizer", None)
                            raw_conf = 0.0
                            raw_text = ""
                            if raw_sr and raw_sr.is_loaded:
                                try:
                                    r_pred = raw_sr.predict(hand, 0.0)
                                    raw_conf = r_pred.confidence
                                    raw_text = r_pred.arabic_text
                                except Exception:
                                    pass
                            karsl_res = KArSLResult(
                                pred_idx=-1,
                                arabic_text=raw_text,
                                english_text="",
                                confidence=raw_conf,
                                hand_detected=True,
                                face_detected=False,
                                pose_detected=False,
                                buffer_fill=0,
                                is_new=False,
                                decision_path="detecting",
                            )
                    else:
                        # No hand detected
                        karsl_res = KArSLResult(
                            pred_idx=-1,
                            arabic_text="",
                            english_text="",
                            confidence=0.0,
                            hand_detected=False,
                            face_detected=False,
                            pose_detected=False,
                            buffer_fill=0,
                            is_new=False,
                            decision_path="no_hand",
                        )
                elif self.active_engine == "multimodal":
                    mm_frame = self._multimodal_detector.detect(frame)
                    has_model = self._multimodal_recognizer.is_loaded
                    vis_info["multimodal_loaded"] = has_model
                    vis_info["hands_detected"] = mm_frame.total_hands_detected
                    vis_info["has_left_hand"] = mm_frame.has_left_hand
                    vis_info["has_right_hand"] = mm_frame.has_right_hand
                    vis_info["face_detected"] = mm_frame.has_face
                    vis_info["pose_detected"] = mm_frame.has_pose

                    if self.show_landmarks:
                        ann_bgr = self._multimodal_renderer.draw(frame.copy(), mm_frame)
                        annotated = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if not has_model:
                        vis_info["multimodal_status"] = "not_trained"
                        karsl_res = KArSLResult(
                            pred_idx=-1,
                            arabic_text="",
                            english_text="",
                            confidence=0.0,
                            hand_detected=mm_frame.has_left_hand or mm_frame.has_right_hand,
                            face_detected=mm_frame.has_face,
                            pose_detected=mm_frame.has_pose,
                            buffer_fill=0,
                            is_new=False,
                            decision_path="multimodal_not_trained",
                        )
                    else:
                        if mm_frame.has_left_hand or mm_frame.has_right_hand:
                            rec_res = self._multimodal_recognizer.predict(mm_frame, self.confidence_threshold)
                            if rec_res.is_valid:
                                karsl_res = KArSLResult(
                                    pred_idx=0,
                                    arabic_text=rec_res.arabic_text,
                                    english_text="",
                                    confidence=rec_res.confidence,
                                    hand_detected=True,
                                    face_detected=mm_frame.has_face,
                                    pose_detected=mm_frame.has_pose,
                                    buffer_fill=1,
                                    is_new=True,
                                    decision_path="multimodal_iraqi",
                                )
                            else:
                                karsl_res = KArSLResult(
                                    pred_idx=-1,
                                    arabic_text="",
                                    english_text="",
                                    confidence=0.0,
                                    hand_detected=True,
                                    face_detected=mm_frame.has_face,
                                    pose_detected=mm_frame.has_pose,
                                    buffer_fill=0,
                                    is_new=False,
                                    decision_path="multimodal_low_confidence",
                                )
                        else:
                            karsl_res = KArSLResult(
                                pred_idx=-1,
                                arabic_text="",
                                english_text="",
                                confidence=0.0,
                                hand_detected=False,
                                face_detected=mm_frame.has_face,
                                pose_detected=mm_frame.has_pose,
                                buffer_fill=0,
                                is_new=False,
                                decision_path="multimodal_no_hands",
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
    - Top Banner: Active model info, trained signs count, retraining alerts, direct Teach Signs CTA.
    - Left Column: Live camera view, DirectShow source selection, clean start/stop controls.
    - Right Column: Recognized Arabic word, confidence meter, sentence builder, TTS speech.
    """

    navigate_to_teach = Signal()
    navigate_to_train = Signal()

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
        self._refresh_model_overview_banner()

    # ── UI Construction ──────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 14, 18, 14)
        main_layout.setSpacing(12)

        # ── Top Workflow & Status Banner ─────────────────────────────────
        top_banner = QFrame()
        top_banner.setObjectName("topBanner")
        top_banner.setStyleSheet("""
            QFrame#topBanner {
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
                padding: 10px 14px;
            }
        """)
        banner_layout = QHBoxLayout(top_banner)
        banner_layout.setContentsMargins(12, 6, 12, 6)
        banner_layout.setSpacing(12)

        # App Title & Subtitle
        header_text_box = QVBoxLayout()
        header_text_box.setSpacing(2)
        title_lbl = QLabel("Live Sign Language Recognition")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 700; color: #F3F4F6;")
        header_text_box.addWidget(title_lbl)

        subtitle_lbl = QLabel("Real-time vision translation · Powered by Multimodal tracking")
        subtitle_lbl.setStyleSheet("font-size: 12px; color: #94A3B8;")
        header_text_box.addWidget(subtitle_lbl)
        banner_layout.addLayout(header_text_box)

        banner_layout.addStretch()

        # Outdated notice button (conditionally visible)
        self._retrain_alert_btn = QPushButton("⚡ New samples recorded — Retrain Model Now →")
        self._retrain_alert_btn.setStyleSheet("""
            QPushButton {
                background-color: #451a03;
                color: #F59E0B;
                border: 1px solid #78350f;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #78350f;
                color: #FDE68A;
            }
        """)
        self._retrain_alert_btn.setVisible(False)
        self._retrain_alert_btn.clicked.connect(self._on_retrain_alert_clicked)
        banner_layout.addWidget(self._retrain_alert_btn)

        # Model Status Badge
        self._model_status_badge = QLabel("● Pretrained (502 signs)")
        self._model_status_badge.setStyleSheet(
            "color: #10B981; font-weight: 600; padding: 6px 12px; background: #064e3b; "
            "border: 1px solid #047857; border-radius: 6px; font-size: 12px;"
        )
        banner_layout.addWidget(self._model_status_badge)

        # Vocabulary Button
        self._vocab_btn = QPushButton("502 Signs List")
        self._vocab_btn.setToolTip("View full dictionary of supported Arabic signs")
        banner_layout.addWidget(self._vocab_btn)

        # Direct CTA: Teach Signs
        self._teach_cta_btn = QPushButton("🎓 Teach Signs →")
        self._teach_cta_btn.setObjectName("primaryButton")
        self._teach_cta_btn.setToolTip("Record new Arabic sign gestures and train your custom vocabulary")
        self._teach_cta_btn.clicked.connect(self._on_teach_cta_clicked)
        banner_layout.addWidget(self._teach_cta_btn)

        main_layout.addWidget(top_banner)

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
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
            }
        """)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        # Top Camera Bar: Dropdown + Refresh + Start/Stop
        cam_bar = QHBoxLayout()
        cam_bar.setSpacing(8)

        cam_label = QLabel("Camera:")
        cam_label.setStyleSheet("font-weight: 600; color: #94A3B8;")
        cam_bar.addWidget(cam_label)

        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(260)
        self._camera_combo.setToolTip("Select camera (HD Webcam or e2eSoft iVCam)")
        cam_bar.addWidget(self._camera_combo, stretch=1)
        self._device_combo = self._camera_combo  # alias

        self._refresh_cam_btn = QPushButton("Refresh")
        self._refresh_cam_btn.setToolTip("Scan system for newly connected webcams or iVCam")
        cam_bar.addWidget(self._refresh_cam_btn)

        self._start_stop_btn = QPushButton("Start Camera")
        self._start_stop_btn.setObjectName("primaryButton")
        self._start_stop_btn.setMinimumWidth(120)
        self._start_stop_btn.setFixedHeight(34)
        cam_bar.addWidget(self._start_stop_btn)

        # Compatibility aliases
        self._connect_btn = self._start_stop_btn
        self._disconnect_btn = self._start_stop_btn
        self._start_btn = self._start_stop_btn
        self._stop_btn = self._start_stop_btn

        left_layout.addLayout(cam_bar)

        # Camera Viewport
        self._preview = CameraPreview()
        self._preview.setMinimumSize(480, 360)
        left_layout.addWidget(self._preview, stretch=1)

        # Camera Bottom Strip
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(12)

        self._status_badge = QLabel("Camera Ready")
        self._status_badge.setStyleSheet("""
            background-color: #101419;
            color: #94A3B8;
            border: 1px solid #2B3442;
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
        self._fps_indicator.setStyleSheet("color: #64748B; font-size: 12px; font-weight: 500;")
        bottom_bar.addWidget(self._fps_indicator)

        self._resolution_combo = QComboBox()
        self._resolution_combo.addItem("640x480 (Fast)", (640, 480))
        self._resolution_combo.addItem("1280x720 (HD)", (1280, 720))
        self._resolution_combo.setMinimumWidth(130)
        self._resolution_combo.setMaximumWidth(150)
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
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
            }
        """)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(14, 12, 14, 12)
        right_layout.setSpacing(12)

        # Active Sign Model Engine Selector (Compact bar)
        model_bar = QHBoxLayout()
        model_bar.setSpacing(8)

        engine_lbl = QLabel("Model Engine:")
        engine_lbl.setStyleSheet("font-weight: 600; color: #94A3B8; font-size: 12px;")
        model_bar.addWidget(engine_lbl)

        self._engine_selector = QComboBox()
        self._engine_selector.addItem("KArSL-502 (502 Standard Signs)", "karsl")
        self._engine_selector.addItem("Iraqi Custom Classifier (Legacy)", "custom")
        self._engine_selector.addItem("Multimodal Iraqi — Hands + Face", "multimodal")
        model_bar.addWidget(self._engine_selector, stretch=1)
        right_layout.addLayout(model_bar)

        # Active Sign Detection Display Box
        detection_box = QFrame()
        detection_box.setObjectName("detectionBox")
        detection_box.setStyleSheet("""
            QFrame#detectionBox {
                background-color: #101419;
                border: 1px solid #2B3442;
                border-radius: 8px;
            }
        """)
        det_vlayout = QVBoxLayout(detection_box)
        det_vlayout.setContentsMargins(16, 14, 16, 14)
        det_vlayout.setSpacing(8)

        det_header = QLabel("DETECTED SIGN")
        det_header.setStyleSheet("color: #64748B; font-size: 11px; font-weight: 700; letter-spacing: 1px; background: transparent; border: none;")
        det_vlayout.addWidget(det_header)

        # Large prominent Arabic text display
        self._caption_label = QLabel("جاهز للترجمة")
        self._caption_label.setAlignment(Qt.AlignCenter)
        self._caption_label.setStyleSheet("""
            QLabel {
                font-size: 36px;
                font-weight: 800;
                color: #14B8A6;
                padding: 6px;
                background: transparent;
                border: none;
            }
        """)
        self._caption_label.setMinimumHeight(60)
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
        self._conf_badge.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: 600; min-width: 95px; background: transparent; border: none;")
        conf_row.addWidget(self._conf_badge)

        det_vlayout.addLayout(conf_row)
        right_layout.addWidget(detection_box)

        # Sentence Construction Card
        sentence_group = QGroupBox("Constructed Arabic Sentence")
        sent_layout = QVBoxLayout(sentence_group)
        sent_layout.setContentsMargins(10, 12, 10, 10)
        sent_layout.setSpacing(8)

        self._sentence_edit = QTextEdit()
        self._sentence_edit.setPlaceholderText("Recognized signs will accumulate here as an Arabic sentence...")
        self._sentence_edit.setMaximumHeight(85)
        self._sentence_edit.setLayoutDirection(Qt.RightToLeft)
        font = self._sentence_edit.font()
        font.setPointSize(15)
        self._sentence_edit.setFont(font)
        sent_layout.addWidget(self._sentence_edit)

        # Sentence Editing Bar: Speak, Delete, Clear
        sent_btn_bar = QHBoxLayout()
        sent_btn_bar.setSpacing(6)

        self._speak_btn = QPushButton("🔊 Speak (TTS)")
        self._speak_btn.setObjectName("primaryButton")
        self._speak_btn.setToolTip("Read constructed sentence aloud using speech synthesis")
        sent_btn_bar.addWidget(self._speak_btn)

        self._space_btn = QPushButton("Space")
        self._space_btn.setToolTip("Add space between words")
        sent_btn_bar.addWidget(self._space_btn)

        self._delete_btn = QPushButton("⌫ Delete Word")
        self._delete_btn.setToolTip("Remove last recognized word")
        sent_btn_bar.addWidget(self._delete_btn)

        self._clear_btn = QPushButton("✕ Clear")
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
        self._history_list.setMaximumHeight(130)
        hist_layout.addWidget(self._history_list)
        right_layout.addWidget(history_group, stretch=1)

        # Secondary Actions
        overlay_bar = QHBoxLayout()
        self._toggle_overlay_btn = QPushButton("Toggle Desktop Subtitles")
        self._toggle_overlay_btn.setToolTip("Show/hide transparent desktop subtitle overlay")
        overlay_bar.addWidget(self._toggle_overlay_btn)

        self._screenshot_btn = QPushButton("Screenshot")
        self._screenshot_btn.setToolTip("Save current annotated frame to data/captures/")
        overlay_bar.addWidget(self._screenshot_btn)

        right_layout.addLayout(overlay_bar)
        content_splitter.addWidget(right_card)

        # Set balanced splitter proportions (60% left, 40% right)
        content_splitter.setStretchFactor(0, 60)
        content_splitter.setStretchFactor(1, 40)
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

    # ── Workflow & Navigation ────────────────────────────────────────────

    def _on_teach_cta_clicked(self) -> None:
        self.navigate_to_teach.emit()

    def _on_retrain_alert_clicked(self) -> None:
        self.navigate_to_train.emit()

    def _refresh_model_overview_banner(self) -> None:
        """Check whether models are trained and if new samples require retraining."""
        # Check retraining needed
        model_file = Path("data/models/multimodal/static_model.pkl")
        if not model_file.exists():
            model_file = Path("data/models/iraqi/static_model.pkl")

        if model_file.exists():
            m_time = model_file.stat().st_mtime
            new_samples = 0
            words_dir = Path("data/datasets/words")
            if words_dir.exists():
                for s_dir in words_dir.glob("*/samples/*"):
                    try:
                        if s_dir.is_dir() and s_dir.stat().st_mtime > m_time:
                            new_samples += 1
                    except Exception:
                        pass
            if new_samples > 0:
                self._retrain_alert_btn.setText(f"⚡ {new_samples} new samples recorded — Retrain Model Now →")
                self._retrain_alert_btn.setVisible(True)
            else:
                self._retrain_alert_btn.setVisible(False)
        else:
            self._retrain_alert_btn.setVisible(False)

    # ── Camera Operations ────────────────────────────────────────────────

    def _refresh_camera_list(self) -> None:
        """Enumerate system cameras via DirectShow COM and populate dropdown."""
        current_sel = self._camera_combo.currentData()
        if current_sel is None and hasattr(self._state, "selected_camera_index"):
            current_sel = self._state.selected_camera_index
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
                self._status_badge.setText("Camera Ready")
                self._status_badge.setStyleSheet("background-color: #101419; color: #94A3B8; border: 1px solid #2B3442; border-radius: 4px; padding: 4px 10px;")

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
        self._status_badge.setStyleSheet("background-color: #101419; color: #94A3B8; border: 1px solid #2B3442; border-radius: 4px; padding: 4px 10px;")
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
        idx = self._camera_combo.currentData()
        if idx is not None and hasattr(self._state, "selected_camera_index"):
            self._state.selected_camera_index = idx

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
                    "background-color: #064e3b; color: #10B981; border: 1px solid #047857; "
                    "border-radius: 4px; padding: 4px 10px; font-weight: 600;"
                )
                self._decision_card.set_value("Active")

        # Update FPS telemetry
        fps = 1000.0 / max(elapsed_ms, 1.0)
        self._fps_indicator.setText(f"{fps:.1f} FPS")
        self._fps_card.set_value(f"{fps:.1f}")
        self._latency_card.set_value(f"{elapsed_ms:.0f} ms")

        # Update Sign Language Recognition results only from live valid frames
        if (
            karsl_res
            and karsl_res.hand_detected
            and karsl_res.confidence >= self._worker.confidence_threshold
            and karsl_res.arabic_text
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
        elif self._worker.active_engine == "custom" and not vis_info.get("custom_model_loaded", True):
            self._conf_bar.setValue(0)
            self._conf_badge.setText("⚠️ Custom Model Not Trained")
            self._caption_label.setText("النموذج المخصص غير مدرب")
        elif self._worker.active_engine == "multimodal" and not vis_info.get("multimodal_loaded", True):
            self._conf_bar.setValue(0)
            self._conf_badge.setText("⚠️ Multimodal Model Not Trained")
            self._caption_label.setText("النموذج المتعدد غير مدرب")
        elif karsl_res and karsl_res.hand_detected:
            conf_pct = int(getattr(karsl_res, "confidence", 0.0) * 100)
            self._conf_bar.setValue(conf_pct)
            raw_text = getattr(karsl_res, "arabic_text", "").strip()
            if raw_text and conf_pct > 10:
                self._conf_badge.setText(f"Detecting: {raw_text} ({conf_pct}%)")
            else:
                self._conf_badge.setText("Hand Detected · Awaiting Sign...")
        else:
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

    def _update_custom_model_status_ui(self) -> None:
        """Update model status badge, caption, and vocabulary list button based on custom model state."""
        if self._engine_selector.currentData() != "custom":
            return
        classes = []
        classes_path = Path("data/models/iraqi/classes.json")
        model_path = Path("data/models/iraqi/static_model.pkl")
        is_trained = model_path.exists()

        if classes_path.exists():
            try:
                with open(classes_path, "r", encoding="utf-8") as f:
                    classes = json.load(f)
            except Exception:
                pass

        if is_trained and classes:
            self._model_status_badge.setText(f"● Trained ({len(classes)} signs)")
            self._model_status_badge.setStyleSheet(
                "color: #10B981; font-weight: 600; padding: 4px 10px; background: #064e3b; border-radius: 4px; font-size: 11px;"
            )
            self._vocab_btn.setText(f"Custom Signs ({len(classes)})")
            self._caption_label.setText("جاهز للترجمة (مخصص)")
            self._conf_badge.setText("Waiting for sign...")
            self._state.active_dialect_changed.emit(
                DialectInfo("iraqi", f"Iraqi Custom ({len(classes)} signs)", "عراقية")
            )
        else:
            self._model_status_badge.setText("⚠️ Not Trained")
            self._model_status_badge.setStyleSheet(
                "color: #F59E0B; font-weight: 600; padding: 4px 10px; background: #451a03; border-radius: 4px; font-size: 11px;"
            )
            self._vocab_btn.setText("Custom Signs (0)")
            self._caption_label.setText("النموذج المخصص غير مدرب")
            self._conf_badge.setText("Custom Model Not Trained")
            self._state.active_dialect_changed.emit(
                DialectInfo("iraqi", "Iraqi Custom (Not Trained)", "عراقية")
            )

    def _update_multimodal_model_status_ui(self) -> None:
        """Update model status badge, caption, and vocabulary list button based on multimodal model state."""
        if self._engine_selector.currentData() != "multimodal":
            return
        self._worker._multimodal_recognizer.load_model()
        is_trained = self._worker._multimodal_recognizer.is_loaded
        classes = self._worker._multimodal_recognizer.classes

        if is_trained and classes:
            self._model_status_badge.setText(f"● Multimodal ({len(classes)} signs)")
            self._model_status_badge.setStyleSheet(
                "color: #38BDF8; font-weight: 600; padding: 4px 10px; background: #0c4a6e; border-radius: 4px; font-size: 11px;"
            )
            self._vocab_btn.setText(f"Multimodal ({len(classes)})")
            self._caption_label.setText("جاهز للترجمة (متعدد الوسائط)")
            self._conf_badge.setText("Waiting for sign...")
            self._state.active_dialect_changed.emit(
                DialectInfo("multimodal", f"Multimodal Iraqi ({len(classes)} signs)", "عراقية")
            )
        else:
            self._model_status_badge.setText("⚠️ Multimodal Not Trained")
            self._model_status_badge.setStyleSheet(
                "color: #F59E0B; font-weight: 600; padding: 4px 10px; background: #451a03; border-radius: 4px; font-size: 11px;"
            )
            self._vocab_btn.setText("Multimodal (0)")
            self._caption_label.setText("النموذج المتعدد غير مدرب")
            self._conf_badge.setText("Multimodal Model Not Trained")
            self._state.active_dialect_changed.emit(
                DialectInfo("multimodal", "Multimodal Iraqi (Not Trained)", "عراقية")
            )

    def _on_engine_changed(self, index: int) -> None:
        engine = self._engine_selector.currentData()
        self._worker.active_engine = engine
        if engine == "custom":
            if self._dialect_manager:
                self._dialect_manager.reload_dialect()
            self._update_custom_model_status_ui()
        elif engine == "multimodal":
            self._update_multimodal_model_status_ui()
        else:
            self._model_status_badge.setText("● Pretrained (502 signs)")
            self._model_status_badge.setStyleSheet(
                "color: #10B981; font-weight: 600; padding: 4px 10px; background: #064e3b; border-radius: 4px; font-size: 11px;"
            )
            self._vocab_btn.setText("502 Signs List")
            self._caption_label.setText("جاهز للترجمة")
            self._conf_badge.setText("Waiting for sign...")
            self._state.active_dialect_changed.emit(
                DialectInfo("karsl", "KArSL-502 / Iraqi", "عراقية")
            )

    def _open_vocab_dialog(self) -> None:
        if self._worker.active_engine == "multimodal":
            classes = []
            classes_path = Path("data/models/multimodal/classes.json")
            if classes_path.exists():
                try:
                    with open(classes_path, "r", encoding="utf-8") as f:
                        classes = json.load(f)
                except Exception:
                    pass

            if not classes:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Multimodal Signs List",
                    "No multimodal signs trained yet.\n\n"
                    "Please record multimodal samples (Hands + Face) in Teach Signs and train the model in Train Model.",
                )
                return

            vocab = []
            for i, c in enumerate(classes):
                s_count = 0
                s_dir = Path("data/datasets/words") / c / "samples"
                if s_dir.exists():
                    s_count = len(list(s_dir.iterdir()))
                vocab.append((i + 1, c.replace("_", " "), f"Multimodal Sign ({s_count} samples)"))

            dialog = KArSLVocabularyDialog(vocab, parent=self)
            dialog.setWindowTitle("Multimodal Sign Vocabulary")
            dialog.exec()
            return

        elif self._worker.active_engine == "custom":
            classes = []
            classes_path = Path("data/models/iraqi/classes.json")
            if classes_path.exists():
                try:
                    with open(classes_path, "r", encoding="utf-8") as f:
                        classes = json.load(f)
                except Exception:
                    pass
            if not classes:
                words_dir = Path("data/datasets/words")
                if words_dir.exists():
                    classes = [
                        d.name
                        for d in words_dir.iterdir()
                        if d.is_dir() and (d / "samples").exists() and list((d / "samples").iterdir())
                    ]

            if not classes:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Custom Signs List",
                    "No custom signs trained yet.\n\nPlease collect sign samples in Teach Signs and train the model in Train Model tab.",
                )
                return

            vocab = []
            for i, c in enumerate(classes):
                s_count = 0
                s_dir = Path("data/datasets/words") / c / "samples"
                if s_dir.exists():
                    s_count = len(list(s_dir.iterdir()))
                vocab.append({
                    "index": i + 1,
                    "arabic": c.replace("_", " "),
                    "english": f"{s_count} samples",
                    "category": "Custom Sign (Static)",
                })
            dlg = KArSLVocabularyDialog(vocabulary=vocab, parent=self)
            dlg.setWindowTitle(f"Iraqi Custom Sign Language Dictionary ({len(classes)} Signs)")
            dlg.exec()
        else:
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
