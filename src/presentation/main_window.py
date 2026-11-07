"""Primary application window with a clean, modern, student-friendly desktop layout.

Provides 4 primary navigation pages:
1. Recognition (Default Home Viewport)
2. Dataset Studio (Gesture Recording & Management)
3. Model Training (Training & Performance Evaluation)
4. Settings (Camera Hardware, Models, Subtitle Overlay, Audio)
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
    QSystemTrayIcon,
    QTabWidget,
    QWidget,
)

from src.application.state import AppState
from src.application.use_cases import (
    CameraUseCases,
    DialectUseCases,
    ModelUseCases,
    VideoSourceUseCases,
)
from src.infrastructure.analytics import AnalyticsEngine
from src.infrastructure.dialect_manager import DialectManager
from src.infrastructure.optimization import PerformanceOptimizer
from src.infrastructure.overlay_system import OverlayManager
from src.infrastructure.recognition.hybrid_pipeline import HybridRecognitionPipeline
from src.infrastructure.text_builder import ArabicTextBuilder
from src.presentation.tabs.analytics import AnalyticsTab
from src.presentation.tabs.dataset import DatasetCollectionTab
from src.presentation.tabs.overlay import OverlayTab
from src.presentation.tabs.recognition import RecognitionTab
from src.presentation.tabs.settings import SettingsTab
from src.presentation.tabs.training import ModelTrainingTab


class MainWindow(QMainWindow):
    """Primary application window with clean 4-page navigation."""

    WINDOW_TITLE = "Vision-Based Sign Language Recognition"
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 800

    def __init__(
        self,
        state: AppState,
        camera_use_cases: CameraUseCases,
        dialect_use_cases: DialectUseCases,
        model_use_cases: ModelUseCases,
        source_use_cases: VideoSourceUseCases,
        pipeline: Optional[HybridRecognitionPipeline] = None,
        text_builder: Optional[ArabicTextBuilder] = None,
        dialect_manager: Optional[DialectManager] = None,
        overlay_manager: Optional[OverlayManager] = None,
        analytics: Optional[AnalyticsEngine] = None,
        optimizer: Optional[PerformanceOptimizer] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._camera_use_cases = camera_use_cases
        self._dialect_use_cases = dialect_use_cases
        self._model_use_cases = model_use_cases
        self._source_use_cases = source_use_cases

        # Injected services
        self._pipeline = pipeline
        self._text_builder = text_builder
        self._dialect_manager = dialect_manager
        self._overlay_manager = overlay_manager
        self._analytics = analytics
        self._optimizer = optimizer
        self._logger = logging.getLogger(__name__)

        self._setup_window()
        self._setup_menu_bar()
        self._setup_tabs()
        self._setup_status_bar()
        self._setup_system_tray()
        self._connect_signals()

        # Start directly on Recognition as Home page
        self._tab_widget.setCurrentIndex(0)

    # ── Setup ──────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setMinimumSize(1024, 680)
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)

        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.DEFAULT_WIDTH) // 2
        y = (screen.height() - self.DEFAULT_HEIGHT) // 2
        self.move(max(0, x), max(0, y))

    def _setup_menu_bar(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        self._select_source_action = QAction("Live Recognition View", self)
        self._select_source_action.setShortcut("Ctrl+1")
        self._select_source_action.triggered.connect(lambda: self._tab_widget.setCurrentIndex(0))
        file_menu.addAction(self._select_source_action)

        self._settings_action = QAction("Preferences...", self)
        self._settings_action.setShortcut("Ctrl+,")
        self._settings_action.triggered.connect(lambda: self._tab_widget.setCurrentIndex(3))
        file_menu.addAction(self._settings_action)

        file_menu.addSeparator()
        self._quit_action = QAction("Exit", self)
        self._quit_action.setShortcut("Ctrl+Q")
        self._quit_action.triggered.connect(self.close)
        file_menu.addAction(self._quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        self._toggle_overlay_action = QAction("Toggle Desktop Subtitles", self)
        self._toggle_overlay_action.setShortcut("Ctrl+T")
        view_menu.addAction(self._toggle_overlay_action)
        view_menu.addSeparator()

        self._fullscreen_action = QAction("Toggle Fullscreen", self)
        self._fullscreen_action.setShortcut("F11")
        self._fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(self._fullscreen_action)

        # Dialect / Model menu
        dialect_menu = menubar.addMenu("&Model")
        self._dialect_actions: dict[str, QAction] = {}
        from src.domain.dialect import DIALECT_DISPLAY_NAMES
        for did, display in DIALECT_DISPLAY_NAMES.items():
            action = QAction(display, self, checkable=True)
            action.triggered.connect(lambda checked, d=did: self._switch_dialect(d))
            dialect_menu.addAction(action)
            self._dialect_actions[did.value] = action

        # Help menu
        help_menu = menubar.addMenu("&Help")
        self._about_action = QAction("About Sign Language Recognition", self)
        self._about_action.triggered.connect(self._show_about)
        help_menu.addAction(self._about_action)

    def _setup_tabs(self) -> None:
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabPosition(QTabWidget.North)
        self._tab_widget.setDocumentMode(True)
        self._tab_widget.setMovable(False)

        # 4 Primary Application Pages
        self._recognition_tab = RecognitionTab(
            self._state,
            self._source_use_cases,
            pipeline=self._pipeline,
            text_builder=self._text_builder,
            dialect_manager=self._dialect_manager,
            analytics=self._analytics,
            optimizer=self._optimizer,
        )
        self._dataset_tab = DatasetCollectionTab(self._state, self._source_use_cases)
        self._training_tab = ModelTrainingTab(
            self._state,
            dialect_manager=self._dialect_manager,
            pipeline=self._pipeline,
        )
        self._settings_tab = SettingsTab(self._state)

        # Preserved secondary modules (accessible via settings or tests)
        self._overlay_tab = OverlayTab(self._state, overlay_manager=self._overlay_manager)
        self._analytics_tab = AnalyticsTab(engine=self._analytics)

        # Add 4 Primary Tabs with clean labels
        self._tab_widget.addTab(self._recognition_tab, "Recognition")
        self._tab_widget.addTab(self._dataset_tab, "Dataset Studio")
        self._tab_widget.addTab(self._training_tab, "Model Training")
        self._tab_widget.addTab(self._settings_tab, "Settings")

        self.setCentralWidget(self._tab_widget)

    def _setup_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("System Ready")
        self._status_bar.addWidget(self._status_label, stretch=1)

        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("padding: 0 10px; color: #94a3b8;")
        self._status_bar.addPermanentWidget(self._fps_label)

        self._latency_label = QLabel("Latency: -- ms")
        self._latency_label.setStyleSheet("padding: 0 10px; color: #94a3b8;")
        self._status_bar.addPermanentWidget(self._latency_label)

        self._source_label = QLabel("Camera: Standby")
        self._source_label.setStyleSheet("padding: 0 10px; color: #94a3b8;")
        self._status_bar.addPermanentWidget(self._source_label)

        self._dialect_label = QLabel("Model: KArSL-502 / Iraqi")
        self._dialect_label.setStyleSheet("padding: 0 10px; color: #14b8a6; font-weight: 500;")
        self._status_bar.addPermanentWidget(self._dialect_label)

    def _setup_system_tray(self) -> None:
        self._tray_icon: Optional[QSystemTrayIcon] = None

        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray_icon = QSystemTrayIcon(self)
            self._tray_icon.setToolTip("Arabic Sign Language Recognition")

            tray_menu = QMenu()
            show_action = QAction("Show Application", self)
            show_action.triggered.connect(self.show_and_raise)
            tray_menu.addAction(show_action)

            toggle_action = QAction("Toggle Desktop Subtitles", self)
            toggle_action.triggered.connect(self._toggle_overlay_from_tray)
            tray_menu.addAction(toggle_action)

            tray_menu.addSeparator()
            quit_action = QAction("Exit", self)
            quit_action.triggered.connect(QApplication.quit)
            tray_menu.addAction(quit_action)

            self._tray_icon.setContextMenu(tray_menu)

            try:
                pixmap = QPixmap(32, 32)
                pixmap.fill(Qt.darkBlue)
                self._tray_icon.setIcon(QIcon(pixmap))
            except Exception:
                pass

            self._tray_icon.activated.connect(self._on_tray_activated)
            self._tray_icon.show()

    # ── Signals ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._state.pipeline_status_changed.connect(self._on_status_changed)
        self._state.fps_updated.connect(lambda f: self._fps_label.setText(f"FPS: {f:.1f}"))
        self._state.active_source_changed.connect(
            lambda s: self._source_label.setText(f"Camera: {s.display_name if s else 'Standby'}")
        )
        self._state.active_dialect_changed.connect(
            lambda d: self._dialect_label.setText(f"Model: {d.display_name if d else 'KArSL-502'}")
        )
        self._state.error_occurred.connect(self._on_error)

        self._toggle_overlay_action.triggered.connect(self._toggle_overlay_action_triggered)
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

    # ── Slots ──────────────────────────────────────────────────────

    def show_and_raise(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason: int) -> None:
        if reason == QSystemTrayIcon.DoubleClick:
            self.show_and_raise()

    def _on_status_changed(self, status: str) -> None:
        self._status_label.setText(status.title())
        if status.lower() in ("streaming", "active"):
            self._status_label.setStyleSheet("color: #10b981; font-weight: 500;")
        elif status.lower() == "error":
            self._status_label.setStyleSheet("color: #ef4444; font-weight: 500;")
        else:
            self._status_label.setStyleSheet("color: #94a3b8;")

    def _on_error(self, message: str) -> None:
        self._status_label.setText(f"Notice: {message}")
        self._status_label.setStyleSheet("color: #ef4444;")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_overlay_action_triggered(self) -> None:
        self._state.overlay_visible = not self._state.overlay_visible

    def _toggle_overlay_from_tray(self) -> None:
        self._state.overlay_visible = not self._state.overlay_visible

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Sign Language Recognition",
            "<h2>Vision-Based Sign Language Recognition</h2>"
            "<p><b>Version:</b> 2.0.0 (Clean Architecture)</p>"
            "<p>Real-time Arabic Sign Language recognition desktop system using MediaPipe keypoint tracking "
            "and KArSL-502 Spatio-Temporal Transformer inference.</p>"
            "<hr>"
            "<p><b>Features:</b> Native Windows DirectShow multi-camera support (HD Webcam & e2eSoft iVCam), "
            "real-time sentence accumulation, Text-to-Speech audio output, transparent desktop subtitle overlay, "
            "and dataset collection studio.</p>"
            "<p>Computer Engineering Graduation Project</p>",
        )

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab switching and safely coordinate camera hardware ownership."""
        if index == 0:  # Recognition tab
            if hasattr(self, "_dataset_tab") and getattr(self._dataset_tab, "_is_streaming", False):
                self._dataset_tab._stop_camera()
            if hasattr(self, "_recognition_tab"):
                self._recognition_tab._refresh_camera_list()
                if hasattr(self._recognition_tab, "_update_custom_model_status_ui"):
                    self._recognition_tab._update_custom_model_status_ui()
            self._refresh_dialect_menu()
        elif index == 1:  # Dataset Studio
            if hasattr(self, "_recognition_tab") and getattr(self._recognition_tab, "_is_streaming", False):
                self._recognition_tab._stop_camera()
            if hasattr(self, "_dataset_tab"):
                self._dataset_tab._refresh_camera_list()
                self._dataset_tab._scan_and_populate_classes()
        elif index == 2:  # Model Training
            if hasattr(self, "_dataset_tab") and getattr(self._dataset_tab, "_is_streaming", False):
                self._dataset_tab._stop_camera()
            if hasattr(self, "_training_tab"):
                self._training_tab._sync_active_dialect()
                if hasattr(self._training_tab, "refresh_model_sync_status"):
                    self._training_tab.refresh_model_sync_status()

    def _refresh_dialect_menu(self) -> None:
        """Update dialect menu check states based on active dialect."""
        if self._dialect_manager is None:
            return
        active = self._dialect_manager.active_dialect.value
        for did, action in self._dialect_actions.items():
            action.setChecked(did == active)

    def _switch_dialect(self, dialect_id) -> None:
        """Switch dialect from menu."""
        from src.domain.dialect import DialectId
        if isinstance(dialect_id, str):
            try:
                dialect_id = DialectId(dialect_id)
            except ValueError:
                return

        did_str = dialect_id.value if hasattr(dialect_id, "value") else str(dialect_id)
        if self._dialect_manager and self._dialect_manager.switch_dialect(dialect_id):
            cfg = self._dialect_manager.active_dialect_config
            self._dialect_label.setText(f"Model: {cfg.display_name if cfg else did_str.title()}")
            self._refresh_dialect_menu()
            for d in getattr(self._state, "dialects", []):
                if getattr(d, "dialect_id", "") == did_str:
                    self._state.active_dialect = d
                    break
            self._state.log("info", f"Switched to dialect: {did_str}")

    def closeEvent(self, event) -> None:
        """Handle application shutdown — clean up all resources."""
        if hasattr(self, "_recognition_tab") and self._recognition_tab:
            try:
                self._recognition_tab.close()
            except Exception:
                pass

        if self._source_use_cases:
            self._source_use_cases.stop_source()

        if self._overlay_manager:
            self._overlay_manager.close()

        if self._tray_icon and self._tray_icon.isVisible():
            self._tray_icon.hide()

        event.accept()
