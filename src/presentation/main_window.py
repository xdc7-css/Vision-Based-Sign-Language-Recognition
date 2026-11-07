"""Primary application window with a clean, modern, student-friendly desktop layout.

Provides exactly 3 primary navigation pages:
1. Recognition (Default Home Viewport)
2. Teach Signs (Gesture Recording & Management)
3. Train Model (Neural Training & Validation)

Settings is accessible via a modal dialog (⚙ button in top header or Ctrl+,).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
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
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.dialect_manager import DialectManager
from src.infrastructure.optimization import PerformanceOptimizer
from src.infrastructure.overlay_system import OverlayManager
from src.infrastructure.recognition.hybrid_pipeline import HybridRecognitionPipeline
from src.infrastructure.text_builder import ArabicTextBuilder
from src.presentation.dialogs.settings_dialog import SettingsDialog
from src.presentation.tabs.analytics import AnalyticsTab
from src.presentation.tabs.dataset import DatasetCollectionTab
from src.presentation.tabs.overlay import OverlayTab
from src.presentation.tabs.recognition import RecognitionTab
from src.presentation.tabs.settings import SettingsTab
from src.presentation.tabs.training import ModelTrainingTab


class MainWindow(QMainWindow):
    """Primary application window with clean 3-page navigation inspired by Teachable Machine."""

    WINDOW_TITLE = "Vision-Based Sign Language Recognition — ArSL Studio"
    DEFAULT_WIDTH = 1320
    DEFAULT_HEIGHT = 820

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
        self._config_mgr = ConfigManager(config_dir="config")
        self._logger = logging.getLogger(__name__)

        self._setup_window()
        self._setup_menu_bar()
        self._setup_tabs()
        self._setup_status_bar()
        self._setup_system_tray()
        self._connect_signals()
        self._update_header_model_status()

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

        self._teach_action = QAction("Teach Signs Studio", self)
        self._teach_action.setShortcut("Ctrl+2")
        self._teach_action.triggered.connect(lambda: self._tab_widget.setCurrentIndex(1))
        file_menu.addAction(self._teach_action)

        self._train_action = QAction("Train Model Studio", self)
        self._train_action.setShortcut("Ctrl+3")
        self._train_action.triggered.connect(lambda: self._tab_widget.setCurrentIndex(2))
        file_menu.addAction(self._train_action)

        file_menu.addSeparator()

        self._settings_action = QAction("Preferences...", self)
        self._settings_action.setShortcut("Ctrl+,")
        self._settings_action.triggered.connect(self.open_settings_dialog)
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

        # Primary 3 Application Pages
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

        # Retained modules for dialogs / tests
        self._settings_tab = SettingsTab(self._state, config_manager=self._config_mgr)
        self._overlay_tab = OverlayTab(self._state, overlay_manager=self._overlay_manager)
        self._analytics_tab = AnalyticsTab(engine=self._analytics)

        # Add EXACTLY 3 Primary Tabs
        self._tab_widget.addTab(self._recognition_tab, "Recognition")
        self._tab_widget.addTab(self._dataset_tab, "Teach Signs")
        self._tab_widget.addTab(self._training_tab, "Train Model")

        # Top-Right Header Corner Widget (Model Status + Preferences Gear Button)
        corner_widget = QWidget()
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 16, 0)
        corner_layout.setSpacing(10)

        self._header_model_badge = QLabel("● Model Ready")
        self._header_model_badge.setStyleSheet(
            "color: #10B981; font-weight: 600; font-size: 12px; padding: 4px 10px; "
            "background: #064e3b; border: 1px solid #047857; border-radius: 4px;"
        )
        corner_layout.addWidget(self._header_model_badge)

        self._header_settings_btn = QPushButton("⚙ Preferences")
        self._header_settings_btn.setToolTip("Open preferences and hardware settings")
        self._header_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #191F28;
                border: 1px solid #2B3442;
                border-radius: 6px;
                padding: 5px 14px;
                font-size: 12px;
                font-weight: 600;
                color: #F3F4F6;
            }
            QPushButton:hover {
                background-color: #222a36;
                border-color: #3b4656;
            }
        """)
        self._header_settings_btn.clicked.connect(self.open_settings_dialog)
        corner_layout.addWidget(self._header_settings_btn)

        self._tab_widget.setCornerWidget(corner_widget, Qt.TopRightCorner)
        self.setCentralWidget(self._tab_widget)

    def _setup_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("System Ready")
        self._status_bar.addWidget(self._status_label, stretch=1)

        # Hidden by default from main view to reduce clutter, preserved for telemetry/tests
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("padding: 0 10px; color: #94A3B8;")
        self._fps_label.setVisible(False)
        self._status_bar.addPermanentWidget(self._fps_label)

        self._latency_label = QLabel("Latency: -- ms")
        self._latency_label.setStyleSheet("padding: 0 10px; color: #94A3B8;")
        self._latency_label.setVisible(False)
        self._status_bar.addPermanentWidget(self._latency_label)

        self._source_label = QLabel("Camera: Standby")
        self._source_label.setStyleSheet("padding: 0 10px; color: #94A3B8; font-weight: 500;")
        self._status_bar.addPermanentWidget(self._source_label)

        self._dialect_label = QLabel("Model: Iraqi Multimodal")
        self._dialect_label.setStyleSheet("padding: 0 10px; color: #14B8A6; font-weight: 600;")
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
        self._state.active_dialect_changed.connect(self._on_active_dialect_changed)
        self._state.error_occurred.connect(self._on_error)

        self._toggle_overlay_action.triggered.connect(self._toggle_overlay_action_triggered)
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # Cross-page guided workflow wiring
        self._recognition_tab.navigate_to_teach.connect(lambda: self._tab_widget.setCurrentIndex(1))
        self._recognition_tab.navigate_to_train.connect(lambda: self._tab_widget.setCurrentIndex(2))
        self._dataset_tab.navigate_to_train.connect(lambda: self._tab_widget.setCurrentIndex(2))
        self._training_tab.navigate_to_recognition.connect(self._on_navigate_to_recognition)

    # ── Slots & Workflow Coordination ──────────────────────────────

    def open_settings_dialog(self) -> None:
        """Open the sleek preferences modal dialog."""
        dlg = SettingsDialog(self._state, config_manager=self._config_mgr, parent=self)
        dlg.settings_saved.connect(self._on_settings_saved)
        dlg.exec()

    def _on_settings_saved(self, settings_data: dict) -> None:
        # Check if UI diagnostics toggled
        ui_cfg = settings_data.get("ui", {})
        show_fps = ui_cfg.get("show_fps", False)
        self._fps_label.setVisible(show_fps)
        self._latency_label.setVisible(show_fps)

        # Check if active model changed in preferences
        recog_cfg = settings_data.get("recognition", {})
        model_key = recog_cfg.get("model", "multimodal")
        if hasattr(self._recognition_tab, "_engine_selector"):
            idx = self._recognition_tab._engine_selector.findData(model_key)
            if idx >= 0:
                self._recognition_tab._engine_selector.setCurrentIndex(idx)

        # Check camera index
        cam_cfg = settings_data.get("camera", {})
        dev_idx = cam_cfg.get("device_index", 0)
        self._state.selected_camera_index = dev_idx
        self._recognition_tab._refresh_camera_list()
        self._dataset_tab._refresh_camera_list()
        self._update_header_model_status()

    def _on_navigate_to_recognition(self) -> None:
        """Handle 'Test Model in Recognition' click from Training tab."""
        self._tab_widget.setCurrentIndex(0)
        if hasattr(self._recognition_tab, "_engine_selector"):
            m_idx = self._recognition_tab._engine_selector.findData("multimodal")
            if m_idx >= 0:
                self._recognition_tab._engine_selector.setCurrentIndex(m_idx)
            else:
                c_idx = self._recognition_tab._engine_selector.findData("custom")
                if c_idx >= 0:
                    self._recognition_tab._engine_selector.setCurrentIndex(c_idx)
        if hasattr(self._recognition_tab, "_update_multimodal_model_status_ui"):
            self._recognition_tab._update_multimodal_model_status_ui()
        self._update_header_model_status()

    def _update_header_model_status(self) -> None:
        """Sync top-right header model badge and status bar with current trained models."""
        mm_classes_path = Path("data/models/multimodal/classes.json")
        mm_model_path = Path("data/models/multimodal/static_model.pkl")

        if mm_model_path.exists() and mm_classes_path.exists():
            try:
                with open(mm_classes_path, "r", encoding="utf-8") as f:
                    classes = json.load(f)
                self._header_model_badge.setText(f"● Model Ready: {len(classes)} signs")
                self._header_model_badge.setStyleSheet(
                    "color: #10B981; font-weight: 600; font-size: 12px; padding: 4px 10px; "
                    "background: #064e3b; border: 1px solid #047857; border-radius: 4px;"
                )
                self._dialect_label.setText(f"Model: Multimodal ({len(classes)} signs)")
                return
            except Exception:
                pass

        iraqi_classes_path = Path("data/models/iraqi/classes.json")
        iraqi_model_path = Path("data/models/iraqi/static_model.pkl")
        if iraqi_model_path.exists() and iraqi_classes_path.exists():
            try:
                with open(iraqi_classes_path, "r", encoding="utf-8") as f:
                    classes = json.load(f)
                self._header_model_badge.setText(f"● Model Ready: {len(classes)} signs")
                self._header_model_badge.setStyleSheet(
                    "color: #10B981; font-weight: 600; font-size: 12px; padding: 4px 10px; "
                    "background: #064e3b; border: 1px solid #047857; border-radius: 4px;"
                )
                self._dialect_label.setText(f"Model: Iraqi Custom ({len(classes)} signs)")
                return
            except Exception:
                pass

        # Fallback to KArSL
        self._header_model_badge.setText("● KArSL-502 (502 signs)")
        self._header_model_badge.setStyleSheet(
            "color: #38BDF8; font-weight: 600; font-size: 12px; padding: 4px 10px; "
            "background: #0c4a6e; border: 1px solid #0369a1; border-radius: 4px;"
        )
        self._dialect_label.setText("Model: KArSL-502 / Iraqi")

    def _on_active_dialect_changed(self, d) -> None:
        self._dialect_label.setText(f"Model: {d.display_name if d else 'KArSL-502'}")
        self._update_header_model_status()

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
            self._status_label.setStyleSheet("color: #10B981; font-weight: 600;")
        elif status.lower() == "error":
            self._status_label.setStyleSheet("color: #EF4444; font-weight: 600;")
        else:
            self._status_label.setStyleSheet("color: #94A3B8;")

    def _on_error(self, message: str) -> None:
        self._status_label.setText(f"Notice: {message}")
        self._status_label.setStyleSheet("color: #EF4444;")

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
            "About ArSL Studio",
            "<h2>Arabic Sign Language Studio (ArSL Studio)</h2>"
            "<p><b>Version:</b> 2.1.0</p>"
            "<p>A streamlined, professional desktop application for Arabic Sign Language Recognition "
            "using Multimodal MediaPipe tracking (Dual Hands + Face) and deep learning.</p>"
            "<hr>"
            "<p><b>Workflow:</b> Teach Signs → Train Model → Recognize Signs</p>"
            "<p><b>Features:</b> DirectShow multi-camera capture (HD Webcam & e2eSoft iVCam), "
            "instant sample recording with burst mode, automated dataset management, "
            "multimodal neural network training, Arabic text sentence accumulator, and Text-to-Speech audio.</p>"
            "<p><i>Computer Engineering Graduation Project</i></p>",
        )

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab switching and safely coordinate camera hardware ownership."""
        if index == 0:  # Recognition tab
            if hasattr(self, "_dataset_tab") and getattr(self._dataset_tab, "_is_streaming", False):
                self._dataset_tab._stop_camera()
            if hasattr(self, "_recognition_tab"):
                self._recognition_tab._refresh_camera_list()
                if hasattr(self._recognition_tab, "_refresh_model_overview_banner"):
                    self._recognition_tab._refresh_model_overview_banner()
            self._refresh_dialect_menu()
            self._update_header_model_status()
        elif index == 1:  # Teach Signs
            if hasattr(self, "_recognition_tab") and getattr(self._recognition_tab, "_is_streaming", False):
                self._recognition_tab._stop_camera()
            if hasattr(self, "_dataset_tab"):
                self._dataset_tab._refresh_camera_list()
                self._dataset_tab._scan_and_populate_classes()
        elif index == 2:  # Train Model
            if hasattr(self, "_recognition_tab") and getattr(self._recognition_tab, "_is_streaming", False):
                self._recognition_tab._stop_camera()
            if hasattr(self, "_dataset_tab") and getattr(self._dataset_tab, "_is_streaming", False):
                self._dataset_tab._stop_camera()
            if hasattr(self, "_training_tab"):
                self._training_tab._sync_active_dialect()
                if hasattr(self._training_tab, "refresh_model_sync_status"):
                    self._training_tab.refresh_model_sync_status()

    def _refresh_dialect_menu(self) -> None:
        if self._dialect_manager is None:
            return
        active = self._dialect_manager.active_dialect.value
        for did, action in self._dialect_actions.items():
            action.setChecked(did == active)

    def _switch_dialect(self, dialect_id) -> None:
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
            self._update_header_model_status()

    def closeEvent(self, event) -> None:
        """Handle application shutdown — clean up all resources."""
        if hasattr(self, "_recognition_tab") and self._recognition_tab:
            try:
                self._recognition_tab.close()
            except Exception:
                pass

        if hasattr(self, "_dataset_tab") and self._dataset_tab:
            try:
                self._dataset_tab.close()
            except Exception:
                pass

        if self._source_use_cases:
            self._source_use_cases.stop_source()

        if self._overlay_manager:
            self._overlay_manager.close()

        if self._tray_icon and self._tray_icon.isVisible():
            self._tray_icon.hide()

        event.accept()
