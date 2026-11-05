from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QSystemTrayIcon,
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
from src.presentation.tabs.dashboard import DashboardTab
from src.presentation.tabs.dataset import DatasetCollectionTab
from src.presentation.tabs.logs import LogsTab
from src.presentation.tabs.overlay import OverlayTab
from src.presentation.tabs.recognition import RecognitionTab
from src.presentation.tabs.settings import SettingsTab
from src.presentation.tabs.training import ModelTrainingTab


class MainWindow(QMainWindow):
    """Primary application window with tabbed interface — 8 tabs total."""

    WINDOW_TITLE = "ArSL Translation Overlay System"
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

        self._setup_window()
        self._setup_menu_bar()
        self._setup_tabs()
        self._setup_status_bar()
        self._setup_system_tray()
        self._connect_signals()

        # If no trained models exist for the active dialect, open Dataset tab by default
        has_trained_model = False
        if self._dialect_manager and self._dialect_manager.active_dialect_config:
            from pathlib import Path
            model_p = Path(self._dialect_manager.active_dialect_config.model_path)
            if (model_p / "static_model.pkl").exists() or (model_p / "dynamic_model.pt").exists():
                has_trained_model = True
        if not has_trained_model:
            self._tab_widget.setCurrentIndex(1)  # Dataset Tab

    # ── Setup ──────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setMinimumSize(1024, 700)
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)

        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.DEFAULT_WIDTH) // 2
        y = (screen.height() - self.DEFAULT_HEIGHT) // 2
        self.move(x, y)

    def _setup_menu_bar(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        self._select_source_action = QAction("Select Video Source...", self)
        self._select_source_action.setShortcut("Ctrl+O")
        file_menu.addAction(self._select_source_action)
        file_menu.addSeparator()
        self._settings_action = QAction("Preferences...", self)
        self._settings_action.setShortcut("Ctrl+,")
        file_menu.addAction(self._settings_action)
        file_menu.addSeparator()
        self._quit_action = QAction("Exit", self)
        self._quit_action.setShortcut("Ctrl+Q")
        self._quit_action.triggered.connect(self.close)
        file_menu.addAction(self._quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        self._toggle_overlay_action = QAction("Toggle Overlay", self)
        self._toggle_overlay_action.setShortcut("Ctrl+T")
        view_menu.addAction(self._toggle_overlay_action)
        view_menu.addSeparator()
        self._fullscreen_action = QAction("Toggle Fullscreen", self)
        self._fullscreen_action.setShortcut("F11")
        self._fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(self._fullscreen_action)

        # Dialect menu
        dialect_menu = menubar.addMenu("&Dialect")
        self._dialect_actions: dict[str, QAction] = {}
        from src.domain.dialect import DialectId, DIALECT_DISPLAY_NAMES
        for did, display in DIALECT_DISPLAY_NAMES.items():
            action = QAction(display, self, checkable=True)
            action.triggered.connect(lambda checked, d=did: self._switch_dialect(d))
            dialect_menu.addAction(action)
            self._dialect_actions[did.value] = action

        # Help menu
        help_menu = menubar.addMenu("&Help")
        self._about_action = QAction("About ArSL Overlay", self)
        self._about_action.triggered.connect(self._show_about)
        help_menu.addAction(self._about_action)

    def _setup_tabs(self) -> None:
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabPosition(QTabWidget.North)
        self._tab_widget.setDocumentMode(True)
        self._tab_widget.setMovable(False)

        # Create tabs (8 tabs total)
        self._dashboard_tab = DashboardTab(self._state)
        self._dataset_tab = DatasetCollectionTab(self._state, self._source_use_cases)
        self._training_tab = ModelTrainingTab(
            self._state,
            dialect_manager=self._dialect_manager,
            pipeline=self._pipeline,
        )

        self._recognition_tab = RecognitionTab(
            self._state, self._source_use_cases,
            pipeline=self._pipeline,
            text_builder=self._text_builder,
            dialect_manager=self._dialect_manager,
            analytics=self._analytics,
            optimizer=self._optimizer,
        )

        self._overlay_tab = OverlayTab(self._state, overlay_manager=self._overlay_manager)
        self._analytics_tab = AnalyticsTab(engine=self._analytics)
        self._settings_tab = SettingsTab(self._state)
        self._logs_tab = LogsTab(self._state)

        # Add tabs
        self._tab_widget.addTab(self._dashboard_tab, "📊  Dashboard")
        self._tab_widget.addTab(self._dataset_tab, "📷  Dataset")
        self._tab_widget.addTab(self._training_tab, "🧠  Training")
        self._tab_widget.addTab(self._recognition_tab, "🎯  Recognition")
        self._tab_widget.addTab(self._overlay_tab, "🔲  Overlay")
        self._tab_widget.addTab(self._analytics_tab, "📈  Analytics")
        self._tab_widget.addTab(self._settings_tab, "⚙  Settings")
        self._tab_widget.addTab(self._logs_tab, "📋  Logs")

        self.setCentralWidget(self._tab_widget)

    def _setup_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label, stretch=1)

        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("padding: 0 12px;")
        self._status_bar.addPermanentWidget(self._fps_label)

        self._latency_label = QLabel("Delay: -- ms")
        self._latency_label.setStyleSheet("padding: 0 12px;")
        self._status_bar.addPermanentWidget(self._latency_label)

        self._source_label = QLabel("Source: None")
        self._source_label.setStyleSheet("padding: 0 12px;")
        self._status_bar.addPermanentWidget(self._source_label)

        self._dialect_label = QLabel("Dialect: Iraqi")
        self._dialect_label.setStyleSheet("padding: 0 12px;")
        self._status_bar.addPermanentWidget(self._dialect_label)

    def _setup_system_tray(self) -> None:
        self._tray_icon: Optional[QSystemTrayIcon] = None

        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray_icon = QSystemTrayIcon(self)
            self._tray_icon.setToolTip("ArSL Translation Overlay")

            tray_menu = QMenu()
            show_action = QAction("Show Window", self)
            show_action.triggered.connect(self.show_and_raise)
            tray_menu.addAction(show_action)

            toggle_action = QAction("Toggle Overlay", self)
            toggle_action.triggered.connect(self._toggle_overlay_from_tray)
            tray_menu.addAction(toggle_action)

            tray_menu.addSeparator()
            quit_action = QAction("Exit", self)
            quit_action.triggered.connect(QApplication.quit)
            tray_menu.addAction(quit_action)

            self._tray_icon.setContextMenu(tray_menu)

            try:
                from PySide6.QtGui import QPixmap
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
            lambda s: self._source_label.setText(f"Source: {s.display_name if s else 'None'}")
        )
        self._state.active_dialect_changed.connect(
            lambda d: self._dialect_label.setText(f"Dialect: {d.display_name if d else 'None'}")
        )
        self._state.error_occurred.connect(self._on_error)

        self._toggle_overlay_action.triggered.connect(self._toggle_overlay_action_triggered)
        self._select_source_action.triggered.connect(self._select_source)

        # Tab switch → refresh content
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

    # ── Slots ──────────────────────────────────────────────────────

    def show_and_raise(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_status_changed(self, status: str) -> None:
        self._status_label.setText(status.title())
        if status == "streaming":
            self._status_label.setStyleSheet("color: #4ec9b0;")
        elif status == "error":
            self._status_label.setStyleSheet("color: #f14c4c;")
        else:
            self._status_label.setStyleSheet("color: inherit;")

    def _on_error(self, message: str) -> None:
        self._status_label.setText(f"Error: {message}")
        self._status_label.setStyleSheet("color: #f14c4c;")

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
            "About ArSL Overlay",
            "<h2>Real-Time Arabic Sign Language Translation Overlay</h2>"
            "<p>Version 1.0.0</p>"
            "<p>A desktop application that translates Arabic Sign Language "
            "into Arabic text in real time using computer vision and deep learning.</p>"
            "<hr>"
            "<p><b>Modules:</b> GUI, Video Sources, Hand Detection, Motion Tracking, "
            "Dataset Collection, Static/Dynamic/Word Recognition, Arabic Text Builder, "
            "Dialect Manager (Iraqi/Saudi/Egyptian), Overlay System, Analytics</p>"
            "<p>Computer Engineering Graduation Project</p>",
        )

    def _select_source(self) -> None:
        self._tab_widget.setCurrentIndex(3)  # Recognition tab

    def _on_tray_activated(self, reason: int) -> None:
        if reason == QSystemTrayIcon.DoubleClick:
            self.show_and_raise()

    def _on_tab_changed(self, index: int) -> None:
        if index == 3:  # Recognition tab
            self._refresh_dialect_menu()
        elif index == 5:  # Analytics tab
            if hasattr(self, '_analytics_tab'):
                self._analytics_tab.start()

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
            self._dialect_label.setText(f"Dialect: {cfg.display_name if cfg else did_str.title()}")
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
