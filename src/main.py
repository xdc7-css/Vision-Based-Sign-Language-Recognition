"""Application entry point — composition root with full dependency injection.

Integrates all modules:
    - GUI (MainWindow with 8 tabs)
    - Video Sources (adapter pattern)
    - Hand Detection (MediaPipe)
    - Motion Tracking
    - Dataset Collection
    - Static / Dynamic / Word Recognition
    - Arabic Text Builder (state machine)
    - Dialect Manager (Iraqi, Saudi, Egyptian)
    - Hybrid Recognition Pipeline (word > dynamic > alphabet priority)
    - Transparent Overlay System
    - Analytics Dashboard (PySide6 charts)
    - Performance Optimizer
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

# Ensure project root is in sys.path
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_cur_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Auto-dispatch to virtual environment if executed via system Python
if __name__ == "__main__" and not getattr(sys, "frozen", False):
    _venv_py = os.path.join(_project_root, ".venv", "Scripts", "python.exe")
    if os.path.isfile(_venv_py) and os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(_venv_py)):
        if os.environ.get("_ARSL_MAIN_DISPATCH") != "1":
            _env = os.environ.copy()
            _env["_ARSL_MAIN_DISPATCH"] = "1"
            _res = subprocess.run([_venv_py, "-m", "src.main"] + sys.argv[1:], cwd=_project_root, env=_env)
            sys.exit(_res.returncode)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.application.state import AppState
from src.application.use_cases import (
    CameraUseCases,
    DialectUseCases,
    ModelUseCases,
    VideoSourceUseCases,
)
from src.infrastructure.analytics import AnalyticsEngine
from src.infrastructure.camera_service import OpenCVCameraService
from src.infrastructure.arabic_renderer import ArabicTextRenderer
from src.infrastructure.config_manager import ConfigManager
from src.infrastructure.dialect_manager import DialectManager
from src.infrastructure.optimization import (
    OptimizationLevel,
    PerformanceOptimizer,
    PRESETS,
)
from src.infrastructure.overlay_system import OverlayConfig, OverlayManager
from src.infrastructure.recognition.hybrid_pipeline import HybridConfig, HybridRecognitionPipeline
from src.infrastructure.recognition.static_recognizer import StaticModelConfig
from src.infrastructure.recognition.dynamic_recognizer import DynamicModelConfig
from src.infrastructure.text_builder import ArabicTextBuilder
from src.presentation.main_window import MainWindow
from src.presentation.theme import apply_dark_theme


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_directories() -> None:
    """Ensure required data directories exist for all dialects."""
    dirs = [
        "data/models/iraqi",
        "data/models/saudi",
        "data/models/egyptian",
        "data/models/general",
        "data/captures",
        "data/datasets/iraqi",
        "data/datasets/saudi",
        "data/datasets/egyptian",
        "data/cache",
        "data/vocabulary",
        "data/logs",
    ]
    for d in dirs:
        os.makedirs(os.path.join(_project_root, d), exist_ok=True)


def build_services(state: AppState) -> dict:
    """Build and wire all application services. Returns a service registry dict."""

    # ── Configuration ────────────────────────────────────────────
    config_mgr = ConfigManager(config_dir="config")
    config_mgr.load()
    log_level = config_mgr.get("logging.level", "INFO")
    setup_logging(log_level)

    logger = logging.getLogger(__name__)

    # ── Infrastructure Services ──────────────────────────────────
    camera_service = OpenCVCameraService()
    arabic_renderer = ArabicTextRenderer()

    # ── Analytics (must be early — used everywhere) ──────────────
    analytics = AnalyticsEngine(window_size=60)

    # ── Performance Optimizer ────────────────────────────────────
    level_str = config_mgr.get("performance.profile", "balanced")
    level_map = {
        "max_quality": OptimizationLevel.MAX_QUALITY,
        "balanced": OptimizationLevel.BALANCED,
        "max_performance": OptimizationLevel.MAX_PERFORMANCE,
        "battery_saver": OptimizationLevel.BATTERY_SAVER,
    }
    opt_level = level_map.get(level_str, OptimizationLevel.BALANCED)
    optimizer = PerformanceOptimizer(PRESETS[opt_level])

    # ── Recognition Pipeline ─────────────────────────────────────
    pipeline_config = HybridConfig(
        word_threshold=config_mgr.get("recognition.hybrid.word_threshold", 0.85),
        dynamic_threshold=config_mgr.get("recognition.hybrid.dynamic_threshold", 0.75),
        alphabet_threshold=config_mgr.get("recognition.hybrid.alphabet_threshold", 0.90),
        cooldown_frames=config_mgr.get("recognition.hybrid.cooldown_frames", 20),
        enable_word_recognition=config_mgr.get("recognition.hybrid.enable_word", True),
        enable_dynamic_recognition=config_mgr.get("recognition.hybrid.enable_dynamic", True),
        enable_alphabet_recognition=config_mgr.get("recognition.hybrid.enable_alphabet", True),
    )
    pipeline = HybridRecognitionPipeline(config=pipeline_config)

    # ── Arabic Text Builder ──────────────────────────────────────
    text_builder = ArabicTextBuilder()

    # ── Model Configs (from YAML) ────────────────────────────────
    static_model_config = StaticModelConfig(
        model_type=config_mgr.get("recognition.static.model_type", "mlp"),
        test_size=0.2,
    )
    dynamic_model_config = DynamicModelConfig(
        architecture=config_mgr.get("recognition.dynamic.architectures", ["lstm"])[0],
        sequence_length=config_mgr.get("recognition.dynamic.sequence_length", 30),
        n_hidden=config_mgr.get("recognition.dynamic.hidden_size", 128),
        n_layers=config_mgr.get("recognition.dynamic.num_layers", 2),
        epochs=config_mgr.get("recognition.dynamic.epochs", 50),
        batch_size=config_mgr.get("recognition.dynamic.batch_size", 32),
    )

    # ── Dialect Manager ──────────────────────────────────────────
    dialect_manager = DialectManager(
        pipeline=pipeline,
        hybrid_config=pipeline_config,
        static_config=static_model_config,
        dynamic_config=dynamic_model_config,
        base_path=".",
    )
    active_dialect_str = config_mgr.get("dialect.active", "iraqi")
    from src.domain.dialect import DialectId
    from src.domain.entities import DialectInfo
    dialects_info = [
        DialectInfo("iraqi", "Iraqi", "عراقية"),
        DialectInfo("saudi", "Saudi", "سعودية"),
        DialectInfo("egyptian", "Egyptian", "مصرية"),
    ]
    state.dialects = dialects_info
    state.active_dialect = next((d for d in dialects_info if d.dialect_id == active_dialect_str), dialects_info[0])

    try:
        dialect_manager.switch_dialect(DialectId(active_dialect_str))
    except (ValueError, KeyError):
        dialect_manager.switch_dialect(DialectId.IRAQI)

    # ── Overlay System ───────────────────────────────────────────
    overlay_cfg = OverlayConfig(
        enabled=config_mgr.get("overlay.enabled", True),
        x=config_mgr.get("overlay.x", 100),
        y=config_mgr.get("overlay.y", 100),
        width=config_mgr.get("overlay.width", 600),
        height=config_mgr.get("overlay.height", 120),
        font_size=config_mgr.get("overlay.font_size", 36),
        font_name=config_mgr.get("overlay.font_name", "Arial"),
        rtl_enabled=config_mgr.get("overlay.rtl_enabled", True),
        background_opacity=config_mgr.get("overlay.background_opacity", 0),
    )
    overlay_manager = OverlayManager(overlay_cfg)

    # ── Use Cases ────────────────────────────────────────────────
    camera_use_cases = CameraUseCases(camera_service, state)
    dialect_use_cases = DialectUseCases.__new__(DialectUseCases)
    dialect_use_cases._dialect_repository = None  # type: ignore
    dialect_use_cases._state = state
    model_use_cases = ModelUseCases(state)
    source_use_cases = VideoSourceUseCases(state)

    # ── Wire signals ─────────────────────────────────────────────
    state.overlay_visibility_changed.connect(_on_overlay_visibility_changed(overlay_manager, text_builder))
    state.caption_updated.connect(_on_caption_updated(overlay_manager, analytics))

    logger.info("All services initialized successfully")

    # ── Model Exporter ───────────────────────────────────────────
    from src.infrastructure.model_exporter import ModelExporter
    model_exporter = ModelExporter()

    # ── Dataset Augmenter ────────────────────────────────────────
    from src.infrastructure.dataset.augmenter import DatasetAugmenter
    dataset_augmenter = DatasetAugmenter()

    logger.info("All services initialized successfully")

    return {
        "state": state,
        "config_mgr": config_mgr,
        "arabic_renderer": arabic_renderer,
        "camera_service": camera_service,
        "camera_use_cases": camera_use_cases,
        "dialect_use_cases": dialect_use_cases,
        "model_use_cases": model_use_cases,
        "source_use_cases": source_use_cases,
        "pipeline": pipeline,
        "text_builder": text_builder,
        "dialect_manager": dialect_manager,
        "overlay_manager": overlay_manager,
        "analytics": analytics,
        "optimizer": optimizer,
        "model_exporter": model_exporter,
        "dataset_augmenter": dataset_augmenter,
        "logger": logger,
    }


def _on_overlay_visibility_changed(overlay: OverlayManager, text_builder: ArabicTextBuilder):
    """Factory for overlay visibility signal handler."""
    def handler(visible: bool) -> None:
        if visible:
            overlay.update_text(text_builder.full_text)
            overlay.show()
        else:
            overlay.hide()
    return handler


def _on_caption_updated(overlay: OverlayManager, analytics: AnalyticsEngine):
    """Factory for caption updated signal handler."""
    def handler(text: str) -> None:
        overlay.update_text(text)
    return handler


def bootstrap() -> None:
    """Wire up all dependencies and launch the application."""
    state = AppState()
    services = build_services(state)

    app = QApplication(sys.argv)
    app.setApplicationName("ArSL Translation Overlay")
    app.setOrganizationName("ArSL-Project")
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # type: ignore

    apply_dark_theme(app)

    window = MainWindow(
        state=state,
        camera_use_cases=services["camera_use_cases"],
        dialect_use_cases=services["dialect_use_cases"],
        model_use_cases=services["model_use_cases"],
        source_use_cases=services["source_use_cases"],
        pipeline=services["pipeline"],
        text_builder=services["text_builder"],
        dialect_manager=services["dialect_manager"],
        overlay_manager=services["overlay_manager"],
        analytics=services["analytics"],
        optimizer=services["optimizer"],
    )
    window.show()

    services["camera_use_cases"].refresh_cameras()

    # Initialize overlay
    services["overlay_manager"].initialize()

    services["logger"].info("Application started successfully")
    if "ARSL_TEST_TIMEOUT" in os.environ:
        timeout_ms = int(os.environ["ARSL_TEST_TIMEOUT"])
        from PySide6.QtCore import QTimer
        def _auto_exit():
            window.close()
            app.quit()
        QTimer.singleShot(timeout_ms, _auto_exit)

    sys.exit(app.exec())


def main() -> None:
    """Entry point with global error handling."""
    try:
        from src.infrastructure.hand_detection.mediapipe_compatibility import patch_mediapipe
        patch_mediapipe()
        create_directories()
        bootstrap()
    except Exception as exc:
        logging.critical("Fatal error during startup: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
