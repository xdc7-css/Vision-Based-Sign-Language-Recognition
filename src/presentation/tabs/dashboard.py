from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.presentation.widgets import SectionHeader, StatCard


class DashboardTab(QWidget):
    """Main overview tab with system status and quick actions."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        # Header
        header = SectionHeader(
            "Dashboard",
            "System overview and quick actions",
        )
        layout.addWidget(header)

        # Stats row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(12)

        self._status_card = StatCard("Pipeline Status", "Idle", "●")
        self._status_card.set_value_color("#9d9d9d")
        stats_layout.addWidget(self._status_card)

        self._fps_card = StatCard("Frame Rate", "0 FPS", "◉")
        stats_layout.addWidget(self._fps_card)

        self._source_card = StatCard("Video Source", "None", "▶")
        stats_layout.addWidget(self._source_card)

        self._dialect_card = StatCard("Active Dialect", "None", "◈")
        stats_layout.addWidget(self._dialect_card)

        layout.addLayout(stats_layout)

        # Quick actions
        actions_section = QFrame()
        actions_section.setObjectName("actionsSection")
        actions_section.setStyleSheet("""
            QFrame#actionsSection {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        actions_layout = QVBoxLayout(actions_section)
        actions_layout.setSpacing(12)

        actions_label = QLabel("Quick Actions")
        actions_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #e0e0e0;")
        actions_layout.addWidget(actions_label)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)

        self._start_btn = QPushButton("▶  Start Recognition")
        self._start_btn.setObjectName("successButton")
        buttons_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setObjectName("dangerButton")
        self._stop_btn.setEnabled(False)
        buttons_layout.addWidget(self._stop_btn)

        self._overlay_btn = QPushButton("◻  Toggle Overlay")
        buttons_layout.addWidget(self._overlay_btn)

        buttons_layout.addStretch()
        actions_layout.addLayout(buttons_layout)
        layout.addWidget(actions_section)

        # Recent activity / status feed
        feed_section = QFrame()
        feed_section.setObjectName("feedSection")
        feed_section.setStyleSheet("""
            QFrame#feedSection {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        feed_layout = QVBoxLayout(feed_section)

        feed_label = QLabel("System Activity")
        feed_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #e0e0e0;")
        feed_layout.addWidget(feed_label)

        self._feed_text = QLabel("Ready. Select a video source and dialect to begin.")
        self._feed_text.setWordWrap(True)
        self._feed_text.setStyleSheet("font-size: 12px; color: #9d9d9d; padding: 8px 0;")
        feed_layout.addWidget(self._feed_text)

        feed_layout.addStretch()
        layout.addWidget(feed_section, stretch=1)

    def _connect_signals(self) -> None:
        self._state.pipeline_status_changed.connect(self._on_status_changed)
        self._state.fps_updated.connect(self._on_fps_updated)
        self._state.active_source_changed.connect(self._on_source_changed)
        self._state.active_dialect_changed.connect(self._on_dialect_changed)
        self._state.log_message.connect(self._on_log_message)

        self._start_btn.clicked.connect(self._on_start_clicked)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._overlay_btn.clicked.connect(self._on_overlay_toggle)

    def _on_start_clicked(self) -> None:
        self._state.pipeline_status = "starting"

    def _on_stop_clicked(self) -> None:
        self._state.pipeline_status = "stopping"

    def _on_overlay_toggle(self) -> None:
        self._state.overlay_visible = not self._state.overlay_visible

    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        color_map = {
            "idle": "#9d9d9d",
            "starting": "#cca700",
            "streaming": "#4ec9b0",
            "stopping": "#cca700",
            "error": "#f14c4c",
        }
        self._status_card.set_value(status.title())
        self._status_card.set_value_color(color_map.get(status, "#9d9d9d"))

        is_running = status in ("streaming", "starting")
        self._start_btn.setEnabled(not is_running)
        self._stop_btn.setEnabled(is_running)

    @Slot(float)
    def _on_fps_updated(self, fps: float) -> None:
        self._fps_card.set_value(f"{fps:.1f} FPS")

    @Slot(object)
    def _on_source_changed(self, source) -> None:
        if source:
            self._source_card.set_value(source.display_name)
        else:
            self._source_card.set_value("None")

    @Slot(object)
    def _on_dialect_changed(self, dialect) -> None:
        if dialect:
            self._dialect_card.set_value(dialect.display_name)
        else:
            self._dialect_card.set_value("None")

    @Slot(str, str)
    def _on_log_message(self, level: str, message: str) -> None:
        current = self._feed_text.text()
        lines = current.split("\n")[-5:]
        lines.append(f"• {message}")
        self._feed_text.setText("\n".join(lines))
