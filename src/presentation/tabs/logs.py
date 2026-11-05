from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.presentation.widgets import LogViewer, SectionHeader


class LogsTab(QWidget):
    """Real-time log viewer tab."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        header = SectionHeader(
            "Logs",
            "Application event log for debugging and monitoring",
        )
        layout.addWidget(header)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)

        toolbar.addWidget(QLabel("Filter Level:"))

        self._level_combo = QComboBox()
        self._level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._level_combo.setCurrentIndex(1)
        toolbar.addWidget(self._level_combo)

        toolbar.addWidget(QLabel("Search:"))

        from PySide6.QtWidgets import QLineEdit

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Filter logs...")
        self._search_input.setFixedWidth(250)
        toolbar.addWidget(self._search_input)

        toolbar.addStretch()

        self._clear_btn = QPushButton("🗑  Clear Logs")
        self._clear_btn.setFixedWidth(120)
        toolbar.addWidget(self._clear_btn)

        self._save_btn = QPushButton("💾  Save Logs")
        self._save_btn.setFixedWidth(120)
        toolbar.addWidget(self._save_btn)

        layout.addLayout(toolbar)

        # Log viewer
        self._log_viewer = LogViewer()
        layout.addWidget(self._log_viewer, stretch=1)

        # Connect toolbar actions
        self._clear_btn.clicked.connect(self._log_viewer.clear_logs)
        self._save_btn.clicked.connect(self._save_logs)
        self._level_combo.currentTextChanged.connect(self._on_level_changed)

    def _connect_signals(self) -> None:
        self._state.log_message.connect(self._log_viewer.append_log)

    def _on_level_changed(self, level_text: str) -> None:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        self._log_viewer.set_min_level(level_map.get(level_text, logging.INFO))

    def _save_logs(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Logs",
            "arsl_logs.txt",
            "Text Files (*.txt);;All Files (*.*)",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._log_viewer.toPlainText())
                self._state.log("info", f"Logs saved to {path}")
            except OSError as e:
                self._state.error(f"Failed to save logs: {e}")
