"""Modern, clean dark theme and typography for Vision-Based Sign Language Recognition."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette


class Color:
    """Central color palette for the modern dark desktop interface."""

    BACKGROUND = "#111418"
    SURFACE = "#181d24"
    CARD = "#1e242d"
    CARD_HOVER = "#262e3a"
    BORDER = "#2b3441"
    BORDER_LIGHT = "#3b4656"

    # Primary brand & accent colors
    PRIMARY = "#2563eb"
    PRIMARY_HOVER = "#3b82f6"
    PRIMARY_PRESSED = "#1d4ed8"

    ACCENT_TEAL = "#0d9488"
    ACCENT_TEAL_LIGHT = "#14b8a6"
    ACCENT_GREEN = "#10b981"
    ACCENT_RED = "#ef4444"
    ACCENT_RED_HOVER = "#dc2626"
    ACCENT_AMBER = "#f59e0b"

    # Typography
    TEXT_PRIMARY = "#f8fafc"
    TEXT_SECONDARY = "#94a3b8"
    TEXT_MUTED = "#64748b"
    TEXT_DISABLED = "#475569"

    # Inputs & Controls
    INPUT_BG = "#14181f"
    INPUT_BORDER = "#2e3744"
    INPUT_FOCUS = "#2563eb"


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #111418;
    color: #f8fafc;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", "Helvetica Neue", sans-serif;
    font-size: 13px;
}

/* ── Tab Widget (Modern Clean Header) ───────────────────── */
QTabWidget::pane {
    background-color: #111418;
    border: none;
    padding: 0px;
}

QTabBar {
    background-color: #181d24;
    border-bottom: 1px solid #2b3441;
}

QTabBar::tab {
    background-color: transparent;
    color: #94a3b8;
    padding: 10px 24px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 13px;
    font-weight: 500;
    min-height: 24px;
}

QTabBar::tab:selected {
    background-color: #1e242d;
    color: #f8fafc;
    font-weight: 600;
    border-bottom: 2px solid #2563eb;
}

QTabBar::tab:hover:!selected {
    background-color: rgba(255, 255, 255, 0.03);
    color: #f1f5f9;
}

/* ── Push Buttons ────────────────────────────────────────── */
QPushButton {
    background-color: #1e242d;
    color: #f8fafc;
    border: 1px solid #2b3441;
    padding: 6px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #262e3a;
    border-color: #3b4656;
}

QPushButton:pressed {
    background-color: #151920;
}

QPushButton:disabled {
    background-color: #161a20;
    border-color: #222832;
    color: #475569;
}

QPushButton#primaryButton, QPushButton[primary="true"] {
    background-color: #2563eb;
    border: 1px solid #1d4ed8;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#primaryButton:hover, QPushButton[primary="true"]:hover {
    background-color: #3b82f6;
    border-color: #2563eb;
}

QPushButton#primaryButton:pressed, QPushButton[primary="true"]:pressed {
    background-color: #1d4ed8;
}

QPushButton#successButton {
    background-color: #0d9488;
    border: 1px solid #0f766e;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#successButton:hover {
    background-color: #14b8a6;
}

QPushButton#successButton:pressed {
    background-color: #0f766e;
}

QPushButton#dangerButton {
    background-color: #dc2626;
    border: 1px solid #b91c1c;
    color: #ffffff;
}

QPushButton#dangerButton:hover {
    background-color: #ef4444;
}

QPushButton#dangerButton:pressed {
    background-color: #991b1b;
}

/* ── Inputs & Selectors ──────────────────────────────────── */
QComboBox {
    background-color: #14181f;
    color: #f8fafc;
    border: 1px solid #2b3441;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    min-height: 22px;
}

QComboBox:hover {
    border-color: #3b4656;
}

QComboBox:focus {
    border-color: #2563eb;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 26px;
    border-left: none;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #94a3b8;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #181d24;
    color: #f8fafc;
    border: 1px solid #2b3441;
    border-radius: 6px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
    padding: 4px;
    outline: none;
}

QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #14181f;
    color: #f8fafc;
    border: 1px solid #2b3441;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    min-height: 22px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #2563eb;
}

/* ── CheckBox & RadioButton ──────────────────────────────── */
QCheckBox, QRadioButton {
    color: #e2e8f0;
    font-size: 13px;
    spacing: 8px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #3b4656;
    background-color: #14181f;
}

QRadioButton::indicator {
    border-radius: 8px;
}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #2563eb;
}

QCheckBox::indicator:checked {
    background-color: #2563eb;
    border-color: #2563eb;
}

QRadioButton::indicator:checked {
    background-color: #2563eb;
    border-color: #2563eb;
}

/* ── Sliders ─────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #2b3441;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #2563eb;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #2563eb;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #f1f5f9;
    border-color: #3b82f6;
}

/* ── Group Boxes ─────────────────────────────────────────── */
QGroupBox {
    background-color: #181d24;
    border: 1px solid #2b3441;
    border-radius: 8px;
    margin-top: 18px;
    padding-top: 14px;
    padding-left: 14px;
    padding-right: 14px;
    padding-bottom: 14px;
    font-weight: 600;
    font-size: 13px;
    color: #f8fafc;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: #94a3b8;
}

/* ── List Widgets ────────────────────────────────────────── */
QListWidget {
    background-color: #14181f;
    border: 1px solid #2b3441;
    border-radius: 6px;
    color: #f8fafc;
    padding: 4px;
}

QListWidget::item {
    padding: 6px 10px;
    border-radius: 4px;
}

QListWidget::item:hover {
    background-color: #1e242d;
}

QListWidget::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}

/* ── Status Bar ──────────────────────────────────────────── */
QStatusBar {
    background-color: #14181f;
    border-top: 1px solid #2b3441;
    color: #94a3b8;
    font-size: 12px;
    padding: 2px 8px;
}

/* ── Progress Bar ────────────────────────────────────────── */
QProgressBar {
    background-color: #14181f;
    border: 1px solid #2b3441;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #0d9488;
    border-radius: 3px;
}

/* ── Splitter ────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #2b3441;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}
"""


def apply_dark_theme(app) -> None:
    """Apply the unified dark theme to the QApplication instance."""
    app.setStyleSheet(DARK_STYLESHEET)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(Color.BACKGROUND))
    palette.setColor(QPalette.WindowText, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(Color.INPUT_BG))
    palette.setColor(QPalette.AlternateBase, QColor(Color.SURFACE))
    palette.setColor(QPalette.ToolTipBase, QColor(Color.CARD))
    palette.setColor(QPalette.ToolTipText, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.Text, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(Color.CARD))
    palette.setColor(QPalette.ButtonText, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.Highlight, QColor(Color.PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
