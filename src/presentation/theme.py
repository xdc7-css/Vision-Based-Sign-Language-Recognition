from __future__ import annotations

from PySide6.QtGui import QColor, QPalette


class Color:
    """Central color palette for the dark theme."""

    BACKGROUND = "#1e1e1e"
    SURFACE = "#252526"
    CARD = "#2d2d2d"
    CARD_HOVER = "#333333"
    BORDER = "#3c3c3c"
    PRIMARY = "#007acc"
    PRIMARY_HOVER = "#0997e8"
    PRIMARY_DARK = "#005a9e"
    ACCENT_GREEN = "#4ec9b0"
    ACCENT_RED = "#f14c4c"
    ACCENT_ORANGE = "#cca700"
    ACCENT_YELLOW = "#dcdcaa"
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#9d9d9d"
    TEXT_DISABLED = "#5a5a5a"
    INPUT_BG = "#3c3c3c"
    INPUT_BORDER = "#555555"


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}

/* ── Tab Widget ─────────────────────────────────────────── */
QTabWidget::pane {
    background-color: #252526;
    border: none;
    border-top: 1px solid #3c3c3c;
    padding: 0px;
}

QTabBar {
    background-color: #1e1e1e;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #9d9d9d;
    padding: 10px 22px;
    border: none;
    border-right: 1px solid #3c3c3c;
    min-width: 90px;
    font-size: 12px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #252526;
    color: #e0e0e0;
    border-bottom: 2px solid #007acc;
}

QTabBar::tab:hover:!selected {
    background-color: #333333;
    color: #e0e0e0;
}

/* ── Buttons ────────────────────────────────────────────── */
QPushButton {
    background-color: #0e639c;
    color: #ffffff;
    border: none;
    padding: 8px 20px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #094771;
}

QPushButton:pressed {
    background-color: #07355a;
}

QPushButton:disabled {
    background-color: #3c3c3c;
    color: #5a5a5a;
}

QPushButton#dangerButton {
    background-color: #a12626;
}

QPushButton#dangerButton:hover {
    background-color: #7a1c1c;
}

QPushButton#successButton {
    background-color: #1a6e4a;
}

QPushButton#successButton:hover {
    background-color: #145236;
}

/* ── ComboBox ────────────────────────────────────────────── */
QComboBox {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 6px 12px;
    border-radius: 4px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #007acc;
}

QComboBox:focus {
    border-color: #007acc;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #9d9d9d;
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #3c3c3c;
    selection-background-color: #094771;
    padding: 4px;
    outline: none;
}

/* ── Line Edit ──────────────────────────────────────────── */
QLineEdit {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 6px 12px;
    border-radius: 4px;
    min-height: 20px;
}

QLineEdit:focus {
    border-color: #007acc;
}

/* ── SpinBox ────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 6px 8px;
    border-radius: 4px;
    min-height: 20px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #007acc;
}

/* ── Slider ─────────────────────────────────────────────── */
QSlider::groove:horizontal {
    background: #3c3c3c;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #007acc;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #0997e8;
}

QSlider::sub-page:horizontal {
    background: #007acc;
    border-radius: 3px;
}

/* ── CheckBox / RadioButton ─────────────────────────────── */
QCheckBox, QRadioButton {
    color: #e0e0e0;
    spacing: 8px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 2px solid #555555;
    background-color: #3c3c3c;
}

QCheckBox::indicator:checked {
    background-color: #007acc;
    border-color: #007acc;
}

QRadioButton::indicator {
    border-radius: 9px;
}

QRadioButton::indicator:checked {
    background-color: #007acc;
    border-color: #007acc;
}

/* ── GroupBox ───────────────────────────────────────────── */
QGroupBox {
    color: #e0e0e0;
    border: 1px solid #3c3c3c;
    border-radius: 6px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    font-size: 13px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    background-color: #252526;
}

/* ── Labels ─────────────────────────────────────────────── */
QLabel {
    color: #e0e0e0;
    background: transparent;
}

QLabel#headingLabel {
    font-size: 20px;
    font-weight: 700;
    color: #e0e0e0;
    padding: 8px 0;
}

QLabel#subheadingLabel {
    font-size: 14px;
    font-weight: 500;
    color: #9d9d9d;
}

QLabel#valueLabel {
    font-size: 28px;
    font-weight: 700;
    color: #e0e0e0;
}

/* ── Text Edit / PlainTextEdit ──────────────────────────── */
QPlainTextEdit, QTextEdit {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 8px;
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 12px;
}

/* ── Scroll Bars ────────────────────────────────────────── */
QScrollBar:vertical {
    background: #1e1e1e;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #3c3c3c;
    min-height: 30px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #555555;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #1e1e1e;
    height: 10px;
}

QScrollBar::handle:horizontal {
    background: #3c3c3c;
    min-width: 30px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #555555;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ── Progress Bar ───────────────────────────────────────── */
QProgressBar {
    background-color: #3c3c3c;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #e0e0e0;
    font-size: 11px;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #007acc;
    border-radius: 4px;
}

/* ── Splitter ───────────────────────────────────────────── */
QSplitter::handle {
    background-color: #3c3c3c;
    width: 2px;
}

/* ── Status Bar ─────────────────────────────────────────── */
QStatusBar {
    background-color: #007acc;
    color: #ffffff;
    font-size: 12px;
    padding: 2px 8px;
}

QStatusBar::item {
    border: none;
}

/* ── Menu Bar ───────────────────────────────────────────── */
QMenuBar {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border-bottom: 1px solid #3c3c3c;
    padding: 2px;
}

QMenuBar::item {
    padding: 6px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #094771;
}

QMenu {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #3c3c3c;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #094771;
}

QMenu::separator {
    height: 1px;
    background: #3c3c3c;
    margin: 4px 8px;
}

/* ── Tool Tips ──────────────────────────────────────────── */
QToolTip {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 6px;
    font-size: 12px;
}
"""


def apply_dark_theme(app) -> None:
    """Apply the dark theme palette and stylesheet to a QApplication."""
    palette = QPalette()

    palette.setColor(QPalette.Window, QColor(Color.BACKGROUND))
    palette.setColor(QPalette.WindowText, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(Color.INPUT_BG))
    palette.setColor(QPalette.AlternateBase, QColor(Color.SURFACE))
    palette.setColor(QPalette.ToolTipBase, QColor(Color.SURFACE))
    palette.setColor(QPalette.ToolTipText, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.Text, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(Color.INPUT_BG))
    palette.setColor(QPalette.ButtonText, QColor(Color.TEXT_PRIMARY))
    palette.setColor(QPalette.BrightText, QColor(Color.ACCENT_RED))
    palette.setColor(QPalette.Highlight, QColor(Color.PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(Color.TEXT_DISABLED))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(Color.TEXT_DISABLED))

    app.setPalette(palette)
    app.setStyleSheet(DARK_STYLESHEET)
