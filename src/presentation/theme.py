"""Modern, clean dark theme and typography for Vision-Based Sign Language Recognition."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette


class Color:
    """Central color palette for the modern dark desktop interface."""

    BACKGROUND = "#101419"
    SURFACE = "#151a21"
    CARD = "#191F28"
    CARD_HOVER = "#222a36"
    BORDER = "#2B3442"
    BORDER_LIGHT = "#3b4656"

    # Primary brand & accent colors
    PRIMARY = "#2563EB"
    PRIMARY_HOVER = "#3B82F6"
    PRIMARY_PRESSED = "#1D4ED8"

    ACCENT_TEAL = "#14B8A6"
    ACCENT_TEAL_LIGHT = "#2DD4BF"
    ACCENT_GREEN = "#10B981"
    ACCENT_RED = "#EF4444"
    ACCENT_RED_HOVER = "#DC2626"
    ACCENT_AMBER = "#F59E0B"

    # Typography
    TEXT_PRIMARY = "#F3F4F6"
    TEXT_SECONDARY = "#94A3B8"
    TEXT_MUTED = "#64748B"
    TEXT_DISABLED = "#475569"

    # Inputs & Controls
    INPUT_BG = "#13171e"
    INPUT_BORDER = "#2B3442"
    INPUT_FOCUS = "#2563EB"


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #101419;
    color: #F3F4F6;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", "Helvetica Neue", sans-serif;
    font-size: 13px;
}

/* ── Tab Widget (Modern Clean Header) ───────────────────── */
QTabWidget::pane {
    background-color: #101419;
    border: none;
    padding: 0px;
}

QTabBar {
    background-color: #151a21;
    border-bottom: 1px solid #2B3442;
}

QTabBar::tab {
    background-color: transparent;
    color: #94A3B8;
    padding: 12px 28px;
    border: none;
    border-bottom: 3px solid transparent;
    font-size: 14px;
    font-weight: 600;
    min-height: 24px;
}

QTabBar::tab:selected {
    background-color: #191F28;
    color: #F3F4F6;
    font-weight: 700;
    border-bottom: 3px solid #2563EB;
}

QTabBar::tab:hover:!selected {
    background-color: rgba(255, 255, 255, 0.04);
    color: #F3F4F6;
}

/* ── Push Buttons ────────────────────────────────────────── */
QPushButton {
    background-color: #191F28;
    color: #F3F4F6;
    border: 1px solid #2B3442;
    padding: 7px 18px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #222a36;
    border-color: #3b4656;
}

QPushButton:pressed {
    background-color: #13171e;
}

QPushButton:disabled {
    background-color: #151a21;
    border-color: #222a36;
    color: #475569;
}

QPushButton#primaryButton, QPushButton[primary="true"] {
    background-color: #2563EB;
    border: 1px solid #1D4ED8;
    color: #FFFFFF;
    font-weight: 600;
}

QPushButton#primaryButton:hover, QPushButton[primary="true"]:hover {
    background-color: #3B82F6;
    border-color: #2563EB;
}

QPushButton#primaryButton:pressed, QPushButton[primary="true"]:pressed {
    background-color: #1D4ED8;
}

QPushButton#successButton {
    background-color: #14B8A6;
    border: 1px solid #0d9488;
    color: #FFFFFF;
    font-weight: 600;
}

QPushButton#successButton:hover {
    background-color: #2DD4BF;
    color: #042f2e;
}

QPushButton#successButton:pressed {
    background-color: #0d9488;
}

QPushButton#dangerButton {
    background-color: #EF4444;
    border: 1px solid #DC2626;
    color: #FFFFFF;
}

QPushButton#dangerButton:hover {
    background-color: #F87171;
}

QPushButton#dangerButton:pressed {
    background-color: #DC2626;
}

QPushButton#actionButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: 1px solid #1D4ED8;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: 700;
}

QPushButton#actionButton:hover {
    background-color: #3B82F6;
}

/* ── Inputs & Selectors ──────────────────────────────────── */
QComboBox {
    background-color: #13171e;
    color: #F3F4F6;
    border: 1px solid #2B3442;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    min-height: 22px;
}

QComboBox:hover {
    border-color: #3b4656;
}

QComboBox:focus {
    border-color: #2563EB;
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
    border-top: 5px solid #94A3B8;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #151a21;
    color: #F3F4F6;
    border: 1px solid #2B3442;
    border-radius: 6px;
    selection-background-color: #2563EB;
    selection-color: #FFFFFF;
    padding: 4px;
    outline: none;
}

QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #13171e;
    color: #F3F4F6;
    border: 1px solid #2B3442;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    min-height: 22px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #2563EB;
}

/* ── CheckBox & RadioButton ──────────────────────────────── */
QCheckBox, QRadioButton {
    color: #F3F4F6;
    font-size: 13px;
    spacing: 8px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #3b4656;
    background-color: #13171e;
}

QRadioButton::indicator {
    border-radius: 8px;
}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #2563EB;
}

QCheckBox::indicator:checked {
    background-color: #2563EB;
    border-color: #2563EB;
}

QRadioButton::indicator:checked {
    background-color: #2563EB;
    border-color: #2563EB;
}

/* ── Sliders ─────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #2B3442;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #2563EB;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #2563EB;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #F3F4F6;
    border-color: #3B82F6;
}

/* ── Group Boxes ─────────────────────────────────────────── */
QGroupBox {
    background-color: #151a21;
    border: 1px solid #2B3442;
    border-radius: 8px;
    margin-top: 18px;
    padding-top: 14px;
    padding-left: 14px;
    padding-right: 14px;
    padding-bottom: 14px;
    font-weight: 600;
    font-size: 13px;
    color: #F3F4F6;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: #94A3B8;
}

/* ── List Widgets ────────────────────────────────────────── */
QListWidget {
    background-color: #13171e;
    border: 1px solid #2B3442;
    border-radius: 6px;
    color: #F3F4F6;
    padding: 4px;
}

QListWidget::item {
    padding: 8px 12px;
    border-radius: 4px;
    margin-bottom: 2px;
}

QListWidget::item:hover {
    background-color: #191F28;
}

QListWidget::item:selected {
    background-color: #2563EB;
    color: #FFFFFF;
}

/* ── Status Bar ──────────────────────────────────────────── */
QStatusBar {
    background-color: #13171e;
    border-top: 1px solid #2B3442;
    color: #94A3B8;
    font-size: 12px;
    padding: 2px 8px;
}

/* ── Progress Bar ────────────────────────────────────────── */
QProgressBar {
    background-color: #13171e;
    border: 1px solid #2B3442;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #14B8A6;
    border-radius: 3px;
}

/* ── Splitter ────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #2B3442;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* ── Dialogs ─────────────────────────────────────────────── */
QDialog {
    background-color: #101419;
    color: #F3F4F6;
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
