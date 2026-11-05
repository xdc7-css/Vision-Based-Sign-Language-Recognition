"""AnalyticsTab — real-time performance monitoring dashboard with PySide6 charts.

Displays:
    - FPS gauge (line chart)
    - CPU + GPU usage (area chart)
    - Memory usage (bar chart)
    - Recognition accuracy (percentage + rolling chart)
    - Translation delay (line chart)

All charts update in real-time from the AnalyticsEngine.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QCategoryAxis, QBarSeries, QBarSet, QBarCategoryAxis, QPieSeries
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from src.infrastructure.analytics import AnalyticsEngine, AnalyticsSnapshot


# ── Chart Styles ────────────────────────────────────────────────────

CHART_BG = "background: transparent;"
AXIS_COLOR = QColor(180, 180, 180)
GRID_COLOR = QColor(50, 50, 50)
FPS_COLOR = QColor(0, 200, 255)
CPU_COLOR = QColor(255, 180, 50)
GPU_COLOR = QColor(180, 80, 255)
MEM_COLOR = QColor(80, 200, 120)
ACC_COLOR = QColor(100, 255, 150)
DELAY_COLOR = QColor(255, 120, 100)


def _make_chart_view(chart: QChart, title: str, height: int = 200) -> QChartView:
    chart.setTitle(title)
    chart.setBackgroundBrush(QColor(30, 30, 30))
    chart.setTitleBrush(QColor(220, 220, 220))
    chart.setAnimationOptions(QChart.SeriesAnimations)
    chart.legend().setLabelColor(QColor(200, 200, 200))
    chart.legend().setFont(QFont("Segoe UI", 9))

    view = QChartView(chart)
    view.setRenderHint(QPainter.Antialiasing)
    view.setMinimumHeight(height)
    view.setStyleSheet(CHART_BG)
    return view


def _create_value_axis(label: str, min_v: float, max_v: float) -> QValueAxis:
    axis = QValueAxis()
    axis.setLabelsColor(AXIS_COLOR)
    axis.setGridLineColor(GRID_COLOR)
    axis.setTitleText(label)
    axis.setTitleBrush(AXIS_COLOR)
    axis.setRange(min_v, max_v)
    axis.setLabelsVisible(True)
    return axis


# ── Metrics Card ────────────────────────────────────────────────────

class MetricCard(QFrame):
    """Single metric display card with label, value, and unit."""

    def __init__(self, title: str, unit: str = "", color: str = "#00CCFF") -> None:
        super().__init__()
        self._unit = unit
        self._value_label = QLabel("0.0")
        self._title_label = QLabel(title)

        self.setStyleSheet(f"""
            MetricCard {{
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        self.setMinimumHeight(100)

        self._value_label.setStyleSheet(f"color: {color}; font-size: 36px; font-weight: bold;")
        self._title_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")

        layout = QVBoxLayout(self)
        layout.addWidget(self._title_label)
        layout.addWidget(self._value_label)
        layout.addStretch()

    def update_value(self, value: float, decimals: int = 1) -> None:
        self._value_label.setText(f"{value:.{decimals}f} {self._unit}")


# ── Analytics Tab ───────────────────────────────────────────────────

class AnalyticsTab(QWidget):
    """Main analytics dashboard tab with live charts and metric cards."""

    def __init__(self, engine: Optional[AnalyticsEngine] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._engine = engine or AnalyticsEngine()
        self._logger = logging.getLogger(__name__)
        self._max_points = 60
        self._time_counter = 0.0

        # Data series
        self._fps_series = QLineSeries()
        self._cpu_series = QLineSeries()
        self._gpu_series = QLineSeries()
        self._mem_series = QLineSeries()
        self._acc_series = QLineSeries()
        self._delay_series = QLineSeries()

        for s in [self._fps_series, self._cpu_series, self._gpu_series,
                   self._mem_series, self._acc_series, self._delay_series]:
            s.setPen(QColor(100, 100, 100))

        self._fps_series.setPen(QColor(FPS_COLOR))
        self._cpu_series.setPen(QColor(CPU_COLOR))
        self._gpu_series.setPen(QColor(GPU_COLOR))
        self._mem_series.setPen(QColor(MEM_COLOR))
        self._acc_series.setPen(QColor(ACC_COLOR))
        self._delay_series.setPen(QColor(DELAY_COLOR))

        # Cards
        self._fps_card = MetricCard("FPS", "", "#00CCFF")
        self._cpu_card = MetricCard("CPU", "%", "#FFB432")
        self._gpu_card = MetricCard("GPU", "%", "#B450FF")
        self._mem_card = MetricCard("Memory", "MB", "#50C878")
        self._acc_card = MetricCard("Accuracy", "%", "#64FF96")
        self._delay_card = MetricCard("Delay", "ms", "#FF7864")

        self._setup_ui()
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(200)  # 5 Hz
        self._update_timer.timeout.connect(self._refresh)
        self._update_timer.start()

    # ── UI Setup ────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Header ──
        header = QLabel("📊 Performance Analytics")
        header.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: bold; padding: 4px 0;")
        layout.addWidget(header)

        # ── Metric Cards Row ──
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)
        for card in [self._fps_card, self._cpu_card, self._gpu_card,
                     self._mem_card, self._acc_card, self._delay_card]:
            cards_layout.addWidget(card)
        layout.addLayout(cards_layout)

        # ── Charts Grid ──
        grid = QGridLayout()
        grid.setSpacing(12)

        # FPS chart
        fps_chart = QChart()
        fps_chart.addSeries(self._fps_series)
        fps_chart.createDefaultAxes()
        fps_chart.axisX().setVisible(False)
        self._fps_axis_y = _create_value_axis("FPS", 0, 90)
        fps_chart.removeAxis(fps_chart.axisY())
        fps_chart.addAxis(self._fps_axis_y, Qt.AlignLeft)
        self._fps_series.attachAxis(self._fps_axis_y)
        grid.addWidget(_make_chart_view(fps_chart, "FPS (target: 30+ / preferred: 60+)"), 0, 0)

        # CPU/GPU chart
        cpu_gpu_chart = QChart()
        cpu_gpu_chart.addSeries(self._cpu_series)
        cpu_gpu_chart.addSeries(self._gpu_series)
        cpu_gpu_chart.createDefaultAxes()
        cpu_gpu_chart.axisX().setVisible(False)
        self._cpu_gpu_axis_y = _create_value_axis("%", 0, 100)
        cpu_gpu_chart.removeAxis(cpu_gpu_chart.axisY())
        cpu_gpu_chart.addAxis(self._cpu_gpu_axis_y, Qt.AlignLeft)
        self._cpu_series.attachAxis(self._cpu_gpu_axis_y)
        self._gpu_series.attachAxis(self._cpu_gpu_axis_y)
        grid.addWidget(_make_chart_view(cpu_gpu_chart, "CPU / GPU Usage"), 0, 1)

        # Memory chart
        mem_chart = QChart()
        mem_chart.addSeries(self._mem_series)
        mem_chart.createDefaultAxes()
        mem_chart.axisX().setVisible(False)
        self._mem_axis_y = _create_value_axis("MB", 0, 2048)
        mem_chart.removeAxis(mem_chart.axisY())
        mem_chart.addAxis(self._mem_axis_y, Qt.AlignLeft)
        self._mem_series.attachAxis(self._mem_axis_y)
        grid.addWidget(_make_chart_view(mem_chart, "Memory Usage"), 1, 0)

        # Accuracy chart
        acc_chart = QChart()
        acc_chart.addSeries(self._acc_series)
        acc_chart.createDefaultAxes()
        acc_chart.axisX().setVisible(False)
        self._acc_axis_y = _create_value_axis("%", 0, 100)
        acc_chart.removeAxis(acc_chart.axisY())
        acc_chart.addAxis(self._acc_axis_y, Qt.AlignLeft)
        self._acc_series.attachAxis(self._acc_axis_y)
        grid.addWidget(_make_chart_view(acc_chart, "Recognition Accuracy"), 1, 1)

        # Delay chart (spans full width)
        delay_chart = QChart()
        delay_chart.addSeries(self._delay_series)
        delay_chart.createDefaultAxes()
        delay_chart.axisX().setVisible(False)
        self._delay_axis_y = _create_value_axis("ms", 0, 500)
        delay_chart.removeAxis(delay_chart.axisY())
        delay_chart.addAxis(self._delay_axis_y, Qt.AlignLeft)
        self._delay_series.attachAxis(self._delay_axis_y)
        grid.addWidget(_make_chart_view(delay_chart, "Translation Delay (target: <150ms)", 160), 2, 0, 1, 2)

        layout.addLayout(grid)
        layout.addStretch()

    # ── Refresh ─────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Update all charts and cards from the analytics engine."""
        snap = self._engine.snapshot()

        # Cards
        self._fps_card.update_value(snap.fps, 0)
        self._cpu_card.update_value(snap.cpu_percent, 1)
        self._gpu_card.update_value(snap.gpu_percent, 1)
        self._mem_card.update_value(snap.memory_mb, 0)
        self._acc_card.update_value(snap.accuracy, 1)
        self._delay_card.update_value(snap.translation_delay_ms, 1)

        # Data points
        self._time_counter += 1

        pt = QPointF(self._time_counter, snap.fps)
        self._fps_series.append(pt)
        self._cpu_series.append(QPointF(self._time_counter, snap.cpu_percent))
        self._gpu_series.append(QPointF(self._time_counter, snap.gpu_percent))
        self._mem_series.append(QPointF(self._time_counter, snap.memory_mb))
        self._acc_series.append(QPointF(self._time_counter, snap.accuracy))
        self._delay_series.append(QPointF(self._time_counter, snap.translation_delay_ms))

        # Trim
        for series in [self._fps_series, self._cpu_series, self._gpu_series,
                       self._mem_series, self._acc_series, self._delay_series]:
            while series.count() > self._max_points:
                series.remove(0)

    def start(self) -> None:
        self._update_timer.start()

    def stop(self) -> None:
        self._update_timer.stop()
