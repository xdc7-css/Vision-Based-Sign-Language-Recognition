"""Tests for AnalyticsEngine — FPS, CPU, GPU, memory, accuracy, delay tracking."""

import pytest
import time
from unittest.mock import Mock, patch

from src.infrastructure.analytics import AnalyticsEngine, AnalyticsSnapshot


class TestAnalyticsEngine:

    def test_initial_snapshot(self):
        engine = AnalyticsEngine()
        snap = engine.snapshot()

        assert snap.fps == 0.0
        assert snap.cpu_percent == 0.0
        assert snap.memory_mb == 0.0
        assert snap.accuracy == 0.0
        assert snap.translation_delay_ms == 0.0
        assert snap.total_predictions == 0

    def test_record_frame(self):
        engine = AnalyticsEngine()
        engine.record_frame()
        time.sleep(0.05)
        engine.record_frame()

        snap = engine.snapshot()
        assert snap.fps > 0

    def test_record_prediction_correct(self):
        engine = AnalyticsEngine()
        engine.record_prediction(correct=True, delay_ms=50)
        engine.record_prediction(correct=True, delay_ms=60)
        engine.record_prediction(correct=False, delay_ms=70)

        snap = engine.snapshot()
        assert snap.total_predictions == 3
        assert snap.correct_predictions == 2
        assert 60.0 < snap.accuracy < 70.0
        assert 50 <= snap.translation_delay_ms <= 70

    def test_accuracy_percentage(self):
        engine = AnalyticsEngine()
        for _ in range(10):
            engine.record_prediction(correct=True, delay_ms=50)
        for _ in range(10):
            engine.record_prediction(correct=False, delay_ms=50)

        snap = engine.snapshot()
        assert snap.accuracy == 50.0

    def test_delay_averaging(self):
        engine = AnalyticsEngine(window_size=10)
        delays = [100, 200, 300]
        for d in delays:
            engine.record_prediction(correct=True, delay_ms=d)

        snap = engine.snapshot()
        assert snap.translation_delay_ms == 200.0  # average of 100, 200, 300

    def test_reset(self):
        engine = AnalyticsEngine()
        engine.record_frame()
        engine.record_prediction(correct=True, delay_ms=50)
        engine.reset()

        snap = engine.snapshot()
        assert snap.total_predictions == 0
        assert snap.correct_predictions == 0

    def test_system_metrics_no_crash(self):
        engine = AnalyticsEngine()
        # Should not crash even without psutil
        engine.record_system_metrics()
        snap = engine.snapshot()
        assert snap.cpu_percent >= 0

    def test_gpu_sampling_fallback(self):
        from src.infrastructure.analytics import AnalyticsEngine
        value = AnalyticsEngine._sample_gpu()
        assert value == -1.0 or value >= 0  # Either unavailable or valid reading

    def test_window_bounds(self):
        engine = AnalyticsEngine(window_size=3)
        # Add more than window
        for _ in range(10):
            engine.record_prediction(correct=True, delay_ms=100)

        assert len(engine._delay_samples) == 3  # maxlen = 3

    def test_uptime(self):
        engine = AnalyticsEngine()
        snap = engine.snapshot()
        assert snap.uptime_seconds >= 0

    def test_fps_zero_with_single_frame(self):
        engine = AnalyticsEngine()
        engine.record_frame()
        snap = engine.snapshot()
        assert snap.fps == 0.0  # need at least 2 frames

    def test_fps_with_multiple_frames(self):
        engine = AnalyticsEngine()
        for _ in range(10):
            engine.record_frame()
            time.sleep(0.01)

        snap = engine.snapshot()
        assert snap.fps > 0

    def test_snapshot_is_dataclass(self):
        engine = AnalyticsEngine()
        snap = engine.snapshot()
        assert isinstance(snap, AnalyticsSnapshot)
        assert snap.total_predictions == 0
        assert snap.correct_predictions == 0
