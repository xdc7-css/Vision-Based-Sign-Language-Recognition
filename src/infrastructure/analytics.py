"""AnalyticsEngine — real-time performance monitoring for the recognition pipeline.

Captures:
    - FPS (frames per second)
    - CPU usage (%)
    - GPU usage (%)
    - Memory usage (MB)
    - Recognition accuracy (rolling average)
    - Translation delay (ms)
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalyticsSnapshot:
    """Single-point metrics snapshot."""
    fps: float = 0.0
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    memory_mb: float = 0.0
    accuracy: float = 0.0
    translation_delay_ms: float = 0.0
    total_predictions: int = 0
    correct_predictions: int = 0
    uptime_seconds: float = 0.0


class AnalyticsEngine:
    """Real-time analytics collector with rolling windows for smooth metrics."""

    def __init__(self, window_size: int = 60) -> None:
        self._window_size = window_size
        self._fps_times: deque[float] = deque(maxlen=window_size)
        self._cpu_samples: deque[float] = deque(maxlen=window_size)
        self._gpu_samples: deque[float] = deque(maxlen=window_size)
        self._memory_samples: deque[float] = deque(maxlen=window_size)
        self._delay_samples: deque[float] = deque(maxlen=window_size)
        self._total_predictions: int = 0
        self._correct_predictions: int = 0
        self._start_time = time.perf_counter()
        self._last_frame_time: Optional[float] = None
        self._logger = logging.getLogger(__name__)
        self._process: Optional[object] = None
        self._imported_psutil = False

        try:
            import psutil
            self._process = psutil.Process(os.getpid())
            self._imported_psutil = True
        except ImportError:
            self._logger.warning("psutil not available — CPU/memory metrics disabled")
            self._process = None

    # ── Recording ───────────────────────────────────────────────────

    def record_frame(self, frame_time: Optional[float] = None) -> None:
        """Record a frame for FPS calculation.
        
        Args:
            frame_time: Optional explicit frame time in seconds.
                        If None, the internal timer is used.
        """
        if frame_time is not None:
            self._fps_times.append(frame_time)
            self._last_frame_time = time.perf_counter()
        else:
            now = time.perf_counter()
            if self._last_frame_time is not None:
                self._fps_times.append(now - self._last_frame_time)
            self._last_frame_time = now

    def record_prediction(self, correct: bool, delay_ms: float) -> None:
        """Record a recognition prediction result."""
        self._total_predictions += 1
        if correct:
            self._correct_predictions += 1
        self._delay_samples.append(delay_ms)

    def record_system_metrics(self) -> None:
        """Sample CPU, GPU, and memory usage."""
        if self._process is not None and self._imported_psutil:
            try:
                cpu = self._process.cpu_percent(interval=None)
                self._cpu_samples.append(cpu)
            except Exception:
                pass
            try:
                mem = self._process.memory_info().rss / (1024 * 1024)
                self._memory_samples.append(mem)
            except Exception:
                pass

        # GPU metrics (NVIDIA only)
        gpu = self._sample_gpu()
        if gpu >= 0:
            self._gpu_samples.append(gpu)

    # ── Snapshot ────────────────────────────────────────────────────

    def snapshot(self) -> AnalyticsSnapshot:
        """Get current metrics snapshot."""
        fps = self._calculate_fps()
        delay = self._average(self._delay_samples)
        acc = self._calculate_accuracy()

        return AnalyticsSnapshot(
            fps=fps,
            cpu_percent=self._average(self._cpu_samples),
            gpu_percent=self._average(self._gpu_samples),
            memory_mb=self._average(self._memory_samples),
            accuracy=acc,
            translation_delay_ms=delay,
            total_predictions=self._total_predictions,
            correct_predictions=self._correct_predictions,
            uptime_seconds=time.perf_counter() - self._start_time,
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self._fps_times.clear()
        self._cpu_samples.clear()
        self._gpu_samples.clear()
        self._memory_samples.clear()
        self._delay_samples.clear()
        self._total_predictions = 0
        self._correct_predictions = 0
        self._start_time = time.perf_counter()
        self._last_frame_time = None

    # ── Private ─────────────────────────────────────────────────────

    def _calculate_fps(self) -> float:
        if len(self._fps_times) == 0:
            return 0.0
        avg_dt = sum(self._fps_times) / len(self._fps_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def _calculate_accuracy(self) -> float:
        if self._total_predictions == 0:
            return 0.0
        return (self._correct_predictions / self._total_predictions) * 100.0

    @staticmethod
    def _average(samples: deque[float]) -> float:
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    @staticmethod
    def _sample_gpu() -> float:
        """Sample GPU utilization via nvidia-smi or pynvml."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return -1.0
