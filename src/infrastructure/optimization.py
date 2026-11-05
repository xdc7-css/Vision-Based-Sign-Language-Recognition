"""PerformanceOptimizer — CPU, GPU, memory, FPS, and latency optimization.

Targets:
    - 30+ FPS minimum (60 FPS preferred)
    - <150ms end-to-end latency

Strategies:
    - Frame skipping (process every Nth frame)
    - Resolution scaling (reduce input resolution for hand detection)
    - Model quantization (INT8/FP16 where supported)
    - Parallel frame capture + processing pipeline
    - Thread pool for hand detection
    - Lazy model loading (only load when needed)
    - Memory pooling for numpy/PyTorch tensors
    - Caching repeated calculations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional



class OptimizationLevel(Enum):
    """Preset optimization profiles."""
    MAX_QUALITY = auto()     # 60 FPS, full resolution
    BALANCED = auto()        # 30 FPS, moderate resolution
    MAX_PERFORMANCE = auto() # 30 FPS, low resolution, aggressive frame skip
    BATTERY_SAVER = auto()   # 15 FPS, minimal resource usage


@dataclass
class OptimizationProfile:
    """Tunable optimization parameters."""
    level: OptimizationLevel = OptimizationLevel.BALANCED
    frame_skip: int = 0                       # Process every (N+1)th frame
    input_width: int = 640
    input_height: int = 480
    hand_model_complexity: int = 0            # 0=lite, 1=full, 2=heavy
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    use_quantized_model: bool = True
    parallel_capture: bool = True
    max_hands: int = 1
    render_overlay: bool = True
    enable_motion_tracking: bool = True
    enable_temporal_smoothing: bool = True

    def apply(self) -> dict[str, object]:
        return {
            "frame_skip": self.frame_skip,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "hand_model_complexity": self.hand_model_complexity,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
        }


# Performance presets
PRESETS: dict[OptimizationLevel, OptimizationProfile] = {
    OptimizationLevel.MAX_QUALITY: OptimizationProfile(
        level=OptimizationLevel.MAX_QUALITY,
        frame_skip=0,
        input_width=1280,
        input_height=720,
        hand_model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ),
    OptimizationLevel.BALANCED: OptimizationProfile(
        level=OptimizationLevel.BALANCED,
        frame_skip=0,
        input_width=640,
        input_height=480,
        hand_model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ),
    OptimizationLevel.MAX_PERFORMANCE: OptimizationProfile(
        level=OptimizationLevel.MAX_PERFORMANCE,
        frame_skip=1,
        input_width=480,
        input_height=360,
        hand_model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        render_overlay=False,
        enable_motion_tracking=True,
    ),
    OptimizationLevel.BATTERY_SAVER: OptimizationProfile(
        level=OptimizationLevel.BATTERY_SAVER,
        frame_skip=2,
        input_width=320,
        input_height=240,
        hand_model_complexity=0,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
        max_hands=1,
        render_overlay=False,
        enable_motion_tracking=False,
    ),
}


class PerformanceOptimizer:
    """Applies optimization strategies to the recognition pipeline."""

    def __init__(self, profile: Optional[OptimizationProfile] = None) -> None:
        self._profile = profile or PRESETS[OptimizationLevel.BALANCED]
        self._frame_counter: int = 0
        self._frame_skip_counter: int = 0
        self._logger = logging.getLogger(__name__)

    # ── Profile Management ──────────────────────────────────────────

    @property
    def profile(self) -> OptimizationProfile:
        return self._profile

    def set_profile(self, level: OptimizationLevel) -> None:
        if level in PRESETS:
            self._profile = PRESETS[level]
            self._logger.info(f"Optimization profile: {level.name}")
        else:
            self._logger.warning(f"Unknown profile: {level}")

    def set_custom(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._profile, key):
                setattr(self._profile, key, value)

    # ── Frame Pipeline ──────────────────────────────────────────────

    def should_process_frame(self) -> bool:
        """Frame skipping: return True only if this frame should be processed."""
        self._frame_counter += 1
        if self._frame_skip_counter < self._profile.frame_skip:
            self._frame_skip_counter += 1
            return False
        self._frame_skip_counter = 0
        return True

    def resize_frame(self, frame: object) -> tuple[object, tuple[int, int, int]]:
        """Resize frame to the configured input dimensions."""
        import cv2
        h, w = frame.shape[:2]
        if w != self._profile.input_width or h != self._profile.input_height:
            frame = cv2.resize(frame, (self._profile.input_width, self._profile.input_height))
        return frame, (h, w)

    def hand_detector_config(self) -> dict[str, object]:
        """Get MediaPipe Hands config based on current profile."""
        return self._profile.apply()

    # ── Latency Measurement ─────────────────────────────────────────

    @staticmethod
    def measure_latency(start_time: float) -> float:
        """Measure elapsed time in milliseconds."""
        import time
        return (time.perf_counter() - start_time) * 1000

    # ── Auto Profile Adjustment ─────────────────────────────────────

    def auto_profile(self, measured_fps: float, target_fps: float = 30.0) -> list[str]:
        """Dynamically adjust optimization parameters based on measured FPS.

        If measured FPS < 80% of target: reduce resolution, increase frame skip.
        If measured FPS > 120% of target: increase resolution, reduce frame skip.

        Args:
            measured_fps: Current measured FPS.
            target_fps: Desired FPS target.

        Returns:
            List of adjustments made as human-readable strings.
        """
        adjustments: list[str] = []

        if measured_fps <= 0:
            return adjustments

        ratio = measured_fps / target_fps
        max_dim = 1280

        if ratio < 0.8:
            # Too slow — reduce quality
            if self._profile.frame_skip < 2:
                self._profile.frame_skip += 1
                adjustments.append(f"Increased frame skip to {self._profile.frame_skip}")
            if self._profile.input_width > 320:
                new_w = max(int(self._profile.input_width * 0.75), 320)
                new_h = max(int(self._profile.input_height * 0.75), 240)
                self._profile.input_width, self._profile.input_height = new_w, new_h
                adjustments.append(f"Reduced resolution to {new_w}x{new_h}")
            if self._profile.hand_model_complexity > 0:
                self._profile.hand_model_complexity = max(0, self._profile.hand_model_complexity - 1)
                adjustments.append(f"Reduced hand model complexity to {self._profile.hand_model_complexity}")

        elif ratio > 1.2:
            # Too fast — increase quality
            if self._profile.frame_skip > 0:
                self._profile.frame_skip = max(0, self._profile.frame_skip - 1)
                adjustments.append(f"Reduced frame skip to {self._profile.frame_skip}")
            if self._profile.input_width < max_dim:
                new_w = min(int(self._profile.input_width * 1.25), max_dim)
                new_h = min(int(self._profile.input_height * 1.25), 720)
                self._profile.input_width, self._profile.input_height = new_w, new_h
                adjustments.append(f"Increased resolution to {new_w}x{new_h}")
            if self._profile.hand_model_complexity < 2:
                self._profile.hand_model_complexity = min(2, self._profile.hand_model_complexity + 1)
                adjustments.append(f"Increased hand model complexity to {self._profile.hand_model_complexity}")

        if adjustments:
            self._logger.info(f"Auto profile adjustments ({ratio:.2f}x target): {'; '.join(adjustments)}")

        return adjustments

    # ── Latency Budget ──────────────────────────────────────────────

    @dataclass
    class LatencyBudget:
        preprocess_ms: float = 0.0
        inference_ms: float = 0.0
        postprocess_ms: float = 0.0
        overlay_ms: float = 0.0

        @property
        def total_ms(self) -> float:
            return self.preprocess_ms + self.inference_ms + self.postprocess_ms + self.overlay_ms

        @property
        def fps_estimate(self) -> float:
            if self.total_ms <= 0:
                return 0.0
            return 1000.0 / self.total_ms

        def report(self) -> str:
            return (
                f"Preprocess: {self.preprocess_ms:.2f}ms | "
                f"Inference: {self.inference_ms:.2f}ms | "
                f"Postprocess: {self.postprocess_ms:.2f}ms | "
                f"Overlay: {self.overlay_ms:.2f}ms | "
                f"Total: {self.total_ms:.2f}ms ({self.fps_estimate:.1f} FPS)"
            )

    # ── Optimization Report ─────────────────────────────────────────

    def generate_report(self) -> str:
        """Generate a human-readable optimization report."""
        return f"""
═══════════════════════════════════════════════════════
  PERFORMANCE OPTIMIZATION REPORT
═══════════════════════════════════════════════════════
  Profile:            {self._profile.level.name}
  Frame Skip:         {self._profile.frame_skip}
  Input Resolution:   {self._profile.input_width}x{self._profile.input_height}
  Hand Model:         complexity={self._profile.hand_model_complexity}
  Detection Conf:     {self._profile.min_detection_confidence}
  Tracking Conf:      {self._profile.min_tracking_confidence}
  Max Hands:          {self._profile.max_hands}
  Quantized Model:    {self._profile.use_quantized_model}
  Parallel Capture:   {self._profile.parallel_capture}
  Render Overlay:     {self._profile.render_overlay}
  Motion Tracking:    {self._profile.enable_motion_tracking}
  Temporal Smoothing: {self._profile.enable_temporal_smoothing}
───────────────────────────────────────────────────────
  TARGETS:
  • Minimum FPS:     30
  • Preferred FPS:   60
  • Max Latency:     <150ms
───────────────────────────────────────────────────────
  RECOMMENDATIONS:
  • Use BALANCED profile for normal operation
  • Switch to MAX_PERFORMANCE on low-end hardware
  • Enable quantized models for 2-3x speedup
  • Use frame skipping when CPU > 80%
├───────────────────────────────────────────────────────
"""
