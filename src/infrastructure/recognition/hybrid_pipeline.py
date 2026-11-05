"""HybridRecognitionPipeline — priority-based fusion of word, dynamic, and alphabet recognition.

Priority:
    1. Word Recognition  (if confidence > word_threshold)
    2. Dynamic Gesture   (if confidence > dynamic_threshold)
    3. Alphabet Letter   (if confidence > alphabet_threshold)
    4. Reject            (if all below thresholds)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.domain.hand import Hand
from src.domain.motion import HandMotionTrack
from src.domain.recognition import GestureCategory, RecognitionResult
from src.domain.text import TextCommand
from src.infrastructure.recognition.dynamic_recognizer import DynamicSignRecognizer
from src.infrastructure.recognition.static_recognizer import StaticSignRecognizer
from src.infrastructure.recognition.word_recognizer import WordRecognizer


@dataclass
class HybridConfig:
    word_threshold: float = 0.65
    dynamic_threshold: float = 0.70
    alphabet_threshold: float = 0.90
    cooldown_frames: int = 20
    temporal_smoothing: int = 3
    enable_word_recognition: bool = True
    enable_dynamic_recognition: bool = True
    enable_alphabet_recognition: bool = True


@dataclass
class HybridResult:
    """Result from the hybrid pipeline with full decision trace."""
    final_result: RecognitionResult
    text_command: Optional[TextCommand] = None
    word_result: Optional[RecognitionResult] = None
    dynamic_result: Optional[RecognitionResult] = None
    alphabet_result: Optional[RecognitionResult] = None
    decision_path: str = "none"
    processing_time_ms: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.final_result.is_valid

    @property
    def arabic_text(self) -> str:
        return self.final_result.arabic_text


class HybridRecognitionPipeline:
    """Orchestrates word, dynamic, and alphabet recognition with priority-based fusion.

    Flowchart:
        Input (hand / track)
            │
            ├── Word Recognizer
            │   └── confidence > word_threshold? ──► OUTPUT word
            │
            ├── Dynamic Recognizer
            │   └── confidence > dynamic_threshold? ──► OUTPUT gesture
            │
            └── Alphabet Recognizer (letter-by-letter)
                └── confidence > alphabet_threshold? ──► OUTPUT letter
                            │
                        REJECT (all below thresholds)
    """

    def __init__(
        self,
        word_recognizer: Optional[WordRecognizer] = None,
        static_recognizer: Optional[StaticSignRecognizer] = None,
        dynamic_recognizer: Optional[DynamicSignRecognizer] = None,
        config: Optional[HybridConfig] = None,
    ) -> None:
        self._word_recognizer = word_recognizer
        self._static_recognizer = static_recognizer
        self._dynamic_recognizer = dynamic_recognizer
        self._config = config or HybridConfig()
        self._logger = logging.getLogger(__name__)
        self._cooldown_counter: int = 0
        self._last_gesture_id: str = ""
        self._prediction_buffer: list[str] = []
        self._alphabet_letter_buffer: list[str] = []

    # ── Public API ──────────────────────────────────────────────────

    def process(
        self,
        hand: Optional[Hand] = None,
        track: Optional[HandMotionTrack] = None,
    ) -> HybridResult:
        """Run the full hybrid pipeline. Returns the best recognition result."""
        start = time.perf_counter()

        word_result: Optional[RecognitionResult] = None
        dynamic_result: Optional[RecognitionResult] = None
        alphabet_result: Optional[RecognitionResult] = None
        decision_path = "none"

        # Priority 1: Word recognition
        if self._config.enable_word_recognition and hand is not None and self._word_recognizer is not None:
            word_result = self._word_recognizer.recognize_static(hand)
            if word_result.is_valid and word_result.confidence >= self._config.word_threshold:
                decision_path = "word"
                elapsed = (time.perf_counter() - start) * 1000
                result = self._build_result(word_result, decision_path, elapsed)
                self._apply_cooldown(result.final_result)
                return result

        # Priority 2: Dynamic gesture recognition
        if self._config.enable_dynamic_recognition and track is not None and self._dynamic_recognizer is not None:
            if track.is_motion_significant and len(track.states) > 5:
                dynamic_result = self._dynamic_recognizer.predict_from_track(track, self._config.dynamic_threshold)
                if dynamic_result is not None and dynamic_result.is_valid and dynamic_result.confidence >= self._config.dynamic_threshold:
                    decision_path = "dynamic"
                    elapsed = (time.perf_counter() - start) * 1000
                    result = self._build_result(dynamic_result, decision_path, elapsed)
                    self._apply_cooldown(result.final_result)
                    return result

        # Priority 3: Alphabet letter recognition
        if self._config.enable_alphabet_recognition and hand is not None and self._static_recognizer is not None:
            if self._static_recognizer.is_loaded:
                letter_result = self._static_recognizer.predict(hand, self._config.alphabet_threshold)
                if letter_result.is_valid and letter_result.confidence >= self._config.alphabet_threshold:
                    alphabet_result = letter_result
                    decision_path = "alphabet"
                    elapsed = (time.perf_counter() - start) * 1000
                    result = self._build_result(letter_result, decision_path, elapsed)
                    self._apply_cooldown(result.final_result)
                    return result

        # All below threshold → reject
        elapsed = (time.perf_counter() - start) * 1000
        reject_result = RecognitionResult(
            gesture_id="", arabic_text="",
            category=GestureCategory.STATIC_WORD,
            confidence=0.0, is_rejected=True,
            rejection_reason="All recognition levels below confidence threshold",
            processing_time_ms=elapsed,
        )
        return HybridResult(
            final_result=reject_result,
            word_result=word_result,
            dynamic_result=dynamic_result,
            alphabet_result=alphabet_result,
            decision_path="rejected",
            processing_time_ms=elapsed,
        )

    def process_to_command(self, hand: Optional[Hand] = None, track: Optional[HandMotionTrack] = None) -> tuple[HybridResult, Optional[TextCommand]]:
        """Run hybrid pipeline and convert the result to a TextCommand."""
        hybrid_result = self.process(hand, track)

        if hybrid_result.is_valid:
            if hybrid_result.decision_path == "alphabet":
                # Alphabet → letter command
                cmd = TextCommand.letter(hybrid_result.arabic_text, hybrid_result.final_result.confidence)
            else:
                # Word/dynamic → word command
                cmd = TextCommand.word(hybrid_result.arabic_text, hybrid_result.final_result.confidence)
        else:
            cmd = None

        hybrid_result.text_command = cmd
        return hybrid_result, cmd

    @property
    def config(self) -> HybridConfig:
        return self._config

    def reset_cooldown(self) -> None:
        self._cooldown_counter = 0
        self._last_gesture_id = ""
        self._prediction_buffer.clear()

    # ── Private ─────────────────────────────────────────────────────

    def _build_result(self, recognition: RecognitionResult, path: str, elapsed_ms: float) -> HybridResult:
        recognition.processing_time_ms = elapsed_ms
        return HybridResult(
            final_result=recognition,
            word_result=recognition if path == "word" else None,
            dynamic_result=recognition if path == "dynamic" else None,
            alphabet_result=recognition if path == "alphabet" else None,
            decision_path=path,
            processing_time_ms=elapsed_ms,
        )

    def _apply_cooldown(self, result: RecognitionResult) -> None:
        """Prevent duplicate detections within the cooldown window."""
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            if result.gesture_id == self._last_gesture_id:
                result.is_rejected = True
                result.rejection_reason = "Cooldown (duplicate suppression)"
                return

        if result.is_valid:
            self._last_gesture_id = result.gesture_id
            self._cooldown_counter = self._config.cooldown_frames
