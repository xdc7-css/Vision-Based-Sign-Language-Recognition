"""WordRecognizer — word-level sign language recognition with confidence rejection.

Combines static and dynamic recognition to detect complete words.
Supports vocabulary-based matching with configurable confidence thresholds
and rejection of uncertain predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


from src.domain.hand import Hand
from src.domain.motion import HandMotionTrack
from src.domain.recognition import (
    GestureCategory,
    RecognitionResult,
    RecognitionSession,
    VocabularyEntry,
    WordVocabulary,
    DEFAULT_VOCABULARY,
)
from src.infrastructure.recognition.dynamic_recognizer import DynamicSignRecognizer
from src.infrastructure.recognition.static_recognizer import StaticSignRecognizer


@dataclass
class WordRecognizerConfig:
    min_confidence_static: float = 0.50
    min_confidence_dynamic: float = 0.50
    min_confidence_alphabet: float = 0.50
    rejection_threshold: float = 0.5
    temporal_smoothing_window: int = 5
    cooldown_frames: int = 30


class WordRecognizer:
    """Word-level sign language recognition engine.

    Architecture:
    - Static whole-word gestures → StaticSignRecognizer
    - Dynamic motion gestures → DynamicSignRecognizer
    - Finger-spelled words → StaticSignRecognizer (letter-by-letter)
    - Decision fusion → priority-based with confidence rejection

    The vocabulary is extensible — add new words by registering
    VocabularyEntry objects.
    """

    def __init__(
        self,
        config: Optional[WordRecognizerConfig] = None,
        vocabulary: Optional[WordVocabulary] = None,
        static_recognizer: Optional[StaticSignRecognizer] = None,
        dynamic_recognizer: Optional[DynamicSignRecognizer] = None,
    ) -> None:
        self._config = config or WordRecognizerConfig()
        self._vocabulary = vocabulary or WordVocabulary()
        self._static_recognizer = static_recognizer
        self._dynamic_recognizer = dynamic_recognizer
        self._session = RecognitionSession()
        self._predictions_buffer: list[RecognitionResult] = []
        self._cooldown_counter: int = 0
        self._last_gesture_id: str = ""

        # Load default vocabulary
        if len(self._vocabulary) == 0:
            self._vocabulary.register_many(DEFAULT_VOCABULARY)

    @property
    def is_loaded(self) -> bool:
        return (
            (self._static_recognizer is not None and self._static_recognizer.is_loaded)
            or (self._dynamic_recognizer is not None and self._dynamic_recognizer.is_loaded)
        )

    # ── Public API ──────────────────────────────────────────────────

    def register_vocabulary(self, entries: list[VocabularyEntry]) -> None:
        """Register new word entries into the vocabulary."""
        self._vocabulary.register_many(entries)

    def load_vocabulary_from_json(self, path: str | Path) -> int:
        """Load vocabulary from a JSON file.

        Expected format:
            { "gestures": { "gesture_id": { "arabic": "...", "type": "...", ... } } }
        """
        path = Path(path)
        if not path.exists():
            return 0

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for gesture_id, info in data.get("gestures", {}).items():
            category_map = {
                "static": GestureCategory.STATIC_WORD,
                "dynamic": GestureCategory.DYNAMIC_GESTURE,
                "alphabet": GestureCategory.ALPHABET,
            }
            cat = category_map.get(info.get("type", "static"), GestureCategory.STATIC_WORD)
            entry = VocabularyEntry(
                gesture_id=gesture_id,
                arabic_text=info.get("arabic", gesture_id),
                category=cat,
                description=info.get("description", ""),
                confidence_threshold=info.get("threshold", 0.7),
            )
            self._vocabulary.register(entry)
            count += 1

        return count

    def recognize_static(self, hand: Hand) -> RecognitionResult:
        """Recognize a static whole-word gesture from a single hand frame.

        Uses the static recognizer and looks up the Arabic text from vocabulary.
        """
        if self._static_recognizer is None or not self._static_recognizer.is_loaded:
            return RecognitionResult(
                gesture_id="", arabic_text="",
                category=GestureCategory.STATIC_WORD,
                confidence=0.0, is_rejected=True,
                rejection_reason="Static recognizer not available",
            )

        result = self._static_recognizer.predict(hand, self._config.min_confidence_static)

        # Look up Arabic text from vocabulary
        if result.is_valid:
            entry = self._vocabulary.get(result.gesture_id)
            if not entry and result.gesture_id.startswith("static_"):
                entry = self._vocabulary.get(result.gesture_id.removeprefix("static_"))
            if entry:
                result.arabic_text = entry.arabic_text
                result.category = entry.category
            elif result.arabic_text:
                # Retain prediction from StaticSignRecognizer
                result.category = GestureCategory.STATIC_WORD
            else:
                result.is_rejected = True
                result.rejection_reason = f"Unknown gesture: {result.gesture_id}"

        return self._apply_cooldown(result)

    def recognize_dynamic(self, track: HandMotionTrack) -> RecognitionResult:
        """Recognize a dynamic gesture from a motion track.

        Uses the dynamic recognizer on the landmark sequence.
        """
        if self._dynamic_recognizer is None or not self._dynamic_recognizer.is_loaded:
            return RecognitionResult(
                gesture_id="", arabic_text="",
                category=GestureCategory.DYNAMIC_GESTURE,
                confidence=0.0, is_rejected=True,
                rejection_reason="Dynamic recognizer not available",
            )

        result = self._dynamic_recognizer.predict_from_track(track, self._config.min_confidence_dynamic)

        if result.is_valid:
            entry = self._vocabulary.get(result.gesture_id)
            if entry:
                result.arabic_text = entry.arabic_text
                result.category = entry.category

        return self._apply_cooldown(result)

    def recognize_word(
        self,
        hand: Optional[Hand] = None,
        track: Optional[HandMotionTrack] = None,
    ) -> RecognitionResult:
        """Fused word recognition — tries static first, then dynamic.

        Priority: static whole-word > dynamic > alphabet

        If both static and dynamic recognize the same word with confidence
        above threshold, the result is boosted.
        """
        static_result: Optional[RecognitionResult] = None
        dynamic_result: Optional[RecognitionResult] = None

        if hand is not None and self._static_recognizer and self._static_recognizer.is_loaded:
            static_result = self.recognize_static(hand)

        if track is not None and self._dynamic_recognizer and self._dynamic_recognizer.is_loaded:
            dynamic_result = self.recognize_dynamic(track)

        # Decision fusion
        result = self._fuse_decisions(static_result, dynamic_result)

        if result.is_valid:
            self._session.results.append(result)

        return result

    def recognize(self, hand: Optional[Hand] = None, track: Optional[HandMotionTrack] = None) -> RecognitionResult:
        """Unified recognition — prefers static for speed, falls back to dynamic."""
        return self.recognize_word(hand, track)

    def reject(self, result: RecognitionResult, reason: str = "Uncertain prediction") -> RecognitionResult:
        """Explicitly reject a recognition result."""
        result.is_rejected = True
        result.rejection_reason = reason
        result.confidence = 0.0
        return result

    # ── Session management ──────────────────────────────────────────

    def reset_session(self) -> None:
        self._session = RecognitionSession()
        self._predictions_buffer.clear()
        self._cooldown_counter = 0
        self._last_gesture_id = ""

    @property
    def session(self) -> RecognitionSession:
        return self._session

    @property
    def current_text(self) -> str:
        return self._session.text_output

    @property
    def vocabulary(self) -> WordVocabulary:
        return self._vocabulary

    @property
    def config(self) -> WordRecognizerConfig:
        return self._config

    # ── Private ─────────────────────────────────────────────────────

    def _fuse_decisions(
        self,
        static: Optional[RecognitionResult],
        dynamic: Optional[RecognitionResult],
    ) -> RecognitionResult:
        """Priority-based decision fusion with confidence boosting.

        Strategy:
        1. If both agree on the same gesture → boost confidence.
        2. If static is confident → use static (faster, more reliable).
        3. If static is unavailable / low confidence → use dynamic.
        4. If neither is confident → reject.
        """
        if static is None and dynamic is None:
            return RecognitionResult(
                gesture_id="", arabic_text="",
                category=GestureCategory.STATIC_WORD,
                confidence=0.0, is_rejected=True,
                rejection_reason="No input available",
            )

        if static is None and dynamic is not None:
            return dynamic

        if dynamic is None and static is not None:
            return static

        if static.is_valid and dynamic.is_valid and static.gesture_id == dynamic.gesture_id:
            # Agreement → boost confidence
            boosted_confidence = min(1.0, static.confidence + dynamic.confidence * 0.3)
            return RecognitionResult(
                gesture_id=static.gesture_id,
                arabic_text=static.arabic_text,
                category=static.category,
                confidence=boosted_confidence,
            )

        # Preference: static over dynamic
        if static and static.is_valid:
            return static
        if dynamic and dynamic.is_valid:
            return dynamic

        # Both low confidence → pick the higher
        candidates = [r for r in (static, dynamic) if r is not None]
        if candidates:
            best = max(candidates, key=lambda r: r.confidence)
            if best.confidence > self._config.rejection_threshold:
                return best

        return RecognitionResult(
            gesture_id="", arabic_text="",
            category=GestureCategory.STATIC_WORD,
            confidence=0.0, is_rejected=True,
            rejection_reason="Confidence below rejection threshold",
        )

    def _apply_cooldown(self, result: RecognitionResult) -> RecognitionResult:
        """Prevent duplicate detections within a cooldown window."""
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            if result.gesture_id == self._last_gesture_id:
                result.is_rejected = True
                result.rejection_reason = "Cooldown (duplicate suppression)"
                return result

        if result.is_valid:
            self._last_gesture_id = result.gesture_id
            self._cooldown_counter = self._config.cooldown_frames

        return result
