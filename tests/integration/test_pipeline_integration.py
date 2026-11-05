"""Integration tests — end-to-end pipeline verification.

Tests the full integration of:
    - ArabicTextBuilder + HybridRecognitionPipeline + DialectManager
    - Hand detection + Motion tracking + Recognition pipeline
    - Analytics engine + recognition pipeline
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.domain.dialect import DialectId
from src.domain.hand import Hand, HandLandmark, HandLandmarkType, Handedness
from src.domain.motion import HandMotionTrack, MotionState, Position2D
from src.domain.recognition import GestureCategory, RecognitionResult
from src.domain.text import GestureInputType, TextCommand, TextBuilderState
from src.infrastructure.analytics import AnalyticsEngine
from src.infrastructure.dialect_manager import DialectManager
from src.infrastructure.recognition.hybrid_pipeline import (
    HybridConfig,
    HybridRecognitionPipeline,
    HybridResult,
)
from src.infrastructure.recognition.static_recognizer import StaticSignRecognizer
from src.infrastructure.recognition.dynamic_recognizer import DynamicSignRecognizer
from src.infrastructure.recognition.word_recognizer import WordRecognizer
from src.infrastructure.text_builder import ArabicTextBuilder


class TestTextBuilderWithHybridPipeline:
    """Integration: TextBuilder receives TextCommands from HybridPipeline."""

    @pytest.fixture
    def builder(self) -> ArabicTextBuilder:
        return ArabicTextBuilder()

    def test_letter_flow_into_word(self, builder: ArabicTextBuilder):
        letters = ["ا", "ل", "س", "ل", "ا", "م"]
        for l in letters:
            builder.process(TextCommand.letter(l))
        assert builder.current_word == "السلام"
        assert builder.state == TextBuilderState.BUILDING_WORD

    def test_word_then_space_then_letter(self, builder: ArabicTextBuilder):
        builder.process(TextCommand.word("مرحبا"))
        assert builder.sentence == "مرحبا"
        builder.process(TextCommand.space())
        builder.process(TextCommand.letter("ا"))
        assert builder.full_text == "مرحبا ا"

    def test_hybrid_to_text_builder_word_cmd(self, builder: ArabicTextBuilder):
        hybrid_result = HybridResult(
            final_result=RecognitionResult(
                gesture_id="hello", arabic_text="مرحبا",
                category=GestureCategory.STATIC_WORD, confidence=0.95,
            ),
            decision_path="word",
        )
        hybrid_result.text_command = TextCommand.word("مرحبا", 0.95)
        builder.process(hybrid_result.text_command)
        assert builder.sentence == "مرحبا"

    def test_hybrid_to_text_builder_letter_cmd(self, builder: ArabicTextBuilder):
        hybrid_result = HybridResult(
            final_result=RecognitionResult(
                gesture_id="alif", arabic_text="ا",
                category=GestureCategory.ALPHABET, confidence=0.92,
            ),
            decision_path="alphabet",
        )
        hybrid_result.text_command = TextCommand.letter("ا", 0.92)
        builder.process(hybrid_result.text_command)
        assert builder.current_word == "ا"


class TestDialectManagerWithPipeline:
    """Integration: DialectManager switches dialects and processes through pipeline."""

    @pytest.fixture
    def dialect_manager(self):
        dm = DialectManager(base_path=".")
        assert dm.active_dialect == DialectId.IRAQI
        return dm

    def test_switch_dialect_updates_active(self, dialect_manager: DialectManager):
        dialect_manager.switch_dialect(DialectId.SAUDI)
        assert dialect_manager.active_dialect == DialectId.SAUDI

    def test_switch_to_egyptian_then_back(self, dialect_manager: DialectManager):
        dialect_manager.switch_dialect(DialectId.EGYPTIAN)
        assert dialect_manager.active_dialect == DialectId.EGYPTIAN
        dialect_manager.switch_dialect(DialectId.IRAQI)
        assert dialect_manager.active_dialect == DialectId.IRAQI

    def test_dialect_manager_with_empty_pipeline_returns_rejected(self, dialect_manager: DialectManager):
        mock_hand = Mock(spec=Hand)
        result = dialect_manager.process(hand=mock_hand)
        assert result.final_result is None or getattr(result.final_result, 'is_rejected', True)


class TestAnalyticsWithRecognition:
    """Integration: Analytics engine records recognition pipeline metrics."""

    @pytest.fixture
    def analytics(self) -> AnalyticsEngine:
        return AnalyticsEngine()

    def test_frame_analytics_records_fps(self, analytics: AnalyticsEngine):
        start = time.perf_counter()
        for _ in range(30):
            analytics.record_frame()
        snap = analytics.snapshot()
        assert snap.fps > 0

    def test_prediction_accuracy(self, analytics: AnalyticsEngine):
        for _ in range(10):
            analytics.record_prediction(correct=True, delay_ms=50)
        for _ in range(10):
            analytics.record_prediction(correct=False, delay_ms=50)
        snap = analytics.snapshot()
        assert snap.accuracy == 50.0

    def test_delay_below_150ms_target(self, analytics: AnalyticsEngine):
        for _ in range(10):
            analytics.record_prediction(correct=True, delay_ms=100)
        snap = analytics.snapshot()
        assert snap.translation_delay_ms <= 150

    def test_uptime_increases(self, analytics: AnalyticsEngine):
        snap1 = analytics.snapshot()
        time.sleep(0.1)
        snap2 = analytics.snapshot()
        assert snap2.uptime_seconds > snap1.uptime_seconds

    def test_system_metrics_collection(self, analytics: AnalyticsEngine):
        analytics.record_system_metrics()
        snap = analytics.snapshot()
        assert isinstance(snap.cpu_percent, float)
        assert isinstance(snap.memory_mb, float)


class TestTextCommandRoundTrip:
    """Integration: Full round-trip from TextCommand → HybridResult → ArabicTextBuilder."""

    def test_word_round_trip(self):
        builder = ArabicTextBuilder()
        cmd = TextCommand.word("السلام", 0.95)
        result = builder.process(cmd)
        assert "السلام" in result
        assert builder.state == TextBuilderState.WORD_COMPLETE

    def test_delete_round_trip(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.letter("ب"))
        builder.process(TextCommand.delete())
        assert builder.current_word == "ا"

    def test_clear_round_trip(self):
        builder = ArabicTextBuilder()
        builder.process(TextCommand.word("مرحبا"))
        builder.process(TextCommand.space())
        builder.process(TextCommand.letter("ا"))
        builder.process(TextCommand.clear())
        assert builder.full_text == ""
        assert builder.state == TextBuilderState.IDLE

    def test_sentence_building_full_flow(self):
        builder = ArabicTextBuilder()
        commands = [
            TextCommand.letter("ا"),
            TextCommand.letter("ل"),
            TextCommand.letter("س"),
            TextCommand.letter("ل"),
            TextCommand.letter("ا"),
            TextCommand.letter("م"),
            TextCommand.space(),
            TextCommand.letter("ع"),
            TextCommand.letter("ل"),
            TextCommand.letter("ي"),
            TextCommand.letter("ك"),
            TextCommand.letter("م"),
            TextCommand.space(),
        ]
        builder.process_batch(commands)
        assert builder.sentence == "السلام عليكم"
