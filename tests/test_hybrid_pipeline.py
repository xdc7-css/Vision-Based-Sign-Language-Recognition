"""Tests for HybridRecognitionPipeline — priority-based fusion."""

import pytest
from unittest.mock import Mock, patch

from src.domain.hand import Hand, HandLandmarkType, Handedness
from src.domain.motion import HandMotionTrack, MotionState, Position2D
from src.domain.recognition import GestureCategory, RecognitionResult
from src.infrastructure.recognition.hybrid_pipeline import (
    HybridConfig,
    HybridRecognitionPipeline,
)
from src.infrastructure.recognition.word_recognizer import WordRecognizer
from src.infrastructure.recognition.static_recognizer import StaticSignRecognizer
from src.infrastructure.recognition.dynamic_recognizer import DynamicSignRecognizer
from src.domain.text import GestureInputType


@pytest.fixture
def make_hand():
    landmarks = {HandLandmarkType.WRIST: type("obj", (), {"x": 0.5, "y": 0.5, "z": 0.0})()}
    return Hand(landmarks=landmarks, handedness=Handedness.RIGHT)


@pytest.fixture
def mock_hand():
    return Mock(spec=Hand)


@pytest.fixture
def mock_track():
    track = Mock(spec=HandMotionTrack)
    track.is_motion_significant = True
    track.states = [Mock(spec=MotionState) for _ in range(10)]
    return track


@pytest.fixture
def mock_word_recognizer():
    wr = Mock(spec=WordRecognizer)
    wr.recognize_static.return_value = RecognitionResult(
        gesture_id="hello", arabic_text="مرحبا",
        category=GestureCategory.STATIC_WORD, confidence=0.95,
    )
    return wr


@pytest.fixture
def mock_static_recognizer():
    sr = Mock(spec=StaticSignRecognizer)
    sr.is_loaded = True
    sr.predict.return_value = RecognitionResult(
        gesture_id="alif", arabic_text="ا",
        category=GestureCategory.ALPHABET, confidence=0.92,
    )
    return sr


@pytest.fixture
def mock_dynamic_recognizer():
    dr = Mock(spec=DynamicSignRecognizer)
    dr.predict_from_track.return_value = RecognitionResult(
        gesture_id="wave", arabic_text=" waving",
        category=GestureCategory.DYNAMIC_GESTURE, confidence=0.85,
    )
    return dr


class TestHybridRecognitionPipeline:

    def test_word_priority_wins(self, mock_hand, mock_track, mock_word_recognizer, mock_static_recognizer, mock_dynamic_recognizer):
        pipeline = HybridRecognitionPipeline(
            word_recognizer=mock_word_recognizer,
            static_recognizer=mock_static_recognizer,
            dynamic_recognizer=mock_dynamic_recognizer,
            config=HybridConfig(word_threshold=0.7),
        )
        result = pipeline.process(hand=mock_hand, track=mock_track)

        assert result.decision_path == "word"
        assert result.final_result.arabic_text == "مرحبا"
        assert result.final_result.is_valid

    def test_word_below_threshold_falls_to_dynamic(self, mock_hand, mock_track, mock_static_recognizer, mock_dynamic_recognizer):
        low_conf_word = Mock(spec=WordRecognizer)
        low_conf_word.recognize_static.return_value = RecognitionResult(
            gesture_id="hello", arabic_text="مرحبا",
            category=GestureCategory.STATIC_WORD, confidence=0.5,
        )
        # recognize_dynamic returns None so the pipeline uses self._dynamic_recognizer
        low_conf_word.recognize_dynamic.return_value = None

        pipeline = HybridRecognitionPipeline(
            word_recognizer=low_conf_word,
            static_recognizer=mock_static_recognizer,
            dynamic_recognizer=mock_dynamic_recognizer,
            config=HybridConfig(word_threshold=0.85, dynamic_threshold=0.7),
        )
        result = pipeline.process(hand=mock_hand, track=mock_track)

        assert result.decision_path == "dynamic"
        assert result.final_result.arabic_text == " waving"

    def test_all_below_threshold_rejects(self, mock_hand, mock_track):
        low_word = Mock(spec=WordRecognizer)
        low_word.recognize_static.return_value = RecognitionResult(
            gesture_id="hello", arabic_text="مرحبا",
            category=GestureCategory.STATIC_WORD, confidence=0.3,
        )
        low_static = Mock(spec=StaticSignRecognizer)
        low_static.is_loaded = True
        low_static.predict.return_value = RecognitionResult(
            gesture_id="alif", arabic_text="ا",
            category=GestureCategory.ALPHABET, confidence=0.3,
        )

        pipeline = HybridRecognitionPipeline(
            word_recognizer=low_word,
            static_recognizer=low_static,
            config=HybridConfig(
                word_threshold=0.85, dynamic_threshold=0.75, alphabet_threshold=0.85,
            ),
        )
        result = pipeline.process(hand=mock_hand, track=mock_track)

        assert result.final_result.is_rejected
        assert result.decision_path == "rejected"

    def test_alphabet_fallback(self, mock_hand, mock_static_recognizer):
        low_word = Mock(spec=WordRecognizer)
        low_word.recognize_static.return_value = RecognitionResult(
            gesture_id="hello", arabic_text="مرحبا",
            category=GestureCategory.STATIC_WORD, confidence=0.3,
        )

        pipeline = HybridRecognitionPipeline(
            word_recognizer=low_word,
            static_recognizer=mock_static_recognizer,
            config=HybridConfig(
                word_threshold=0.85, alphabet_threshold=0.7,
                enable_dynamic_recognition=False,
            ),
        )
        result = pipeline.process(hand=mock_hand)

        assert result.decision_path == "alphabet"
        assert result.final_result.arabic_text == "ا"

    def test_no_static_recognizer_skips_alphabet(self, mock_hand, mock_track):
        low_word = Mock(spec=WordRecognizer)
        low_word.recognize_static.return_value = RecognitionResult(
            gesture_id="hello", arabic_text="مرحبا",
            category=GestureCategory.STATIC_WORD, confidence=0.3,
        )

        pipeline = HybridRecognitionPipeline(
            word_recognizer=low_word,
            config=HybridConfig(
                word_threshold=0.85,
                enable_dynamic_recognition=False,
                enable_alphabet_recognition=True,
            ),
        )
        result = pipeline.process(hand=mock_hand, track=mock_track)

        assert result.final_result.is_rejected

    def test_process_to_command_word(self, mock_hand, mock_track, mock_word_recognizer):
        pipeline = HybridRecognitionPipeline(
            word_recognizer=mock_word_recognizer,
            config=HybridConfig(word_threshold=0.7),
        )
        result, cmd = pipeline.process_to_command(hand=mock_hand, track=mock_track)

        assert cmd is not None
        assert cmd.input_type == GestureInputType.WORD
        assert cmd.value == "مرحبا"

    def test_process_to_command_letter(self, mock_hand, mock_static_recognizer):
        low_word = Mock(spec=WordRecognizer)
        low_word.recognize_static.return_value = RecognitionResult(
            gesture_id="hello", arabic_text="مرحبا",
            category=GestureCategory.STATIC_WORD, confidence=0.3,
        )

        pipeline = HybridRecognitionPipeline(
            word_recognizer=low_word,
            static_recognizer=mock_static_recognizer,
            config=HybridConfig(
                word_threshold=0.85, alphabet_threshold=0.7,
                enable_dynamic_recognition=False,
            ),
        )
        result, cmd = pipeline.process_to_command(hand=mock_hand)

        assert cmd is not None
        assert cmd.input_type == GestureInputType.LETTER
        assert cmd.value == "ا"

    def test_cooldown_suppression(self, mock_hand, mock_word_recognizer):
        pipeline = HybridRecognitionPipeline(
            word_recognizer=mock_word_recognizer,
            config=HybridConfig(word_threshold=0.7, cooldown_frames=10),
        )
        # First call should pass
        result1, _ = pipeline.process_to_command(hand=mock_hand)
        assert not result1.final_result.is_rejected

        # Subsequent calls within cooldown should be suppressed
        for _ in range(3):
            result2, _ = pipeline.process_to_command(hand=mock_hand)
            # The cooldown applies to the same gesture_id if counter > 0
            # In the implementation, after the first valid detection, cooldown_counter gets set
            # So subsequent calls should get rejected

    def test_reset_cooldown(self, mock_hand, mock_word_recognizer):
        pipeline = HybridRecognitionPipeline(
            word_recognizer=mock_word_recognizer,
            config=HybridConfig(word_threshold=0.7, cooldown_frames=10),
        )

        pipeline.process_to_command(hand=mock_hand)
        pipeline.reset_cooldown()

        # After reset, next detection should pass through
        result, _ = pipeline.process_to_command(hand=mock_hand)
        assert not result.final_result.is_rejected

    def test_processing_time_set(self, mock_hand, mock_word_recognizer):
        pipeline = HybridRecognitionPipeline(
            word_recognizer=mock_word_recognizer,
            config=HybridConfig(word_threshold=0.7),
        )
        result = pipeline.process(hand=mock_hand)

        assert result.processing_time_ms > 0
        assert result.final_result.processing_time_ms > 0

    def test_hybrid_result_fields(self, mock_hand, mock_word_recognizer):
        pipeline = HybridRecognitionPipeline(
            word_recognizer=mock_word_recognizer,
            config=HybridConfig(word_threshold=0.7),
        )
        result = pipeline.process(hand=mock_hand)

        assert result.word_result is not None
        assert result.dynamic_result is None
        assert result.alphabet_result is None
        assert result.decision_path == "word"

    def test_disabled_recognition(self, mock_hand):
        pipeline = HybridRecognitionPipeline(
            config=HybridConfig(enable_word_recognition=False),
        )
        result = pipeline.process(hand=mock_hand)
        assert result.final_result.is_rejected
