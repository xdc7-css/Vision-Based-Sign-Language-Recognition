"""Tests for recognition modules — static, dynamic, word-level recognizers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.domain.hand import Hand, HandLandmark, Handedness
from src.domain.motion import HandMotionTrack, MotionState, Position2D, Velocity2D, Acceleration2D
from src.domain.recognition import (
    GestureCategory,
    RecognitionResult,
    VocabularyEntry,
    WordVocabulary,
)
from src.infrastructure.recognition.static_recognizer import (
    StaticModelConfig,
    StaticSignRecognizer,
)
from src.infrastructure.recognition.word_recognizer import WordRecognizer, WordRecognizerConfig


# ── Vocabulary tests ──────────────────────────────────────────────────


class TestWordVocabulary:
    @pytest.fixture
    def vocab(self) -> WordVocabulary:
        v = WordVocabulary()
        v.register(VocabularyEntry("greeting_hello", "مرحبا", GestureCategory.STATIC_WORD))
        v.register(VocabularyEntry("thanks", "شكرا", GestureCategory.STATIC_WORD))
        return v

    def test_register_and_get(self, vocab: WordVocabulary) -> None:
        entry = vocab.get("greeting_hello")
        assert entry is not None
        assert entry.arabic_text == "مرحبا"
        assert entry.category == GestureCategory.STATIC_WORD

    def test_get_unknown(self, vocab: WordVocabulary) -> None:
        assert vocab.get("nonexistent") is None

    def test_contains(self, vocab: WordVocabulary) -> None:
        assert "greeting_hello" in vocab
        assert "nonexistent" not in vocab

    def test_len(self, vocab: WordVocabulary) -> None:
        assert len(vocab) == 2

    def test_add_duplicate_overwrites(self, vocab: WordVocabulary) -> None:
        vocab.register(VocabularyEntry("greeting_hello", "أهلا", GestureCategory.STATIC_WORD))
        entry = vocab.get("greeting_hello")
        assert entry.arabic_text == "أهلا"


# ── StaticSignRecognizer tests ────────────────────────────────────────


class TestStaticSignRecognizer:
    @pytest.fixture
    def recognizer(self) -> StaticSignRecognizer:
        return StaticSignRecognizer(StaticModelConfig(model_type="rf"))

    def test_predict_without_model(self, recognizer: StaticSignRecognizer) -> None:
        hand = Hand(
            landmarks=[HandLandmark(0.5, 0.5, 0.0) for _ in range(21)],
            handedness=Handedness.RIGHT,
            confidence=0.95,
        )
        result = recognizer.predict(hand)
        assert result.is_rejected
        assert "not loaded" in result.rejection_reason

    def test_train_and_predict(self, recognizer: StaticSignRecognizer) -> None:
        X = np.random.rand(20, 63).astype(np.float32)
        y = np.array(["hello"] * 10 + ["thanks"] * 10)
        recognizer.train(X, y, model_types=["rf"])
        assert recognizer.is_loaded
        assert recognizer.model_type == "rf"

    def test_predict_with_confidence_threshold(self, recognizer: StaticSignRecognizer) -> None:
        X = np.random.rand(20, 63).astype(np.float32)
        y = np.array(["class_a"] * 10 + ["class_b"] * 10)
        recognizer.train(X, y, model_types=["rf"])

        hand = Hand(
            landmarks=[HandLandmark(0.5, 0.5, 0.0) for _ in range(21)],
            handedness=Handedness.RIGHT,
            confidence=0.95,
        )
        result = recognizer.predict(hand, confidence_threshold=0.99)
        # With random data and high threshold, likely rejected
        if not result.is_rejected:
            assert result.confidence > 0

    def test_save_load(self, recognizer: StaticSignRecognizer) -> None:
        X = np.random.rand(20, 63).astype(np.float32)
        y = np.array(["a"] * 10 + ["b"] * 10)
        recognizer.train(X, y, model_types=["rf"])

        with tempfile.TemporaryDirectory() as tmp:
            path = recognizer.save(tmp)
            assert Path(path).exists()

            new = StaticSignRecognizer()
            assert new.load(path)
            assert new.is_loaded
            assert new.model_type == "rf"


# ── WordRecognizer tests ──────────────────────────────────────────────


class TestWordRecognizer:
    @pytest.fixture
    def vocab(self) -> WordVocabulary:
        v = WordVocabulary()
        v.register(VocabularyEntry("hello", "مرحبا", GestureCategory.STATIC_WORD))
        v.register(VocabularyEntry("thanks", "شكرا", GestureCategory.STATIC_WORD))
        return v

    @pytest.fixture
    def recognizer(self, vocab: WordVocabulary) -> WordRecognizer:
        return WordRecognizer(vocabulary=vocab)

    def test_no_recognizers(self, recognizer: WordRecognizer) -> None:
        result = recognizer.recognize_static(
            Hand(
                landmarks=[HandLandmark(0.5, 0.5, 0.0) for _ in range(21)],
                handedness=Handedness.RIGHT,
                confidence=0.95,
            )
        )
        assert result.is_rejected
        assert "not available" in result.rejection_reason

    def test_reject(self, recognizer: WordRecognizer) -> None:
        result = RecognitionResult(
            gesture_id="test", arabic_text="test",
            category=GestureCategory.STATIC_WORD,
            confidence=0.3,
        )
        rejected = recognizer.reject(result, "Too uncertain")
        assert rejected.is_rejected
        assert rejected.rejection_reason == "Too uncertain"
        assert rejected.confidence == 0.0

    def test_session_tracking(self, recognizer: WordRecognizer) -> None:
        assert recognizer.session.text_output == ""
        recognizer.reset_session()
        assert len(recognizer.session.results) == 0

    def test_load_vocabulary_json(self, recognizer: WordRecognizer) -> None:
        import json
        with tempfile.TemporaryDirectory() as tmp:
            vocab_file = Path(tmp) / "vocab.json"
            data = {
                "gestures": {
                    "hello": {"arabic": "مرحبا", "type": "static"},
                    "thanks": {"arabic": "شكرا", "type": "static"},
                }
            }
            with open(vocab_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

            count = recognizer.load_vocabulary_from_json(vocab_file)
            assert count == 2

    def test_fuse_both_none(self, recognizer: WordRecognizer) -> None:
        result = recognizer._fuse_decisions(None, None)
        assert result.is_rejected
        assert "no input" in result.rejection_reason.lower()

    def test_fuse_static_priority(self, recognizer: WordRecognizer) -> None:
        static = RecognitionResult("s1", "مرحبا", GestureCategory.STATIC_WORD, 0.9)
        dynamic = RecognitionResult("d1", "شكرا", GestureCategory.DYNAMIC_GESTURE, 0.8)
        result = recognizer._fuse_decisions(static, dynamic)
        assert result.gesture_id == "s1"
        assert result.arabic_text == "مرحبا"

    def test_fuse_agreement_boost(self, recognizer: WordRecognizer) -> None:
        static = RecognitionResult("word1", "نعم", GestureCategory.STATIC_WORD, 0.7)
        dynamic = RecognitionResult("word1", "نعم", GestureCategory.DYNAMIC_GESTURE, 0.6)
        result = recognizer._fuse_decisions(static, dynamic)
        # Confidence should be boosted (0.7 + 0.6*0.3 = 0.88)
        assert result.confidence > 0.7

    def test_cooldown_blocks_duplicates(self, recognizer: WordRecognizer) -> None:
        result1 = RecognitionResult("hello", "مرحبا", GestureCategory.STATIC_WORD, 0.9)
        result2 = RecognitionResult("hello", "مرحبا", GestureCategory.STATIC_WORD, 0.9)

        r1 = recognizer._apply_cooldown(result1)
        assert r1.is_valid

        r2 = recognizer._apply_cooldown(result2)
        assert r2.is_rejected  # blocked by cooldown
        assert "cooldown" in r2.rejection_reason.lower()

    def test_empty_vocabulary_default(self) -> None:
        recognizer = WordRecognizer()
        assert len(recognizer.vocabulary) > 0  # default vocabulary loaded


# ── StaticSignRecognizer alternative model types ──────────────────────


class TestStaticRecognizerModels:
    def test_svm_training(self) -> None:
        r = StaticSignRecognizer(StaticModelConfig(model_type="svm"))
        X = np.random.rand(30, 63)
        y = np.array(["a"] * 15 + ["b"] * 15)
        metrics = r.train(X, y, model_types=["svm"])
        assert "svm" in metrics

    def test_mlp_training(self) -> None:
        r = StaticSignRecognizer(StaticModelConfig(model_type="mlp"))
        X = np.random.rand(30, 63)
        y = np.array(["x"] * 15 + ["y"] * 15)
        metrics = r.train(X, y, model_types=["mlp"])
        assert "mlp" in metrics
