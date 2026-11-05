"""Unit and integration tests for KArSL-502 Pretrained Recognition Engine."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.infrastructure.recognition.karsl_recognizer import (
    KArSLConfig,
    KArSLRecognizer,
    TSNSampler,
    LandmarkerProcessor,
)
from src.presentation.dialogs.karsl_vocab_dialog import KArSLVocabularyDialog


def test_tsn_sampler_interpolation():
    """Verify TSN uniform temporal sampling to exactly 50 frames."""
    sampler = TSNSampler(target_len=50)

    # Test short sequence (15 frames)
    arr_15 = np.random.randn(15, 184, 4).astype(np.float32)
    out_15 = sampler(arr_15)
    assert out_15.shape == (50, 184, 4)

    # Test intermediate sequence (37 frames)
    arr_37 = np.random.randn(37, 184, 4).astype(np.float32)
    out_37 = sampler(arr_37)
    assert out_37.shape == (50, 184, 4)

    # Test exact sequence (50 frames)
    arr_50 = np.random.randn(50, 184, 4).astype(np.float32)
    out_50 = sampler(arr_50)
    assert out_50.shape == (50, 184, 4)


def test_karsl_model_and_labels_integrity():
    """Verify ONNX model file and JSON label vocabulary."""
    model_path = Path("data/models/karsl/karsl_502.onnx")
    labels_path = Path("data/models/karsl/labels.json")

    assert model_path.exists(), "KArSL ONNX model must exist"
    assert model_path.stat().st_size == 7330073, f"Size must be 7330073 bytes, got {model_path.stat().st_size}"

    assert labels_path.exists(), "KArSL labels JSON must exist"
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "AR_WORDS" in data
    assert "EN_WORDS" in data
    assert len(data["AR_WORDS"]) == 502
    assert len(data["EN_WORDS"]) == 502


def test_karsl_recognizer_loading_and_inference():
    """Verify KArSLRecognizer initialization, session loading, and frame processing."""
    recognizer = KArSLRecognizer()
    assert recognizer.is_loaded is True

    vocab = recognizer.get_vocabulary()
    assert len(vocab) == 502
    assert vocab[0]["index"] == 0
    assert vocab[501]["index"] == 501

    # Test dummy frame processing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res, ann_frame, vis_info = recognizer.process_frame(dummy_frame)

    assert ann_frame.shape == (480, 640, 3)
    assert isinstance(vis_info, dict)
    assert res is not None
    assert res.pred_idx == -1  # No sign yet because buffer is empty

    # Reset
    recognizer.reset()
    assert len(recognizer._frame_buffer) == 0


def test_karsl_vocab_loader():
    """Verify KArSLVocabularyDialog helper loads 502 entries."""
    vocab = KArSLVocabularyDialog._load_default_vocabulary()
    assert len(vocab) == 502
    assert any(item["arabic"] == "0" for item in vocab)
    assert any("أرقام" in item["category"] for item in vocab)
    assert any("حروف" in item["category"] for item in vocab)
    assert any("كلمات" in item["category"] for item in vocab)

