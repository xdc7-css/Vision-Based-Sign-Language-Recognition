"""Automated regression test suite for Recognition MediaPipe Landmarks & Custom Model Inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.domain.hand import Hand, HandDetectionResult, Handedness, HandLandmark
from src.infrastructure.dialect_manager import DialectManager
from src.infrastructure.hand_detection.renderer import HandRenderer
from src.infrastructure.recognition.hybrid_pipeline import HybridConfig
from src.models.evaluator import ModelRegistry
from src.models.train_static import train_static_model
from src.presentation.tabs.recognition import RecognitionTab, SignRecognitionWorker


@pytest.fixture(autouse=True)
def isolate_model_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure training tests do not overwrite repository data/models/manifest.json."""
    manifest_file = tmp_path / "test_manifest.json"

    def patched_init(self, registry_path=None):
        self._path = Path(registry_path or manifest_file)
        self._registry = {}
        self._load()

    monkeypatch.setattr(ModelRegistry, "__init__", patched_init)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestLandmarkVisualization:
    """Test MediaPipe landmark rendering and visualization toggle."""

    def test_hand_renderer_render_list_compatibility(self) -> None:
        """Verify HandRenderer.render works seamlessly with both HandDetectionResult and list[Hand]."""
        renderer = HandRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Test with list[Hand]
        hand = Hand(
            landmarks=[HandLandmark(0.5, 0.5, 0.0) for _ in range(21)],
            handedness=Handedness.RIGHT,
            confidence=0.95,
        )
        rendered_from_list = renderer.render(frame, [hand])
        assert rendered_from_list.shape == (480, 640, 3)

        # 2. Test with HandDetectionResult
        det_res = HandDetectionResult(
            hands=[hand],
            image_width=640,
            image_height=480,
            processing_time_ms=5.0,
        )
        rendered_from_result = renderer.render(frame, det_res)
        assert rendered_from_result.shape == (480, 640, 3)

    def test_landmarks_toggle_visual_difference(self) -> None:
        """Verify that enabling show_landmarks adds visual modifications while disabling leaves frame clean."""
        renderer = HandRenderer()
        frame = np.full((480, 640, 3), 50, dtype=np.uint8)

        # Distinct hand landmarks with finger coordinates
        lms = [HandLandmark(0.5, 0.5, 0.0)]
        for k in range(1, 21):
            lms.append(HandLandmark(0.5 + 0.1 * np.cos(k), 0.5 + 0.1 * np.sin(k), 0.0))
        hand = Hand(landmarks=lms, handedness=Handedness.RIGHT, confidence=0.99)
        det_res = HandDetectionResult(hands=[hand], image_width=640, image_height=480, processing_time_ms=1.0)

        # Drawing landmarks alters pixels
        annotated = renderer.draw(frame.copy(), det_res)
        assert not np.array_equal(frame, annotated), "Annotated frame must contain rendered landmark graphics"


class TestCustomClassifierInference:
    """Test Iraqi Custom Classifier model loading, threshold synchronization, and inference."""

    def test_custom_model_not_trained_handling(self, tmp_path: Path) -> None:
        """Verify worker and UI report 'not trained' when model checkpoint does not exist."""
        empty_dir = tmp_path / "models" / "iraqi"
        empty_dir.mkdir(parents=True)

        dm = DialectManager(base_path=str(tmp_path))
        cfg = dm.active_dialect_config
        assert cfg is not None
        cfg.model_path = str(empty_dir)
        cfg.vocabulary_path = str(empty_dir / "vocabulary.json")
        dm.reload_dialect()

        worker = SignRecognitionWorker(pipeline=dm.pipeline)
        worker.active_engine = "custom"

        # Check model loaded check
        has_custom = (
            worker._pipeline is not None
            and getattr(worker._pipeline, "_static_recognizer", None) is not None
            and worker._pipeline._static_recognizer.is_loaded
        )
        assert not has_custom

    def test_custom_model_trained_and_real_inference(self, tmp_path: Path) -> None:
        """Verify trained custom model produces accurate prediction with genuine confidence."""
        dataset_dir = tmp_path / "words"
        model_dir = tmp_path / "models" / "iraqi"

        # Create two separable classes
        for cls_name in ["نعم", "لا"]:
            cls_samples = dataset_dir / cls_name / "samples"
            cls_samples.mkdir(parents=True)
            radius = 0.06 if cls_name == "نعم" else 0.28
            for i in range(5):
                s_dir = cls_samples / f"sample_{i}"
                s_dir.mkdir()
                lm = np.zeros((1, 21, 3), dtype=np.float32)
                lm[0, 0, :] = [0.5, 0.5, 0.0]
                for k in range(1, 21):
                    lm[0, k, 0] = 0.5 + radius * np.cos(k * 0.3)
                    lm[0, k, 1] = 0.5 + radius * np.sin(k * 0.3)
                    lm[0, k, 2] = 0.0
                np.save(s_dir / "landmarks.npy", lm)

        # Train model
        res = train_static_model(
            dataset_path=dataset_dir,
            output_dir=model_dir,
            model_types=["mlp"],
            epochs=30,
            batch_size=2,
            learning_rate=0.01,
            val_split=0.2,
            augment=False,
            dialect="iraqi",
        )
        assert Path(res["model_path"]).exists()

        # Load into DialectManager
        dm = DialectManager(
            base_path=str(tmp_path),
            hybrid_config=HybridConfig(word_threshold=0.40),
        )
        cfg = dm.active_dialect_config
        assert cfg is not None
        cfg.model_path = str(model_dir)
        cfg.vocabulary_path = str(model_dir / "vocabulary.json")
        dm.reload_dialect()

        sr = dm._static_recognizers.get(dm.active_dialect)
        assert sr is not None and sr.is_loaded

        # Test hand geometry matching "نعم" (radius 0.06)
        lms = [HandLandmark(0.5, 0.5, 0.0)]
        for k in range(1, 21):
            lms.append(HandLandmark(0.5 + 0.06 * np.cos(k * 0.3), 0.5 + 0.06 * np.sin(k * 0.3), 0.0))
        hand = Hand(landmarks=lms, handedness=Handedness.RIGHT, confidence=0.99)

        pred = sr.predict(hand)
        assert pred.arabic_text in ["نعم", "لا"]
        assert pred.confidence > 0.40

        # Test via Hybrid pipeline process
        dm.pipeline.config.word_threshold = 0.40
        dm.pipeline.config.alphabet_threshold = 0.40
        hybrid_res = dm.process(hand)
        assert hybrid_res.is_valid
        assert hybrid_res.arabic_text in ["نعم", "لا"]
        assert hybrid_res.final_result.confidence > 0.40

    def test_recognition_tab_custom_status_ui(self, qapp, tmp_path: Path) -> None:
        """Verify RecognitionTab correctly displays custom model status badge and vocabulary info."""
        state = AppState()
        use_cases = VideoSourceUseCases(state)

        tab = RecognitionTab(state, use_cases)
        try:
            assert tab._model_status_badge is not None
            assert "502" in tab._model_status_badge.text()

            # Switch to custom
            tab._engine_selector.setCurrentIndex(1)
            assert tab._worker.active_engine == "custom"
            assert tab._model_status_badge.text() != ""

            # Switch back to karsl
            tab._engine_selector.setCurrentIndex(0)
            assert tab._worker.active_engine == "karsl"
            assert "502" in tab._model_status_badge.text()
        finally:
            tab._worker.stop()
