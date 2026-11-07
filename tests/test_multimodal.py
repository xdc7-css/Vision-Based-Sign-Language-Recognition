"""Comprehensive test suite for Multimodal Sign Language Recognition (Hands + Face + Pose).

Covers:
- Domain models and schema definitions (v2_multimodal vs v1_single_hand).
- MultimodalDetector tracking, handedness inversion, and continuity.
- MultimodalFeatureExtractor tensor vectorization, presence masking, and anchor normalization.
- MultimodalRenderer color-coded drawing and toggle behavior.
- Dataset validation, capture session, and disk persistence.
- Multimodal dataset inspection, model training (MLP/RF/SVM), metrics, and checkpoint generation.
- MultimodalSignRecognizer inference, confidence thresholds, and untrained state handling.
- PySide6 UI Tab integration (Dataset Studio, Recognition, Training).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.domain.hand import Hand, Handedness, HandLandmark
from src.domain.multimodal import (
    DUAL_HAND_DIM,
    FACE_DIM,
    FACE_EXPRESSIVE_INDICES,
    FEATURE_DIM_FULL,
    FEATURE_DIM_HANDS,
    FEATURE_DIM_HANDS_FACE,
    HAND_DIM,
    LEGACY_SCHEMA_VERSION,
    MULTIMODAL_SCHEMA_VERSION,
    POSE_DIM,
    POSE_UPPER_BODY_INDICES,
    FaceLandmark,
    ModalityMode,
    MultimodalFeatureSchema,
    MultimodalFrame,
    PoseLandmark,
)
from src.infrastructure.dataset.collector import CaptureSession, DatasetCollector
from src.infrastructure.dataset.validator import DatasetValidator
from src.infrastructure.recognition.multimodal_recognizer import MultimodalSignRecognizer
from src.infrastructure.vision.feature_extractor import MultimodalFeatureExtractor
from src.infrastructure.vision.multimodal_detector import MultimodalDetector
from src.infrastructure.vision.multimodal_renderer import MultimodalRenderer
from src.models.train_multimodal import (
    inspect_multimodal_dataset,
    load_multimodal_dataset,
    train_multimodal_model,
)
from src.presentation.tabs.dataset import DatasetCollectionTab
from src.presentation.tabs.recognition import RecognitionTab
from src.presentation.tabs.training import ModelTrainingTab


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def create_synthetic_hand(handedness: Handedness = Handedness.RIGHT, offset: float = 0.0) -> Hand:
    landmarks = [
        HandLandmark(
            x=min(0.99, max(0.01, 0.3 + offset + 0.01 * i)),
            y=min(0.99, max(0.01, 0.4 + 0.01 * i)),
            z=0.0,
        )
        for i in range(21)
    ]
    return Hand(landmarks=landmarks, handedness=handedness, confidence=0.95)


def create_synthetic_face(num_landmarks: int = 24) -> list[FaceLandmark]:
    return [
        FaceLandmark(
            x=min(0.99, max(0.01, 0.5 + 0.05 * np.cos(i))),
            y=min(0.99, max(0.01, 0.3 + 0.05 * np.sin(i))),
            z=0.0,
        )
        for i in range(num_landmarks)
    ]


def create_synthetic_pose(num_landmarks: int = 8) -> list[PoseLandmark]:
    return [
        PoseLandmark(
            x=min(0.99, max(0.01, 0.5 + 0.1 * (i % 2 - 0.5))),
            y=min(0.99, max(0.01, 0.4 + 0.05 * i)),
            z=0.0,
            visibility=0.9,
        )
        for i in range(num_landmarks)
    ]


def create_synthetic_multimodal_frame(
    has_left: bool = True,
    has_right: bool = True,
    has_face: bool = True,
    has_pose: bool = False,
) -> MultimodalFrame:
    left = create_synthetic_hand(Handedness.LEFT, offset=-0.2) if has_left else None
    right = create_synthetic_hand(Handedness.RIGHT, offset=0.2) if has_right else None
    face = create_synthetic_face(24) if has_face else None
    pose = create_synthetic_pose(8) if has_pose else None

    return MultimodalFrame(
        left_hand=left,
        right_hand=right,
        face_landmarks=face or [],
        pose_landmarks=pose or [],
        has_left_hand=has_left,
        has_right_hand=has_right,
        has_face=has_face,
        has_pose=has_pose,
        mouth_openness=0.25,
        eye_openness_left=0.45,
        eye_openness_right=0.45,
        eyebrow_elevation=0.60,
        left_hand_to_face_dist=0.35,
        right_hand_to_face_dist=0.38,
        geometric_indicators={
            "mouth_openness": 0.25,
            "eye_openness": 0.45,
            "eyebrow_elevation": 0.60,
            "left_wrist_to_nose": 0.35,
            "right_wrist_to_nose": 0.38,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Domain Models and Schema Specifications
# ─────────────────────────────────────────────────────────────────────────────

class TestMultimodalDomain:
    """Verify domain representations, indices, and dimensions."""

    def test_schema_versions_and_constants(self) -> None:
        assert MULTIMODAL_SCHEMA_VERSION == "v2_multimodal"
        assert LEGACY_SCHEMA_VERSION == "v1_single_hand"
        assert len(FACE_EXPRESSIVE_INDICES) == 24
        assert len(POSE_UPPER_BODY_INDICES) == 8

        # Dimension checks
        assert HAND_DIM == 63
        assert DUAL_HAND_DIM == 126
        assert FACE_DIM == 72
        assert POSE_DIM == 24
        assert FEATURE_DIM_HANDS == 134
        assert FEATURE_DIM_HANDS_FACE == 206
        assert FEATURE_DIM_FULL == 230

    def test_multimodal_frame_properties(self) -> None:
        frame_empty = MultimodalFrame()
        assert not frame_empty.has_left
        assert not frame_empty.has_right
        assert not frame_empty.has_face
        assert not frame_empty.has_pose
        assert frame_empty.num_hands == 0

        frame_dual = create_synthetic_multimodal_frame(has_left=True, has_right=True, has_face=True, has_pose=True)
        assert frame_dual.has_left
        assert frame_dual.has_right
        assert frame_dual.has_face
        assert frame_dual.has_pose
        assert frame_dual.num_hands == 2

    def test_feature_schema_generation(self) -> None:
        schema = MultimodalFeatureSchema.generate(ModalityMode.TWO_HANDS_FACE)
        assert schema.schema_version == MULTIMODAL_SCHEMA_VERSION
        assert schema.feature_dim == 206
        assert schema.includes_face is True
        assert schema.includes_pose is False
        assert len(schema.feature_names) == 206


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multimodal Detector & Hand Tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestMultimodalDetector:
    """Verify multimodal detector initialization, handedness calibration, and geometric calculation."""

    def test_detector_initialization_and_empty_frame(self) -> None:
        detector = MultimodalDetector()
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Detect on blank frame
        result = detector.detect(black_frame)
        assert isinstance(result, MultimodalFrame)
        assert not result.has_left
        assert not result.has_right
        assert not result.has_face
        assert not result.has_pose

    def test_handedness_calibration_under_mirroring(self) -> None:
        calibrated_left = MultimodalDetector.calibrate_handedness("Right", mirror=True)
        assert calibrated_left == Handedness.LEFT

        calibrated_right = MultimodalDetector.calibrate_handedness("Left", mirror=True)
        assert calibrated_right == Handedness.RIGHT

        unmirrored_right = MultimodalDetector.calibrate_handedness("Right", mirror=False)
        assert unmirrored_right == Handedness.RIGHT

    def test_wrist_continuity_prevents_hand_swapping(self, tmp_path: Path) -> None:
        detector = MultimodalDetector(models_dir=tmp_path)

        detector._prev_left_wrist = (0.2, 0.5, 0.0)
        detector._prev_right_wrist = (0.8, 0.5, 0.0)

        c1 = Hand(
            landmarks=[HandLandmark(0.21, 0.49, 0.0) for _ in range(21)],
            handedness=Handedness.UNKNOWN,
            confidence=0.5,
        )

        left_out, right_out = detector.assign_and_stabilize_hands([c1])
        assert left_out is c1
        assert right_out is None
        assert left_out.handedness == Handedness.LEFT


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multimodal Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class TestMultimodalFeatureExtractor:
    """Verify normalization, presence flags, zero-padding, and fixed tensor dimensions."""

    def test_feature_dimensions_per_mode(self) -> None:
        extractor = MultimodalFeatureExtractor()
        frame = create_synthetic_multimodal_frame(has_left=True, has_right=True, has_face=True, has_pose=True)

        vec_full = extractor.extract(frame, mode=ModalityMode.TWO_HANDS_FACE_POSE)
        assert vec_full.shape == (230,)

        vec_face = extractor.extract(frame, mode=ModalityMode.TWO_HANDS_FACE)
        assert vec_face.shape == (206,)

        vec_hands = extractor.extract(frame, mode=ModalityMode.TWO_HANDS)
        assert vec_hands.shape == (134,)

        vec_legacy = extractor.extract(frame, mode=ModalityMode.ONE_HAND)
        assert vec_legacy.shape == (63,)

    def test_presence_flags_and_zero_imputation(self) -> None:
        extractor = MultimodalFeatureExtractor()

        frame_right_only = create_synthetic_multimodal_frame(has_left=False, has_right=True, has_face=False)
        vec = extractor.extract(frame_right_only, mode=ModalityMode.TWO_HANDS_FACE)

        assert vec.shape == (206,)
        left_hand_features = vec[0:63]
        right_hand_features = vec[63:126]
        face_features = vec[126:198]
        presence_flags = vec[198:206]

        assert np.all(left_hand_features == 0.0)
        assert np.any(right_hand_features != 0.0)
        assert np.all(face_features == 0.0)

        assert presence_flags[0] == 0.0  # has_left
        assert presence_flags[1] == 1.0  # has_right
        assert presence_flags[2] == 0.0  # has_face

    def test_anchor_normalization_preserves_relative_distance(self) -> None:
        extractor = MultimodalFeatureExtractor()

        frame = create_synthetic_multimodal_frame(has_left=True, has_right=True, has_face=True)
        vec1 = extractor.extract(frame, mode=ModalityMode.TWO_HANDS_FACE)

        assert np.all(np.isfinite(vec1))
        assert not np.any(np.isnan(vec1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multimodal Renderer
# ─────────────────────────────────────────────────────────────────────────────

class TestMultimodalRenderer:
    """Verify visual rendering, color-coding, and toggle behavior."""

    def test_renderer_toggle_behavior(self) -> None:
        renderer = MultimodalRenderer()
        frame_bgr = np.full((480, 640, 3), 40, dtype=np.uint8)
        frame_data = create_synthetic_multimodal_frame(has_left=True, has_right=True, has_face=True, has_pose=True)

        out_clean = renderer.render(
            frame_bgr,
            frame_data,
            show_hands=False,
            show_face=False,
            show_pose=False,
            show_labels=False,
        )
        assert np.array_equal(frame_bgr, out_clean)

        out_annotated = renderer.render(
            frame_bgr,
            frame_data,
            show_hands=True,
            show_face=True,
            show_pose=True,
            show_labels=True,
        )
        assert not np.array_equal(frame_bgr, out_annotated)
        assert out_annotated.shape == frame_bgr.shape


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset Validation & Capture Session Persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestMultimodalDatasetStorage:
    """Verify dataset validator handles v2_multimodal and v1_single_hand without regression."""

    def test_validator_accepts_v2_multimodal_samples(self, tmp_path: Path) -> None:
        sample_dir = tmp_path / "sample_v2"
        sample_dir.mkdir(parents=True)

        landmarks = np.zeros((1, 206), dtype=np.float32)
        np.save(sample_dir / "landmarks.npy", landmarks)

        meta = {
            "schema_version": "v2_multimodal",
            "feature_dim": 206,
            "modality_mode": "two_hands_face",
            "frame_count": 1,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        val = DatasetValidator()
        report = val.validate_sample(sample_dir)
        assert report.is_valid, f"Multimodal sample failed: {report.errors}"

    def test_validator_accepts_v1_legacy_samples(self, tmp_path: Path) -> None:
        sample_dir = tmp_path / "sample_v1"
        sample_dir.mkdir(parents=True)

        landmarks = np.zeros((1, 21, 3), dtype=np.float32)
        np.save(sample_dir / "landmarks.npy", landmarks)

        meta = {
            "schema_version": "v1_single_hand",
            "frame_count": 1,
            "landmarks_count": 21,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        val = DatasetValidator()
        report = val.validate_sample(sample_dir)
        assert report.is_valid, f"Legacy sample failed: {report.errors}"

    def test_capture_session_saves_multimodal_features(self, tmp_path: Path) -> None:
        collector = DatasetCollector(base_dir=tmp_path)
        session = CaptureSession(
            gesture_type="words",
            gesture_label="مرحبا",
        )
        session.is_recording = True

        dummy_features = np.ones(206, dtype=np.float32) * 0.42
        session.add_multimodal_frame(None, dummy_features)

        saved_path = collector.save_sample(session)
        assert saved_path is not None
        assert saved_path.exists()

        loaded = np.load(saved_path / "landmarks.npy")
        assert loaded.shape == (1, 206)
        assert np.isclose(loaded[0, 0], 0.42)

        with open(saved_path / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["schema_version"] == "v2_multimodal"
        assert meta["modality_mode"] == "two_hands_face"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multimodal Model Training & Inspection
# ─────────────────────────────────────────────────────────────────────────────

class TestMultimodalTrainingPipeline:
    """Verify multimodal dataset inspection, training, metrics, and persistence."""

    def test_inspect_and_load_multimodal_dataset(self, tmp_path: Path) -> None:
        dataset_root = tmp_path / "words"

        # Class 1: Has 5 v2_multimodal samples
        c1_dir = dataset_root / "نعم" / "samples"
        for i in range(5):
            s = c1_dir / f"sample_{i}"
            s.mkdir(parents=True)
            np.save(s / "landmarks.npy", np.random.randn(1, 206).astype(np.float32))
            with open(s / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "v2_multimodal", "feature_dim": 206}, f)

        # Class 2: Has only legacy v1 samples
        c2_dir = dataset_root / "شكرا" / "samples"
        for i in range(5):
            s = c2_dir / f"sample_legacy_{i}"
            s.mkdir(parents=True)
            np.save(s / "landmarks.npy", np.random.randn(1, 21, 3).astype(np.float32))
            with open(s / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({"schema_version": "v1_single_hand"}, f)

        summary = inspect_multimodal_dataset(tmp_path)
        assert "نعم" in summary["qualifying_classes"]
        assert summary["qualifying_classes"]["نعم"] == 5
        assert "شكرا" in summary["legacy_classes"]
        assert summary["multimodal_samples_count"] == 5
        assert summary["legacy_samples_count"] == 5

        # Loading should strictly load the 5 v2 samples
        X, y, classes = load_multimodal_dataset(tmp_path, min_classes=1)
        assert X.shape == (5, 206)
        assert len(y) == 5
        assert classes == ["نعم"]

    def test_train_multimodal_model_end_to_end(self, tmp_path: Path) -> None:
        dataset_root = tmp_path / "words"
        models_root = tmp_path / "models" / "multimodal"

        classes = ["أهلا", "مع_السلامة"]
        for c in classes:
            c_dir = dataset_root / c / "samples"
            for i in range(8):
                s = c_dir / f"s_{i}"
                s.mkdir(parents=True)
                vec = np.ones((1, 206), dtype=np.float32) * (1.0 if c == "أهلا" else -1.0)
                vec += np.random.normal(0, 0.05, (1, 206)).astype(np.float32)
                np.save(s / "landmarks.npy", vec)
                with open(s / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump({"schema_version": "v2_multimodal", "feature_dim": 206}, f)

        results = train_multimodal_model(
            dataset_dir=tmp_path,
            output_dir=models_root,
            model_type="mlp",
            epochs=20,
        )

        assert results["accuracy"] > 0.7
        assert (models_root / "static_model.pkl").exists()
        assert (models_root / "classes.json").exists()
        assert (models_root / "metadata.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Multimodal Sign Recognizer Inference
# ─────────────────────────────────────────────────────────────────────────────

class TestMultimodalSignRecognizer:
    """Verify recognizer inference, confidence thresholds, and untrained state."""

    def test_untrained_model_reports_clean_status(self, tmp_path: Path) -> None:
        empty_model_dir = tmp_path / "empty_models"
        empty_model_dir.mkdir(parents=True)

        recognizer = MultimodalSignRecognizer(model_dir=empty_model_dir)
        assert not recognizer.is_trained

        frame = create_synthetic_multimodal_frame(has_left=True, has_right=True, has_face=True)
        pred = recognizer.predict(frame)

        assert not pred.is_valid

    def test_trained_model_inference_flow(self, tmp_path: Path) -> None:
        models_root = tmp_path / "models"
        dataset_root = tmp_path / "words"

        for c_idx, c in enumerate(["نعم", "لا"]):
            c_dir = dataset_root / c / "samples"
            for i in range(8):
                s = c_dir / f"s_{i}"
                s.mkdir(parents=True)
                vec = np.zeros((1, 206), dtype=np.float32)
                vec[0, c_idx] = 10.0
                np.save(s / "landmarks.npy", vec)
                with open(s / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump({"schema_version": "v2_multimodal", "feature_dim": 206}, f)

        train_multimodal_model(dataset_dir=tmp_path, output_dir=models_root, model_type="rf")

        recognizer = MultimodalSignRecognizer(model_dir=models_root, confidence_threshold=0.2)
        assert recognizer.is_trained
        assert len(recognizer.classes) == 2

        frame = create_synthetic_multimodal_frame(has_left=False, has_right=True, has_face=True)
        recognizer._extractor.extract = MagicMock(return_value=np.array([10.0] + [0.0] * 205, dtype=np.float32))

        pred = recognizer.predict(frame)
        assert pred.is_valid
        assert pred.arabic_text == "نعم"
        assert pred.confidence > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 8. UI Integration (Dataset Studio, Recognition, Training Tabs)
# ─────────────────────────────────────────────────────────────────────────────

class TestMultimodalUITabs:
    """Verify PySide6 desktop tabs support multimodal controls and indicators."""

    def test_dataset_tab_multimodal_controls(self, qapp) -> None:
        state = AppState()
        use_cases = VideoSourceUseCases(state)
        tab = DatasetCollectionTab(state, use_cases)

        # Modality selector present
        assert hasattr(tab, "_modality_combo")
        modes = [tab._modality_combo.itemData(i) for i in range(tab._modality_combo.count())]
        assert "two_hands_face" in modes
        assert "two_hands" in modes
        assert "two_hands_face_pose" in modes
        assert "one_hand" in modes

        # Telemetry cards present
        assert hasattr(tab, "_lh_card")
        assert hasattr(tab, "_rh_card")
        assert hasattr(tab, "_face_card")
        assert hasattr(tab, "_pose_card")

        # Modality change updates configuration
        tab._modality_combo.setCurrentIndex(1)  # Two Hands Only
        assert tab._mm_config.mode == ModalityMode.TWO_HANDS

        tab.close()

    def test_recognition_tab_engine_selector_and_multimodal_status(self, qapp) -> None:
        state = AppState()
        use_cases = VideoSourceUseCases(state)
        tab = RecognitionTab(state, use_cases)

        # Verify Multimodal engine exists in selector
        engine_keys = [tab._engine_selector.itemData(i) for i in range(tab._engine_selector.count())]
        assert "multimodal" in engine_keys
        assert "karsl" in engine_keys
        assert "custom" in engine_keys

        # Selecting multimodal when untrained updates worker engine
        idx = engine_keys.index("multimodal")
        tab._engine_selector.setCurrentIndex(idx)
        assert tab._worker.active_engine == "multimodal"

        tab._worker.stop()
        tab._worker.wait(1000)
        tab.close()

    def test_training_tab_multimodal_model_option(self, qapp) -> None:
        state = AppState()
        tab = ModelTrainingTab(state)

        # Verify Multimodal option exists
        texts = [tab._model_type.itemText(i) for i in range(tab._model_type.count())]
        assert any("Multimodal" in t for t in texts)

        tab.close()
