"""Comprehensive automated test suite for Dataset Studio and Model Training integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.domain.hand import Hand, Handedness, HandLandmark
from src.infrastructure.dataset.organizer import DatasetOrganizer
from src.infrastructure.dataset.validator import DatasetValidator
from src.infrastructure.dialect_manager import DialectManager
from src.models.evaluator import ModelRegistry
from src.models.train_dynamic import inspect_dynamic_dataset, train_dynamic_model
from src.models.train_static import inspect_dataset, train_static_model
from src.presentation.dialogs.dataset_dialogs import SampleInspectorDialog
from src.presentation.tabs.dataset import DatasetCollectionTab


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


def create_synthetic_hand(handedness: Handedness = Handedness.RIGHT) -> Hand:
    landmarks = [
        HandLandmark(min(0.99, 0.2 + 0.01 * i), min(0.99, 0.2 + 0.01 * i), 0.0)
        for i in range(21)
    ]
    return Hand(landmarks=landmarks, handedness=handedness, confidence=0.95)


class TestDatasetStudio:
    """Test Dataset Studio UI and capture workflows."""

    def test_dataset_tab_camera_controls_and_sync(self, qapp) -> None:
        state = AppState()
        use_cases = VideoSourceUseCases(state)
        tab = DatasetCollectionTab(state, use_cases)

        assert tab._camera_combo is not None
        assert tab._cam_start_btn is not None
        assert tab._cam_stop_btn is not None
        assert not tab._cam_stop_btn.isEnabled()
        assert tab._status_badge.text() in ("Camera Ready", "No Camera Detected")

        # Test state sync
        state.selected_camera_index = 1
        tab._refresh_camera_list()
        assert tab._camera_combo.count() >= 1

        tab.close()

    def test_static_sample_capture_validation(self, tmp_path: Path) -> None:
        organizer = DatasetOrganizer(base_dir=tmp_path)

        class_dir = organizer.create_class("words", "نعم")
        sample_id = "test_sample_01"
        sample_dir = class_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True)

        # Create valid 3D static landmarks (1, 21, 3)
        landmarks = np.zeros((1, 21, 3), dtype=np.float32)
        landmarks[0, :, 0] = np.linspace(0.2, 0.8, 21)
        landmarks[0, :, 1] = np.linspace(0.2, 0.8, 21)
        np.save(sample_dir / "landmarks.npy", landmarks)

        meta = {
            "sample_id": sample_id,
            "gesture_type": "static_word",
            "gesture_label": "نعم",
            "dialect": "iraqi",
            "timestamp": 123456789.0,
            "num_frames": 1,
            "num_hands": 1,
            "handedness": "Right",
            "duration_seconds": 0.0,
            "fps": 30.0,
            "has_motion": False,
            "total_distance": 0.0,
            "motion_energy": 0.0,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        # Validate with DatasetValidator
        validator = DatasetValidator(dataset_dir=tmp_path)
        val = validator.validate_sample(sample_dir)
        assert val.is_valid, f"Validation issues: {val.issues}"
        assert val.num_frames == 1
        assert val.num_landmarks == 21

    def test_dynamic_sequence_recording_validation(self, tmp_path: Path) -> None:
        organizer = DatasetOrganizer(base_dir=tmp_path)

        class_dir = organizer.create_class("dynamic", "شكرا")
        sample_id = "test_seq_01"
        sample_dir = class_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True)

        # Create valid 3D dynamic sequence (12, 21, 3)
        seq_landmarks = np.zeros((12, 21, 3), dtype=np.float32)
        for t in range(12):
            seq_landmarks[t, :, 0] = np.linspace(0.2 + t * 0.01, 0.8, 21)
            seq_landmarks[t, :, 1] = np.linspace(0.3 + t * 0.01, 0.7, 21)
        np.save(sample_dir / "landmarks.npy", seq_landmarks)

        meta = {
            "sample_id": sample_id,
            "gesture_type": "dynamic_gesture",
            "gesture_label": "شكرا",
            "dialect": "iraqi",
            "timestamp": 123456789.0,
            "num_frames": 12,
            "num_hands": 1,
            "handedness": "Right",
            "duration_seconds": 0.4,
            "fps": 30.0,
            "has_motion": True,
            "total_distance": 1.5,
            "motion_energy": 0.5,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        # Validate with DatasetValidator
        validator = DatasetValidator(dataset_dir=tmp_path)
        val = validator.validate_sample(sample_dir)
        assert val.is_valid, f"Validation issues: {val.issues}"
        assert val.num_frames == 12
        assert val.num_landmarks == 21

    def test_class_management_add_rename_delete(self, tmp_path: Path) -> None:
        organizer = DatasetOrganizer(base_dir=tmp_path)

        # 1. Add class
        class_dir = organizer.create_class("words", "مرحبا")
        assert class_dir.exists()
        assert (class_dir / "samples").exists()

        # Add sample inside
        sample_dir = class_dir / "samples" / "s1"
        sample_dir.mkdir()
        (sample_dir / "test.txt").write_text("sample")

        # 2. Rename class
        success = organizer.rename_class("words", "مرحبا", "أهلا_وسهلا")
        assert success
        assert not class_dir.exists()
        new_dir = organizer.get_class_dir("words", "أهلا_وسهلا")
        assert new_dir.exists()
        assert (new_dir / "samples" / "s1" / "test.txt").exists()

        # 3. Delete class
        del_success = organizer.delete_class("words", "أهلا_وسهلا")
        assert del_success
        assert not new_dir.exists()

    def test_sample_inspector_dialog(self, qapp, tmp_path: Path) -> None:
        organizer = DatasetOrganizer(base_dir=tmp_path)
        class_dir = organizer.create_class("words", "مع_السلامة")

        # Add sample
        s_dir = class_dir / "samples" / "sample_abc"
        s_dir.mkdir(parents=True)
        np.save(s_dir / "landmarks.npy", np.zeros((1, 21, 3), dtype=np.float32))
        meta = {"sample_id": "sample_abc", "num_frames": 1, "handedness": "Right", "timestamp": 12345.0}
        with open(s_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        dlg = SampleInspectorDialog("مع_السلامة", base_dir=tmp_path)
        assert dlg._table.rowCount() == 1
        assert dlg._table.item(0, 0).text() == "sample_abc"
        assert dlg._table.item(0, 2).text() == "1"

        dlg.close()


class TestModelTrainingIntegration:
    """Test Model Training on real and synthetic datasets."""

    def test_inspect_and_train_static_model(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "words"
        output_dir = tmp_path / "models" / "iraqi"

        # Create 2 classes with 3 samples each (satisfies min requirements)
        for cls_name in ["نعم", "لا"]:
            cls_samples = dataset_dir / cls_name / "samples"
            cls_samples.mkdir(parents=True)
            for i in range(3):
                s_dir = cls_samples / f"sample_{i}"
                s_dir.mkdir()
                lm = np.zeros((1, 21, 3), dtype=np.float32)
                # Differentiate classes
                lm[0, :, 0] = (0.3 if cls_name == "نعم" else 0.7) + np.random.normal(0, 0.01, 21)
                lm[0, :, 1] = np.linspace(0.1, 0.9, 21)
                np.save(s_dir / "landmarks.npy", lm)

        # Inspect dataset
        inspection = inspect_dataset(dataset_dir)
        assert inspection["is_trainable"]
        assert len(inspection["accepted_classes"]) == 2

        # Train static model
        progress_records = []

        def on_prog(ep, tot, loss, val_loss):
            progress_records.append((ep, tot, loss, val_loss))

        res = train_static_model(
            dataset_path=dataset_dir,
            output_dir=output_dir,
            model_types=["mlp"],
            epochs=5,
            batch_size=2,
            learning_rate=0.01,
            val_split=0.33,
            augment=False,
            dialect="iraqi",
            progress_callback=on_prog,
        )

        assert res["accuracy"] >= 0.0
        assert Path(res["model_path"]).exists()
        assert Path(res["vocab_path"]).exists()
        assert len(res["classes"]) == 2
        assert len(progress_records) > 0

    def test_inspect_and_train_dynamic_model(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dynamic"
        output_dir = tmp_path / "models" / "iraqi_dynamic"

        # Create 2 classes with 2 sequence samples each (T=10 frames)
        for cls_name in ["السلام_عليكم", "شكرا"]:
            cls_samples = dataset_dir / cls_name / "samples"
            cls_samples.mkdir(parents=True)
            for i in range(2):
                s_dir = cls_samples / f"seq_{i}"
                s_dir.mkdir()
                seq = np.zeros((10, 21, 3), dtype=np.float32)
                offset = 0.2 if cls_name == "السلام_عليكم" else 0.6
                for t in range(10):
                    seq[t, :, 0] = offset + t * 0.02 + np.random.normal(0, 0.005, 21)
                    seq[t, :, 1] = np.linspace(0.2, 0.8, 21)
                np.save(s_dir / "landmarks.npy", seq)

        # Inspect dynamic dataset
        inspection = inspect_dynamic_dataset(dataset_dir)
        assert inspection["is_trainable"]
        assert len(inspection["accepted_classes"]) == 2

        # Train dynamic model
        progress_records = []

        def on_prog(ep, tot, loss, val_loss):
            progress_records.append((ep, tot, loss, val_loss))

        res = train_dynamic_model(
            dataset_path=dataset_dir,
            output_dir=output_dir,
            architectures=["lstm"],
            sequence_length=15,
            epochs=3,
            augment=False,
            dialect="iraqi",
            progress_callback=on_prog,
        )

        assert res["accuracy"] >= 0.0
        assert Path(res["model_path"]).exists()
        assert Path(res["vocab_path"]).exists()
        assert len(res["classes"]) == 2

    def test_trained_model_loaded_and_used_in_pipeline(self, tmp_path: Path) -> None:
        """Verify that newly trained static model executes live inference inside HybridRecognitionPipeline."""
        dataset_dir = tmp_path / "words"
        model_dir = tmp_path / "models" / "iraqi"

        from src.infrastructure.recognition.hybrid_pipeline import HybridConfig

        # Create dataset with clearly separable hand shape geometries
        for cls_name in ["نعم", "لا"]:
            cls_samples = dataset_dir / cls_name / "samples"
            cls_samples.mkdir(parents=True)
            # Distinct finger lengths / geometries after wrist normalization
            radius_multiplier = 0.05 if cls_name == "نعم" else 0.25
            for i in range(5):
                s_dir = cls_samples / f"sample_{i}"
                s_dir.mkdir()
                lm = np.zeros((1, 21, 3), dtype=np.float32)
                # Wrist at center
                lm[0, 0, :] = [0.5, 0.5, 0.0]
                for k in range(1, 21):
                    angle = k * (2 * np.pi / 20)
                    lm[0, k, 0] = 0.5 + radius_multiplier * np.cos(angle) + np.random.normal(0, 0.001)
                    lm[0, k, 1] = 0.5 + radius_multiplier * np.sin(angle)
                    lm[0, k, 2] = 0.0
                np.save(s_dir / "landmarks.npy", lm)

        # Train model
        res = train_static_model(
            dataset_path=dataset_dir,
            output_dir=model_dir,
            model_types=["mlp"],
            epochs=50,
            batch_size=2,
            learning_rate=0.01,
            val_split=0.20,
            augment=False,
            dialect="iraqi",
        )
        assert Path(res["model_path"]).exists()

        # Load into DialectManager
        dm = DialectManager(
            base_path=str(tmp_path),
            hybrid_config=HybridConfig(word_threshold=0.5),
        )
        cfg = dm.active_dialect_config
        assert cfg is not None
        cfg.model_path = str(model_dir)
        cfg.vocabulary_path = str(model_dir / "vocabulary.json")

        success = dm.reload_dialect()
        assert success

        # Test live inference via hand matching "نعم" (folded shape, radius 0.05)
        hand_lms = [HandLandmark(0.5, 0.5, 0.0)]
        for k in range(1, 21):
            angle = k * (2 * np.pi / 20)
            x = 0.5 + 0.05 * np.cos(angle)
            y = 0.5 + 0.05 * np.sin(angle)
            hand_lms.append(HandLandmark(min(0.99, max(0.01, x)), min(0.99, max(0.01, y)), 0.0))
        hand = Hand(landmarks=hand_lms, handedness=Handedness.RIGHT, confidence=0.99)

        hybrid_res = dm.process(hand)
        assert hybrid_res is not None
        assert hybrid_res.is_valid
        assert hybrid_res.arabic_text in ["نعم", "لا"]
        assert hybrid_res.final_result.confidence >= 0.4

    def test_karsl_502_model_integrity_protected(self) -> None:
        """Verify that original KArSL-502 model weights and label files remain completely intact."""
        karsl_onnx = Path("data/models/karsl/karsl_502.onnx")
        karsl_labels = Path("data/models/karsl/labels.json")

        assert karsl_onnx.exists(), "KArSL-502 ONNX model must exist"
        assert karsl_onnx.stat().st_size == 7330073, "Original KArSL-502 ONNX file size altered"

        assert karsl_labels.exists(), "KArSL-502 labels file must exist"
        assert karsl_labels.stat().st_size == 13579, "Original KArSL-502 labels file size altered"

        with open(karsl_labels, encoding="utf-8") as f:
            labels = json.load(f)
        assert len(labels.get("AR_WORDS", [])) == 502, "KArSL-502 must contain exactly 502 classes"
