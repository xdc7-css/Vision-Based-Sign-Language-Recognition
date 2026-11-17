"""Tests for KArSL-502 Research Pipeline (SVM vs 1D Temporal CNN).

Validates:
1. Dataset manager, metadata, LOSO splits, and feature matrices.
2. Preprocessing transformations (sequence resampling, SVM feature engineering, CNN channel transposition).
3. SVM pipeline training, serialization, and inference.
4. 1D Temporal CNN pipeline training, serialization, and inference.
5. Scientific evaluator metrics calculation, CSV/JSON reporting, and winner declaration.
6. FastAPI research API endpoints and model switcher.
7. Real-time ResearchSignRecognizer streaming and prediction.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from starlette.testclient import TestClient

from backend.main import app
from src.research.cnn_pipeline import ResearchCNNPipeline, TemporalSignCNN
from src.research.evaluator import ResearchScientificEvaluator
from src.research.karsl_dataset import KArSLDatasetManager
from src.research.preprocessing import ResearchFeatureExtractor
from src.research.research_recognizer import ResearchSignRecognizer
from src.research.svm_pipeline import ResearchSVMPipeline


@pytest.fixture(scope="module")
def dataset_manager() -> KArSLDatasetManager:
    return KArSLDatasetManager()


def test_karsl_dataset_status_and_classes(dataset_manager: KArSLDatasetManager) -> None:
    status = dataset_manager.get_status()
    assert status["is_prepared"] is True
    assert status["is_authentic"] is True
    assert status["num_classes"] == 10
    assert len(status["signers"]) == 3
    assert "Signer01" in status["signers"]
    assert "Signer02" in status["signers"]
    assert "Signer03" in status["signers"]
    assert status["total_samples"] == 240
    assert status["feature_dim"] == 736
    assert status["seq_len"] == 30


def test_karsl_loso_splits(dataset_manager: KArSLDatasetManager) -> None:
    splits = dataset_manager.load_loso_splits()
    assert "test_Signer03" in splits
    s03 = splits["test_Signer03"]
    assert s03.get("test_signer", s03.get("held_out_signer")) == "Signer03"
    assert s03["train_signers"] == ["Signer01", "Signer02"]
    assert len(s03["train_indices"]) == 160
    assert len(s03["test_indices"]) == 80

    # Ensure zero data leakage: train and test indices have empty intersection
    assert set(s03["train_indices"]).isdisjoint(set(s03["test_indices"]))


def test_karsl_feature_loading(dataset_manager: KArSLDatasetManager) -> None:
    X, y, signers = dataset_manager.load_features()
    assert X.shape == (240, 30, 736)
    assert len(y) == 240
    assert len(signers) == 240
    assert len(np.unique(y)) == 10


def test_preprocessing_transformations() -> None:
    extractor = ResearchFeatureExtractor(seq_len=30, feat_dim=736)

    # 1. Sequence resampling test
    buffer_frames = [np.random.randn(736).astype(np.float32) for _ in range(22)]
    resampled = extractor.resample_sequence(buffer_frames, target_len=30)
    assert resampled.shape == (30, 736)

    # 2. Sequence to SVM feature aggregation (mean, std, displacement)
    batch_seq = np.random.randn(5, 30, 736).astype(np.float32)
    scaled_feats, _ = extractor.prepare_svm_dataset(batch_seq, fit_scaler=True)
    assert scaled_feats.shape == (5, 736 * 3)  # 2,208 features

    # 3. CNN tensor preparation: transpose from (N, T, C) -> (N, C, T)
    cnn_feats = extractor.prepare_cnn_dataset(batch_seq)
    assert cnn_feats.shape == (5, 736, 30)


def test_svm_pipeline_training_and_inference() -> None:
    pipeline = ResearchSVMPipeline()
    assert pipeline.is_trained() is True
    pipeline.load_model()

    # Dummy test sample (30, 736)
    dummy_seq = np.random.randn(30, 736).astype(np.float32)
    pred_idx, conf, name_ar, name_en = pipeline.predict_sequence(dummy_seq)
    assert isinstance(pred_idx, int)
    assert 0 <= pred_idx < 10
    assert 0.0 <= conf <= 1.0
    assert len(name_ar) > 0


def test_cnn_pipeline_architecture_and_inference() -> None:
    model = TemporalSignCNN(num_classes=10, in_channels=736)
    assert model is not None

    pipeline = ResearchCNNPipeline()
    assert pipeline.is_trained() is True
    pipeline.load_model()

    dummy_seq = np.random.randn(30, 736).astype(np.float32)
    pred_idx, conf, name_ar, name_en = pipeline.predict_sequence(dummy_seq)
    assert isinstance(pred_idx, int)
    assert 0 <= pred_idx < 10
    assert 0.0 <= conf <= 1.0
    assert len(name_ar) > 0


def test_scientific_evaluator_reports(dataset_manager: KArSLDatasetManager) -> None:
    evaluator = ResearchScientificEvaluator(dataset_manager=dataset_manager)
    comp = evaluator.get_latest_comparison()
    assert comp is not None

    assert comp["protocol"] == "Leave-One-Signer-Out (LOSO)"
    assert comp.get("test_signer", comp.get("held_out_signer")) == "Signer03"
    assert comp["train_samples"] == 160
    assert comp["test_samples"] == 80
    assert comp["num_classes"] == 10

    # Check metrics existence
    for model_key in ("svm", "cnn"):
        m = comp[model_key]
        assert "accuracy" in m
        assert "f1_macro" in m
        assert "precision_macro" in m
        assert "recall_macro" in m
        assert "training_time_sec" in m
        assert "inference_latency_ms" in m
        assert "model_size_kb" in m
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1_macro"] <= 1.0

    # Validate winner declaration
    assert comp["winner"]["best_model"] in ("1D Temporal CNN", "Support Vector Machine (SVM)")
    assert "rationale" in comp["winner"]

    # Validate generated CSV and confusion matrices files
    csv_file = evaluator.eval_dir / "comparison_metrics.csv"
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    cm_file = evaluator.eval_dir / "confusion_matrices.json"
    assert cm_file.exists()
    with open(cm_file, "r", encoding="utf-8") as f:
        cm_data = json.load(f)
        assert len(cm_data["svm_confusion_matrix"]) == 10
        assert len(cm_data["cnn_confusion_matrix"]) == 10


def test_fastapi_research_endpoints() -> None:
    client = TestClient(app)

    # 1. GET /api/research/status
    res = client.get("/api/research/status")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset"]["is_prepared"] is True
    assert data["models"]["svm"]["is_trained"] is True
    assert data["models"]["cnn"]["is_trained"] is True
    assert data["comparison"] is not None

    # 2. GET /api/research/comparison
    res_comp = client.get("/api/research/comparison")
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    assert comp_data["winner"]["best_model"] in ("1D Temporal CNN", "Support Vector Machine (SVM)")

    # 3. GET /api/models/available
    res_models = client.get("/api/models/available")
    assert res_models.status_code == 200
    models_list = res_models.json()
    model_ids = {m["id"] for m in models_list}
    assert "multimodal" in model_ids
    assert "research_svm" in model_ids
    assert "research_cnn" in model_ids

    # 4. POST /api/models/select (switch to research_cnn)
    res_sel_cnn = client.post("/api/models/select", json={"model_id": "research_cnn"})
    assert res_sel_cnn.status_code == 200
    assert res_sel_cnn.json()["success"] is True

    # Verify active model is now research_cnn
    res_active = client.get("/api/models/active")
    assert res_active.status_code == 200
    assert res_active.json()["id"] == "research_cnn"

    # Switch back to multimodal
    res_sel_mm = client.post("/api/models/select", json={"model_id": "multimodal"})
    assert res_sel_mm.status_code == 200
    assert res_sel_mm.json()["success"] is True


def test_live_research_recognizer() -> None:
    svm_rec = ResearchSignRecognizer(model_type="svm")
    assert svm_rec.is_loaded is True
    assert len(svm_rec.classes) == 10

    cnn_rec = ResearchSignRecognizer(model_type="cnn")
    assert cnn_rec.is_loaded is True
    assert len(cnn_rec.classes) == 10

    # Provide synthetic 480x640 frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = cnn_rec.process_frame(dummy_frame)
    # Since dummy black frame has no landmarks, recognizer handles it safely (returns None or low conf)
    assert res is None or res.confidence >= 0.0
