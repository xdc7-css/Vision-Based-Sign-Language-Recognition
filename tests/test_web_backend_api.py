"""Tests for ArSL Studio FastAPI backend and WebSockets."""

import json
import pytest
import numpy as np
import cv2
from starlette.testclient import TestClient

from backend.main import app, dataset_service, recognition_service, training_service


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_system_status(client):
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "cpu_percent" in data
    assert "memory_percent" in data
    assert "active_model" in data
    assert data["total_gestures"] > 0


def test_api_gestures_crud(client):
    import uuid
    uid = uuid.uuid4().hex[:6]
    test_sign_name = f"اختبار_{uid}"
    expected_display_name = f"اختبار {uid}"

    # 1. List gestures
    res = client.get("/api/gestures")
    assert res.status_code == 200
    initial_gestures = res.json()
    assert isinstance(initial_gestures, list)

    # 2. Create new gesture
    create_res = client.post("/api/gestures", json={
        "name": test_sign_name,
        "category": "static",
        "description": "Web automated test sign",
    })
    assert create_res.status_code == 201
    created_data = create_res.json()
    gesture_id = created_data["id"]
    assert created_data["name"] == expected_display_name

    # 3. Retrieve by ID
    get_res = client.get(f"/api/gestures/{gesture_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == expected_display_name

    # 4. Rename gesture
    new_sign_name = f"تحديث_{uid}"
    expected_new_display = f"تحديث {uid}"
    patch_res = client.patch(f"/api/gestures/{gesture_id}", json={
        "name": new_sign_name,
    })
    assert patch_res.status_code == 200
    assert patch_res.json()["name"] == expected_new_display

    # 5. Archive and restore
    arc_res = client.post(f"/api/gestures/{gesture_id}/archive")
    assert arc_res.status_code == 200
    assert arc_res.json() == {"success": True}

    rest_res = client.post(f"/api/gestures/{gesture_id}/restore")
    assert rest_res.status_code == 200
    assert rest_res.json() == {"success": True}

    # 6. Delete gesture
    del_res = client.delete(f"/api/gestures/{gesture_id}?purge_samples=true")
    assert del_res.status_code == 200
    assert del_res.json() == {"success": True}

    # 7. Verify deletion
    get_after = client.get(f"/api/gestures/{gesture_id}")
    assert get_after.status_code == 404

    # Clean up test ID from deleted_ids set to keep vocabulary clean
    from backend.services.dataset_service import DatasetService
    _ds = DatasetService()
    _ds.registry._deleted_ids.discard(gesture_id)
    _ds.registry.save()


def test_api_dataset_summary(client):
    res = client.get("/api/dataset/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_classes"] > 0
    assert data["total_samples"] >= 0
    assert data["multimodal_samples"] >= 0
    assert isinstance(data["classes"], list)


def test_api_models_active(client):
    res = client.get("/api/models/active")
    assert res.status_code == 200
    data = res.json()
    assert "name" in data
    assert "model_type" in data
    assert isinstance(data["classes"], list)
    assert data["num_classes"] >= 0


def test_api_training_readiness(client):
    res = client.get("/api/training/readiness")
    assert res.status_code == 200
    data = res.json()
    assert "is_trainable" in data
    assert "trainable_classes_count" in data
    assert "model_status" in data
    assert data["model_status"] in ("Ready", "Outdated", "Untrained")


def test_api_sentence_builder(client):
    # Clear initial
    client.post("/api/sentence/clear")

    # Add word
    res1 = client.post("/api/sentence/word", json={"word": "السلام"})
    assert res1.status_code == 200
    assert "السلام" in res1.json()["sentence"]

    # Add second word
    res2 = client.post("/api/sentence/word", json={"word": "عليكم"})
    assert res2.status_code == 200
    assert "السلام" in res2.json()["sentence"]
    assert "عليكم" in res2.json()["sentence"]

    # Delete last word
    res3 = client.post("/api/sentence/delete")
    assert res3.status_code == 200
    assert "عليكم" not in res3.json()["sentence"]

    # Clear
    res4 = client.post("/api/sentence/clear")
    assert res4.status_code == 200
    assert res4.json()["sentence"] == ""


def test_websocket_recognition_stream(client):
    # Create synthetic RGB frame encoded as JPEG
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple bright pattern simulating hand/face center
    cv2.circle(img, (320, 240), 50, (200, 200, 200), -1)
    _, buffer = cv2.imencode(".jpg", img)
    jpeg_bytes = buffer.tobytes()

    with client.websocket_connect("/ws/recognition") as ws:
        # Send raw frame bytes
        ws.send_bytes(jpeg_bytes)
        # Receive inference response
        data = ws.receive_json()
        assert "recognized_sign" in data
        assert "confidence" in data
        assert "latency_ms" in data
        assert "fps" in data
        assert "landmarks" in data
        assert "indicators" in data
        assert "sentence" in data


def test_websocket_capture_stream(client):
    with client.websocket_connect("/ws/capture") as ws:
        # Send setup configuration
        ws.send_text(json.dumps({
            "class_name": "اختبار_التقاط",
            "target_count": 5,
            "category": "static",
        }))
        res = ws.receive_json()
        assert res["status"] == "ready"
        assert res["class_name"] == "اختبار_التقاط"
        assert res["target_count"] == 5

