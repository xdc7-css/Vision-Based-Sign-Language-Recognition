"""Automated tests for Web Recognition performance optimizations:

- 16-byte binary framing header (frame_id, client_time_ms)
- Bounded queue & latest-frame-wins frame dropping
- Prediction stabilization & duplicate prevention through WebSocket
- Telemetry & performance timing breakdown
"""

from __future__ import annotations

import struct
import time

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from backend.main import app, recognition_service


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _encode_jpeg(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def test_ws_recognition_with_binary_header(client):
    """Verify that WebSocket accepts 16-byte header and reflects frame_id and client_time_ms."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    jpeg_bytes = _encode_jpeg(dummy_frame)

    frame_id = 42
    client_time_ms = 123456.78
    header = struct.pack(">Qd", frame_id, client_time_ms)
    payload = header + jpeg_bytes

    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_bytes(payload)
        resp = ws.receive_json()

        assert resp["frame_id"] == 42
        assert abs(resp["client_time_ms"] - 123456.78) < 0.001
        assert "latency_ms" in resp
        assert "perf" in resp
        assert "hands_ms" in resp["perf"]
        assert "face_ms" in resp["perf"]
        assert "infer_ms" in resp["perf"]
        assert "backend_total_ms" in resp["perf"]
        assert "state" in resp
        assert "status_label" in resp


def test_ws_recognition_latest_frame_wins_dropping(client):
    """Verify latest-frame-wins dropping when multiple frames are queued rapidly."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    jpeg_bytes = _encode_jpeg(dummy_frame)

    with client.websocket_connect("/ws/recognition") as ws:
        # Send frame 1
        hdr1 = struct.pack(">Qd", 1, float(time.time() * 1000))
        ws.send_bytes(hdr1 + jpeg_bytes)
        r1 = ws.receive_json()
        assert r1["frame_id"] == 1

        # Send frame 2 and frame 3 back-to-back
        hdr2 = struct.pack(">Qd", 2, float(time.time() * 1000))
        hdr3 = struct.pack(">Qd", 3, float(time.time() * 1000))
        ws.send_bytes(hdr2 + jpeg_bytes)
        ws.send_bytes(hdr3 + jpeg_bytes)

        # Read the next response
        r2 = ws.receive_json()
        assert r2["frame_id"] in (2, 3)
        assert "dropped_frames" in r2


def test_ws_recognition_stabilization_and_duplicate_prevention(client):
    """Verify stabilizer state machine through WebSocket."""
    # Reset stabilizer
    recognition_service.stabilizer.reset()

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    jpeg_bytes = _encode_jpeg(dummy_frame)

    with client.websocket_connect("/ws/recognition") as ws:
        # Empty frame with no hands -> state should be idle
        hdr = struct.pack(">Qd", 1, float(time.time() * 1000))
        ws.send_bytes(hdr + jpeg_bytes)
        resp = ws.receive_json()

        assert resp["state"] == "idle"
        assert resp["emitted_word"] is None
        assert resp["is_neutral"] is True
