"""WebSocket streaming endpoints for real-time recognition and sample capture."""

from __future__ import annotations

import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(tags=["streaming"])


def get_recognition_service():
    from backend.main import recognition_service
    return recognition_service


@router.websocket("/ws/recognition")
async def ws_recognition(websocket: WebSocket) -> None:
    """Real-time sign language recognition stream.
    
    Accepts binary JPEG bytes from browser webcam; returns JSON inference results and landmarks.
    """
    await websocket.accept()
    rec_svc = get_recognition_service()
    
    try:
        while True:
            # Receive raw frame bytes (JPEG)
            data = await websocket.receive_bytes()
            if not data:
                continue

            # Decode JPEG in-memory
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Process frame through multimodal detector & recognizer
            result = rec_svc.process_frame(frame)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.debug("Recognition WebSocket client disconnected")
    except Exception as e:
        logger.warning(f"Error in recognition stream: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/ws/capture")
async def ws_capture(websocket: WebSocket) -> None:
    """Sample recording stream.
    
    Client connects, sends config message: {"class_name": "...", "target_count": 10, "category": "static"}.
    Then streams binary frames. Server detects valid multimodal landmarks, writes sample to disk,
    and returns progress updates until target_count is reached.
    """
    await websocket.accept()
    rec_svc = get_recognition_service()

    class_name = ""
    target_count = 10
    category = "static"
    saved_count = 0

    try:
        while True:
            message = await websocket.receive()
            
            # Text message: configuration or control
            if "text" in message:
                try:
                    cfg = json.loads(message["text"])
                    if "class_name" in cfg:
                        class_name = cfg["class_name"]
                    if "target_count" in cfg:
                        target_count = int(cfg["target_count"])
                    if "category" in cfg:
                        category = cfg["category"]
                    if cfg.get("command") == "reset":
                        saved_count = 0
                    await websocket.send_json({
                        "status": "ready",
                        "class_name": class_name,
                        "target_count": target_count,
                        "saved_count": saved_count,
                    })
                except Exception as e:
                    await websocket.send_json({"status": "error", "message": str(e)})

            # Binary message: video frame to capture
            elif "bytes" in message and class_name:
                data = message["bytes"]
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Attempt to extract and persist sample
                sample_info = rec_svc.record_multimodal_sample(class_name, frame, category=category)
                if sample_info is not None:
                    saved_count += 1
                    is_complete = saved_count >= target_count
                    await websocket.send_json({
                        "status": "complete" if is_complete else "sample_saved",
                        "sample_id": sample_info["sample_id"],
                        "current_count": saved_count,
                        "target_count": target_count,
                        "class_name": class_name,
                        "has_left": sample_info["has_left"],
                        "has_right": sample_info["has_right"],
                        "has_face": sample_info["has_face"],
                    })
                    if is_complete:
                        # Reset for next recording session
                        saved_count = 0
                else:
                    await websocket.send_json({
                        "status": "invalid_frame",
                        "message": "Hand not clearly detected in frame",
                    })

    except WebSocketDisconnect:
        logger.debug("Capture WebSocket client disconnected")
    except Exception as e:
        logger.warning(f"Error in capture stream: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

