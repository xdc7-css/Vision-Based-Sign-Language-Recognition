"""WebSocket streaming endpoints for real-time recognition and sample capture."""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time

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
    
    Accepts binary JPEG bytes (optionally with 16-byte header: uint64 frame_id, float64 client_time_ms).
    Uses latest-frame-wins bounded queue with non-blocking worker thread.
    Returns JSON inference results, state machine progress, and detailed timing telemetry.
    """
    await websocket.accept()
    rec_svc = get_recognition_service()
    
    # Bounded queue of maxsize=1 enforces strict latest-frame-wins backpressure
    frame_queue: asyncio.Queue[tuple[int, float, bytes]] = asyncio.Queue(maxsize=1)
    dropped_frames = 0

    async def consumer_loop():
        nonlocal dropped_frames
        while True:
            frame_id, client_time_ms, raw_bytes = await frame_queue.get()
            try:
                t_dec = time.perf_counter()
                nparr = np.frombuffer(raw_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                decode_ms = (time.perf_counter() - t_dec) * 1000.0

                if frame is not None:
                    # Execute CPU-intensive computer vision pipeline in worker thread pool
                    result = await asyncio.to_thread(
                        rec_svc.process_frame,
                        frame,
                        frame_id,
                        client_time_ms,
                        decode_ms,
                    )
                    result["dropped_frames"] = dropped_frames
                    await websocket.send_json(result)
            except Exception as ex:
                logger.debug(f"Frame processing error: {ex}")
            finally:
                frame_queue.task_done()

    consumer_task = asyncio.create_task(consumer_loop())

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            # Parse optional 16-byte header: uint64 frame_id, float64 client_time_ms
            frame_id = 0
            client_time_ms = 0.0
            if len(data) > 16 and not (data[0] == 0xFF and data[1] == 0xD8 and data[2] == 0xFF):
                try:
                    frame_id, client_time_ms = struct.unpack(">Qd", data[:16])
                    jpeg_bytes = data[16:]
                except Exception:
                    jpeg_bytes = data
            else:
                jpeg_bytes = data

            # Latest-frame-wins policy: drop previous frame if still queued
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                    dropped_frames += 1
                except (asyncio.QueueEmpty, ValueError):
                    pass

            await frame_queue.put((frame_id, client_time_ms, jpeg_bytes))

    except WebSocketDisconnect:
        logger.debug("Recognition WebSocket client disconnected")
    except Exception as e:
        logger.warning(f"Error in recognition stream: {e}")
    finally:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
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

