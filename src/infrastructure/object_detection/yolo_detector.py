from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class DetectedObject:
    """Represents a single object detected by YOLO."""
    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates


@dataclass
class YOLOConfig:
    """Configuration for YOLO object detection."""
    model_path: str = "data/models/yolo/yolov8n.pt"
    onnx_path: str = "data/models/yolo/yolov8n.onnx"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    input_size: int = 640


class YOLODetector:
    """Lightweight YOLO object detector supporting PyTorch/Ultralytics and ONNX Runtime backends."""

    def __init__(self, config: Optional[YOLOConfig] = None) -> None:
        self.config = config or YOLOConfig()
        self._logger = logging.getLogger(__name__)
        self._model: Any = None
        self._onnx_session: Any = None
        self._backend: str = "none"
        self._is_loaded: bool = False
        self._classes = COCO_CLASSES

        # Distinct color palette for bounding boxes (BGR)
        self._colors = [
            (56, 180, 255), (100, 220, 100), (255, 128, 0), (200, 50, 255),
            (255, 215, 0), (0, 255, 255), (255, 99, 71), (144, 238, 144),
            (255, 105, 180), (64, 224, 208), (220, 20, 60), (30, 144, 255),
        ]

        self.load()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def backend(self) -> str:
        return self._backend

    def load(self) -> bool:
        """Attempt to load YOLO model via Ultralytics or ONNX Runtime."""
        # 1. Try Ultralytics (.pt)
        pt_path = Path(self.config.model_path)
        if pt_path.exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(pt_path))
                self._backend = "ultralytics"
                self._is_loaded = True
                self._logger.info(f"Loaded YOLOv8n via Ultralytics PyTorch ({pt_path})")
                return True
            except Exception as e:
                self._logger.warning(f"Failed to load Ultralytics backend: {e}. Falling back to ONNX.")

        # 2. Try ONNX Runtime (.onnx)
        onnx_p = Path(self.config.onnx_path)
        if onnx_p.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self._onnx_session = ort.InferenceSession(
                    str(onnx_p),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._backend = "onnx"
                self._is_loaded = True
                self._logger.info(f"Loaded YOLOv8n via ONNX Runtime ({onnx_p})")
                return True
            except Exception as e:
                self._logger.warning(f"Failed to load ONNX backend: {e}")

        # 3. If model doesn't exist locally yet, try downloading with Ultralytics
        try:
            from ultralytics import YOLO
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            self._model = YOLO("yolov8n.pt")
            self._backend = "ultralytics"
            self._is_loaded = True
            self._logger.info("Downloaded and initialized YOLOv8n via Ultralytics.")
            return True
        except Exception as e:
            self._logger.error(f"Could not initialize YOLO detector: {e}")
            self._is_loaded = False
            return False

    def detect(
        self,
        frame_rgb: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> list[DetectedObject]:
        """Detect objects in an RGB image. Returns list of DetectedObject."""
        if not self._is_loaded:
            return []

        conf_thresh = confidence_threshold if confidence_threshold is not None else self.config.confidence_threshold

        if self._backend == "ultralytics" and self._model is not None:
            return self._detect_ultralytics(frame_rgb, conf_thresh)
        elif self._backend == "onnx" and self._onnx_session is not None:
            return self._detect_onnx(frame_rgb, conf_thresh)

        return []

    def _detect_ultralytics(self, frame_rgb: np.ndarray, conf_thresh: float) -> list[DetectedObject]:
        try:
            results = self._model(frame_rgb, conf=conf_thresh, iou=self.config.iou_threshold, verbose=False)
            if not results or len(results) == 0:
                return []

            res = results[0]
            detections: list[DetectedObject] = []
            if res.boxes is None:
                return []

            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                c_id = int(classes[i])
                label = self._classes[c_id] if c_id < len(self._classes) else f"object_{c_id}"
                c = float(confs[i])
                x1, y1, x2, y2 = map(int, boxes[i])
                detections.append(DetectedObject(
                    class_id=c_id,
                    label=label,
                    confidence=c,
                    box=(x1, y1, x2, y2),
                ))

            return detections
        except Exception as e:
            self._logger.error(f"Error during Ultralytics YOLO inference: {e}")
            return []

    def _detect_onnx(self, frame_rgb: np.ndarray, conf_thresh: float) -> list[DetectedObject]:
        try:
            h_orig, w_orig = frame_rgb.shape[:2]
            size = self.config.input_size

            # Letterbox resize
            r = min(size / h_orig, size / w_orig)
            new_w, new_h = int(w_orig * r), int(h_orig * r)
            resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            padded = np.full((size, size, 3), 114, dtype=np.uint8)
            pad_x = (size - new_w) // 2
            pad_y = (size - new_h) // 2
            padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

            input_tensor = padded.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, ...]

            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: input_tensor})

            # Shape is (1, 84, 8400) -> transpose to (8400, 84)
            preds = np.transpose(outputs[0][0], (1, 0))

            boxes_xyxy: list[list[int]] = []
            confidences: list[float] = []
            class_ids: list[int] = []

            for p in preds:
                scores = p[4:]
                best_class = int(np.argmax(scores))
                score = float(scores[best_class])
                if score >= conf_thresh:
                    cx, cy, w, h = p[:4]
                    # Undo padding and scale back to original image
                    x1 = int(((cx - w / 2) - pad_x) / r)
                    y1 = int(((cy - h / 2) - pad_y) / r)
                    w_box = int(w / r)
                    h_box = int(h / r)

                    x1 = max(0, min(x1, w_orig - 1))
                    y1 = max(0, min(y1, h_orig - 1))
                    w_box = max(1, min(w_box, w_orig - x1))
                    h_box = max(1, min(h_box, h_orig - y1))

                    boxes_xyxy.append([x1, y1, w_box, h_box])
                    confidences.append(score)
                    class_ids.append(best_class)

            if not boxes_xyxy:
                return []

            indices = cv2.dnn.NMSBoxes(boxes_xyxy, confidences, conf_thresh, self.config.iou_threshold)
            detections: list[DetectedObject] = []
            if len(indices) > 0:
                for idx in indices.flatten():
                    x1, y1, wb, hb = boxes_xyxy[idx]
                    cid = class_ids[idx]
                    lbl = self._classes[cid] if cid < len(self._classes) else f"object_{cid}"
                    detections.append(DetectedObject(
                        class_id=cid,
                        label=lbl,
                        confidence=confidences[idx],
                        box=(x1, y1, x1 + wb, y1 + hb),
                    ))

            return detections
        except Exception as e:
            self._logger.error(f"Error during ONNX YOLO inference: {e}")
            return []

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[DetectedObject],
        show_boxes: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Render distinct, high-contrast bounding boxes and labels onto the image."""
        if not detections:
            return frame

        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = self._colors[det.class_id % len(self._colors)]

            if show_boxes:
                # Outer border
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                # Corner accents for a modern HUD look
                length = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
                if length > 3:
                    cv2.line(out, (x1, y1), (x1 + length, y1), color, 3)
                    cv2.line(out, (x1, y1), (x1, y1 + length), color, 3)
                    cv2.line(out, (x2, y1), (x2 - length, y1), color, 3)
                    cv2.line(out, (x2, y1), (x2, y1 + length), color, 3)
                    cv2.line(out, (x1, y2), (x1 + length, y2), color, 3)
                    cv2.line(out, (x1, y2), (x1, y2 - length), color, 3)
                    cv2.line(out, (x2, y2), (x2 - length, y2), color, 3)
                    cv2.line(out, (x2, y2), (x2, y2 - length), color, 3)

            if show_labels:
                text = f"{det.label}: {det.confidence:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Background banner for text
                by1 = max(0, y1 - th - 6)
                by2 = y1
                bx2 = min(out.shape[1], x1 + tw + 8)

                cv2.rectangle(out, (x1, by1), (bx2, by2), color, -1)
                # Text in dark or white for optimal contrast
                brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                cv2.putText(out, text, (x1 + 4, y1 - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)

        return out
