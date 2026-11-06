"""Object Detection module using lightweight YOLO architecture."""

from src.infrastructure.object_detection.yolo_detector import DetectedObject, YOLOConfig, YOLODetector

__all__ = ["DetectedObject", "YOLOConfig", "YOLODetector"]
