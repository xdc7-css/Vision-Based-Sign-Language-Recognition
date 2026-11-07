"""Dataset management module — collection, organization, registry, and validation.

Services:
    DatasetCollector  — Manages capture sessions for alphabet, dynamic, word gestures.
    DatasetOrganizer  — Creates and maintains standardized dataset directory structure.
    DatasetValidator  — Validates dataset integrity, detects outliers, reports quality.
    GestureRegistry   — Manages gesture class lifecycle (CRUD, persistence, archiving).
"""

from src.infrastructure.dataset.collector import DatasetCollector
from src.infrastructure.dataset.gesture_registry import (
    GestureClass,
    GestureRegistry,
    normalize_gesture_name,
    sanitize_folder_name,
)
from src.infrastructure.dataset.organizer import DatasetOrganizer
from src.infrastructure.dataset.validator import DatasetValidator

__all__ = [
    "DatasetCollector",
    "DatasetOrganizer",
    "DatasetValidator",
    "GestureClass",
    "GestureRegistry",
    "normalize_gesture_name",
    "sanitize_folder_name",
]

