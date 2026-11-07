"""Dataset and Vocabulary service for ArSL Studio."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from src.infrastructure.dataset.gesture_registry import (
    GestureRegistry,
    sanitize_folder_name,
)
from src.domain.multimodal import MULTIMODAL_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class DatasetService:
    """Service wrapping GestureRegistry and dataset filesystem operations."""

    def __init__(self, registry_path: str = "data/vocabulary/gesture_classes.json", dataset_dir: str = "data/datasets") -> None:
        self.registry = GestureRegistry(registry_path=registry_path, dataset_dir=dataset_dir)
        self.dataset_dir = Path(dataset_dir)

    # ── Gestures CRUD ────────────────────────────────────────────────────

    def list_gestures(self, include_archived: bool = False) -> list[dict[str, Any]]:
        """Return all gesture classes with their current sample counts."""
        self.registry.load_or_initialize()
        classes = self.registry.list_active_classes()
        if include_archived:
            classes = classes + self.registry.list_archived_classes()

        results = []
        for c in classes:
            cnt_dict = self.registry.get_sample_counts(c.name)
            cnt = cnt_dict.get("total", 0)
            c_dict = c.to_dict()
            c_dict["sample_count"] = cnt
            c_dict["is_ready"] = cnt >= 15
            results.append(c_dict)
        return results

    def get_gesture(self, gesture_id: str) -> Optional[dict[str, Any]]:
        self.registry.load_or_initialize()
        c = self.registry.get_class(gesture_id)
        if not c:
            return None
        c_dict = c.to_dict()
        cnt_dict = self.registry.get_sample_counts(c.name)
        cnt = cnt_dict.get("total", 0)
        c_dict["sample_count"] = cnt
        c_dict["is_ready"] = cnt >= 15
        return c_dict

    def add_gesture(self, name: str, category: str = "static", description: str = "") -> dict[str, Any]:
        c = self.registry.add_class(name=name, category=category, description=description)
        c_dict = c.to_dict()
        c_dict["sample_count"] = 0
        c_dict["is_ready"] = False
        return c_dict

    def rename_gesture(self, gesture_id: str, new_name: str) -> dict[str, Any]:
        c = self.registry.get_class(gesture_id)
        if not c:
            raise ValueError(f"Class with ID {gesture_id} not found.")
        updated_c = self.registry.rename_class(old_name=c.name, new_name=new_name)
        c_dict = updated_c.to_dict()
        cnt_dict = self.registry.get_sample_counts(updated_c.name)
        cnt = cnt_dict.get("total", 0)
        c_dict["sample_count"] = cnt
        c_dict["is_ready"] = cnt >= 15
        return c_dict

    def archive_gesture(self, gesture_id: str) -> bool:
        c = self.registry.get_class(gesture_id)
        if not c:
            return False
        if c.id in self.registry._classes:
            del self.registry._classes[c.id]
        c.is_active = False
        c.updated_at = time.time()
        self.registry._archived_classes[c.id] = c
        self.registry.save()
        return True

    def restore_gesture(self, gesture_id: str) -> bool:
        c = self.registry._archived_classes.get(gesture_id)
        name_to_restore = c.name if c else gesture_id
        res = self.registry.unarchive_class(name_to_restore)
        return res is not None

    def delete_gesture(self, gesture_id: str, purge_samples: bool = False) -> bool:
        c = self.registry.get_class(gesture_id)
        if not c:
            return False
        self.registry.delete_class(name=c.name, permanent_delete_samples=purge_samples)
        return True

    # ── Dataset Summary ──────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Inspect datasets on disk and return comprehensive inventory."""
        self.registry.load_or_initialize()
        words_dir = self.dataset_dir / "words"
        
        total_samples = 0
        mm_samples = 0
        legacy_samples = 0
        trainable_classes = 0
        class_details = []

        all_classes = self.registry.list_active_classes()
        for c in all_classes:
            folder_name = sanitize_folder_name(c.name)
            samples_dir = words_dir / folder_name / "samples"
            
            c_mm = 0
            c_leg = 0
            if samples_dir.exists():
                for s_dir in samples_dir.iterdir():
                    if not s_dir.is_dir():
                        continue
                    meta_path = s_dir / "metadata.json"
                    is_mm = False
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r", encoding="utf-8") as mf:
                                m = json.load(mf)
                            if m.get("schema_version") == MULTIMODAL_SCHEMA_VERSION:
                                is_mm = True
                        except Exception:
                            pass
                    if is_mm:
                        c_mm += 1
                    else:
                        c_leg += 1

            c_total = c_mm + c_leg
            total_samples += c_total
            mm_samples += c_mm
            legacy_samples += c_leg
            if c_mm >= 2:  # Minimum for multimodal training
                trainable_classes += 1

            class_details.append({
                "id": c.id,
                "name": c.name,
                "category": c.category,
                "multimodal_samples": c_mm,
                "legacy_samples": c_leg,
                "total_samples": c_total,
                "is_trainable": c_mm >= 2,
            })

        return {
            "total_classes": len(all_classes),
            "total_samples": total_samples,
            "trainable_classes": trainable_classes,
            "multimodal_samples": mm_samples,
            "legacy_samples": legacy_samples,
            "classes": class_details,
        }

    # ── Sample Inspection & Deletion ──────────────────────────────────────

    def list_samples_for_gesture(self, gesture_id: str) -> list[dict[str, Any]]:
        c = self.registry.get_class(gesture_id)
        if not c:
            return []
        
        folder_name = sanitize_folder_name(c.name)
        samples_dir = self.dataset_dir / "words" / folder_name / "samples"
        if not samples_dir.exists():
            return []

        samples = []
        for s_dir in sorted(samples_dir.iterdir(), key=os.path.getmtime, reverse=True):
            if not s_dir.is_dir():
                continue
            meta_path = s_dir / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        m = json.load(mf)
                    samples.append({
                        "sample_id": s_dir.name,
                        "timestamp": m.get("timestamp", 0.0),
                        "schema_version": m.get("schema_version", "v1_single_hand"),
                        "num_frames": m.get("num_frames", 1),
                        "has_face": m.get("has_face", False),
                        "has_left": m.get("has_left_hand", False),
                        "has_right": m.get("has_right_hand", False),
                        "handedness": m.get("handedness", ""),
                    })
                except Exception:
                    pass
        return samples

    def delete_sample(self, gesture_id: str, sample_id: str) -> bool:
        c = self.registry.get_class(gesture_id)
        if not c:
            return False
        folder_name = sanitize_folder_name(c.name)
        sample_dir = self.dataset_dir / "words" / folder_name / "samples" / sample_id
        if sample_dir.exists():
            try:
                shutil.rmtree(sample_dir)
                return True
            except Exception as e:
                logger.error(f"Failed to delete sample {sample_dir}: {e}")
                return False
        return False
