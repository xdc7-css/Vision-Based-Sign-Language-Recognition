"""GestureRegistry — Persistent, editable registry for Arabic sign language classes.

Provides complete CRUD lifecycle management:
- Create: Add custom Arabic words/letters with Static, Dynamic, or Both categories.
- Read: List active classes, search, filter (samples, category), get sample counts.
- Update: Transactional rename (updating folder names and sample metadata.json), category change.
- Delete: Safe archive vs. permanent sample destruction.
- Persistence: JSON registry with atomic file writes and first-time migration from DEFAULT_VOCABULARY and disk.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.domain.recognition import DEFAULT_VOCABULARY, GestureCategory

logger = logging.getLogger(__name__)


def normalize_gesture_name(name: str) -> str:
    """Normalize Arabic gesture text: collapse multiple spaces/underscores into single space."""
    return re.sub(r"[\s_]+", " ", name.strip())


def sanitize_folder_name(name: str) -> str:
    """Convert display name to filesystem-safe folder name (spaces replaced with underscore)."""
    clean = normalize_gesture_name(name)
    return clean.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _canonical_key(name: str) -> str:
    """Internal key for duplicate comparison: lowercase, spaces/underscores unified."""
    return sanitize_folder_name(name).lower()


@dataclass
class GestureClass:
    """Domain model representing a single gesture class in the active/archived vocabulary."""

    id: str
    name: str  # Display name in Arabic (e.g. "مرحبا", "السلام عليكم")
    category: str  # "static", "dynamic", "both", "alphabet"
    description: str = ""
    is_active: bool = True
    is_default: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "is_active": self.is_active,
            "is_default": self.is_default,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GestureClass:
        return cls(
            id=data["id"],
            name=data["name"],
            category=data.get("category", "static"),
            description=data.get("description", ""),
            is_active=data.get("is_active", True),
            is_default=data.get("is_default", False),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


class GestureRegistry:
    """Persistent registry managing Arabic gesture classes and vocabulary state."""

    DEFAULT_REGISTRY_PATH = Path("data/vocabulary/gesture_classes.json")
    DEFAULT_DATASET_DIR = Path("data/datasets")

    def __init__(
        self,
        registry_path: Optional[Path | str] = None,
        dataset_dir: Optional[Path | str] = None,
        auto_init: bool = True,
    ) -> None:
        self.registry_path = Path(registry_path or self.DEFAULT_REGISTRY_PATH)
        self.dataset_dir = Path(dataset_dir or self.DEFAULT_DATASET_DIR)

        self._classes: dict[str, GestureClass] = {}  # key: class.id -> GestureClass
        self._archived_classes: dict[str, GestureClass] = {}
        self._deleted_ids: set[str] = set()

        if auto_init:
            self.load_or_initialize()

    # ── Persistence & Initialization ─────────────────────────────────────

    def load_or_initialize(self) -> None:
        """Load existing registry from disk, or perform first-time template migration."""
        if self.registry_path.exists():
            self._load_from_disk()
            self._sync_with_disk_directories()
        else:
            self._initialize_from_defaults_and_disk()
            self.save()

    def _load_from_disk(self) -> None:
        """Parse validated JSON registry."""
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._classes.clear()
            self._archived_classes.clear()
            self._deleted_ids = set(data.get("deleted_ids", []))

            for item in data.get("classes", []):
                cls_obj = GestureClass.from_dict(item)
                self._classes[cls_obj.id] = cls_obj

            for item in data.get("archived_classes", []):
                cls_obj = GestureClass.from_dict(item)
                cls_obj.is_active = False
                self._archived_classes[cls_obj.id] = cls_obj

            logger.info(
                f"Loaded GestureRegistry: {len(self._classes)} active, "
                f"{len(self._archived_classes)} archived, {len(self._deleted_ids)} deleted."
            )
        except Exception as e:
            logger.error(f"Failed to load registry from {self.registry_path}: {e}")
            self._initialize_from_defaults_and_disk()
            self.save()

    def _initialize_from_defaults_and_disk(self) -> None:
        """First-time initialization: migrate DEFAULT_VOCABULARY template + scan disk datasets."""
        self._classes.clear()
        self._archived_classes.clear()
        self._deleted_ids.clear()

        # 1. Template from DEFAULT_VOCABULARY
        for entry in DEFAULT_VOCABULARY:
            cat = "alphabet"
            if entry.category == GestureCategory.STATIC_WORD:
                cat = "static"
            elif entry.category == GestureCategory.DYNAMIC_GESTURE:
                cat = "dynamic"

            cls_obj = GestureClass(
                id=entry.gesture_id,
                name=normalize_gesture_name(entry.arabic_text),
                category=cat,
                description=entry.description,
                is_active=True,
                is_default=True,
            )
            self._classes[cls_obj.id] = cls_obj

        # 2. Merge actual folders on disk
        self._sync_with_disk_directories(is_first_init=True)

    def _sync_with_disk_directories(self, is_first_init: bool = False) -> None:
        """Inspect dataset disk directories; discover and safely register any untracked folders."""
        if not self.dataset_dir.exists():
            return

        active_keys = {_canonical_key(c.name): c for c in self._classes.values()}
        archived_keys = {_canonical_key(c.name): c for c in self._archived_classes.values()}

        # Scan words, dynamic, and alphabet
        found_on_disk: dict[str, set[str]] = {}  # canonical_key -> set of subdirs ('words', 'dynamic', 'alphabet')

        for cat_dir_name in ["words", "dynamic", "alphabet"]:
            sub_path = self.dataset_dir / cat_dir_name
            if not sub_path.exists():
                continue
            for d in sub_path.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    key = _canonical_key(d.name)
                    found_on_disk.setdefault(key, set()).add(cat_dir_name)

        changed = False

        for key, cats in found_on_disk.items():
            # Check if already tracked in active classes
            if key in active_keys:
                cls_obj = active_keys[key]
                # Update category to 'both' if on disk in both words and dynamic
                if "words" in cats and "dynamic" in cats and cls_obj.category != "both":
                    cls_obj.category = "both"
                    changed = True
                continue

            # If in archived classes, don't resurrect unless first init
            if key in archived_keys:
                continue

            # If permanently deleted, do not resurrect
            # (unless it actually has samples and wasn't deleted in this session)
            sample_count = sum(self._count_samples_in_disk_dir(cat, key) for cat in cats)
            if not is_first_init and sample_count == 0:
                continue

            # New untracked folder found on disk with data or during first init
            display_name = normalize_gesture_name(next(iter(cats))) if not cats else key.replace("_", " ")
            # Try to grab the actual directory name from disk
            for cat in cats:
                candidate = self.dataset_dir / cat / key
                if candidate.exists():
                    display_name = candidate.name.replace("_", " ")
                    break

            cat_type = "static"
            if "words" in cats and "dynamic" in cats:
                cat_type = "both"
            elif "dynamic" in cats:
                cat_type = "dynamic"
            elif "alphabet" in cats:
                cat_type = "alphabet"

            gen_id = f"custom_{key}"
            if gen_id in self._classes:
                gen_id = f"custom_{key}_{uuid.uuid4().hex[:6]}"

            new_class = GestureClass(
                id=gen_id,
                name=display_name,
                category=cat_type,
                description="Custom Arabic gesture class",
                is_active=True,
                is_default=False,
            )
            self._classes[new_class.id] = new_class
            active_keys[key] = new_class
            changed = True

        if changed and not is_first_init:
            self.save()

    def save(self) -> None:
        """Save registry atomically to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0.0",
            "updated_at": time.time(),
            "classes": [c.to_dict() for c in sorted(self._classes.values(), key=lambda x: x.name)],
            "archived_classes": [c.to_dict() for c in sorted(self._archived_classes.values(), key=lambda x: x.name)],
            "deleted_ids": sorted(self._deleted_ids),
        }

        temp_path = self.registry_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            temp_path.replace(self.registry_path)
            logger.debug(f"Saved gesture registry to {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to write gesture registry to {self.registry_path}: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    # ── CRUD Operations ──────────────────────────────────────────────────

    def add_class(
        self,
        name: str,
        category: str = "static",
        description: str = "",
        class_id: Optional[str] = None,
    ) -> GestureClass:
        """Create and register a new gesture class.

        Args:
            name: Arabic display name (e.g. 'شكرا', 'سيارة').
            category: 'static', 'dynamic', 'both', or 'alphabet'.
            description: Optional note or description.
            class_id: Optional stable identifier.

        Returns:
            The created GestureClass instance.

        Raises:
            ValueError: If name is empty or already exists in active vocabulary.
        """
        clean_name = normalize_gesture_name(name)
        if not clean_name:
            raise ValueError("Class name cannot be empty.")

        category = category.lower()
        if category not in ("static", "dynamic", "both", "alphabet"):
            category = "static"

        key = _canonical_key(clean_name)

        # Check duplicate in active classes
        for existing in self._classes.values():
            if _canonical_key(existing.name) == key:
                raise ValueError(f"Class «{clean_name}» already exists in the active vocabulary.")

        # Check if was previously archived
        for arch_id, arch_cls in list(self._archived_classes.items()):
            if _canonical_key(arch_cls.name) == key:
                # Reactivate archived class
                del self._archived_classes[arch_id]
                arch_cls.is_active = True
                arch_cls.category = category
                arch_cls.description = description or arch_cls.description
                arch_cls.updated_at = time.time()
                self._classes[arch_cls.id] = arch_cls
                self._ensure_dataset_directories(clean_name, category)
                self.save()
                return arch_cls

        # Generate stable ID
        if not class_id:
            class_id = f"custom_{key}_{uuid.uuid4().hex[:6]}"

        new_class = GestureClass(
            id=class_id,
            name=clean_name,
            category=category,
            description=description,
            is_active=True,
            is_default=False,
            created_at=time.time(),
            updated_at=time.time(),
        )

        self._classes[new_class.id] = new_class
        if class_id in self._deleted_ids:
            self._deleted_ids.remove(class_id)

        self._ensure_dataset_directories(clean_name, category)
        self.save()
        return new_class

    def rename_class(self, old_name: str, new_name: str) -> GestureClass:
        """Rename an existing class and update its sample files and metadata safely.

        Args:
            old_name: Current class name.
            new_name: New Arabic class name.

        Returns:
            The updated GestureClass.

        Raises:
            ValueError: If class not found, new name is invalid or collides.
            FileExistsError: If target directory already exists with independent data.
        """
        old_clean = normalize_gesture_name(old_name)
        new_clean = normalize_gesture_name(new_name)

        if not new_clean:
            raise ValueError("New class name cannot be empty.")

        target_cls = self.get_class(old_clean)
        if not target_cls:
            raise ValueError(f"Class «{old_name}» not found in active vocabulary.")

        if _canonical_key(old_clean) == _canonical_key(new_clean):
            # Only casing/spacing difference
            target_cls.name = new_clean
            target_cls.updated_at = time.time()
            self.save()
            return target_cls

        # Check collision with other active classes
        new_key = _canonical_key(new_clean)
        for other in self._classes.values():
            if other.id != target_cls.id and _canonical_key(other.name) == new_key:
                raise ValueError(f"Class «{new_clean}» already exists in the active vocabulary.")

        old_sanitized = sanitize_folder_name(old_clean)
        new_sanitized = sanitize_folder_name(new_clean)

        # Check filesystem destination conflicts
        for folder_type in ["words", "dynamic", "alphabet"]:
            src = self.dataset_dir / folder_type / old_sanitized
            dst = self.dataset_dir / folder_type / new_sanitized
            if src.exists() and dst.exists() and src != dst:
                raise FileExistsError(f"Target folder already exists on disk: {dst}")

        # Transactional rename of directories
        renamed_dirs: list[tuple[Path, Path]] = []
        try:
            for folder_type in ["words", "dynamic", "alphabet"]:
                src = self.dataset_dir / folder_type / old_sanitized
                dst = self.dataset_dir / folder_type / new_sanitized
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    src.rename(dst)
                    renamed_dirs.append((src, dst))

                    # Update gesture_label in all sample metadata.json files
                    samples_dir = dst / "samples"
                    if samples_dir.exists():
                        for s_dir in samples_dir.iterdir():
                            if not s_dir.is_dir():
                                continue
                            meta_file = s_dir / "metadata.json"
                            if meta_file.exists():
                                try:
                                    with open(meta_file, "r", encoding="utf-8") as f:
                                        meta = json.load(f)
                                    meta["gesture_label"] = new_clean
                                    with open(meta_file, "w", encoding="utf-8") as f:
                                        json.dump(meta, f, ensure_ascii=False, indent=2)
                                except Exception as err:
                                    logger.warning(f"Could not update sample metadata in {meta_file}: {err}")
        except Exception as e:
            # Rollback any moved directories
            logger.error(f"Error during folder rename, rolling back: {e}")
            for src, dst in reversed(renamed_dirs):
                if dst.exists():
                    dst.rename(src)
            raise

        # Update registry entry
        target_cls.name = new_clean
        target_cls.updated_at = time.time()
        self.save()
        return target_cls

    def update_category(self, name: str, new_category: str) -> GestureClass:
        """Update category of an active class ('static', 'dynamic', 'both', 'alphabet')."""
        cls_obj = self.get_class(name)
        if not cls_obj:
            raise ValueError(f"Class «{name}» not found.")

        new_cat = new_category.lower()
        if new_cat not in ("static", "dynamic", "both", "alphabet"):
            raise ValueError(f"Invalid category: {new_category}")

        cls_obj.category = new_cat
        cls_obj.updated_at = time.time()
        self._ensure_dataset_directories(cls_obj.name, new_cat)
        self.save()
        return cls_obj

    def delete_class(
        self,
        name: str,
        permanent_delete_samples: bool = False,
    ) -> tuple[str, int]:
        """Delete or archive a gesture class.

        Args:
            name: Class name to remove.
            permanent_delete_samples: If True, permanently erase saved sample files from disk.
                                      If False, safely archive the class and keep all samples.

        Returns:
            tuple[action_taken, affected_samples_count]:
            - ('deleted', 0): empty class cleanly removed.
            - ('archived', N): class archived, N samples preserved.
            - ('permanently_deleted', N): class and N samples erased.
        """
        cls_obj = self.get_class(name)
        if not cls_obj:
            raise ValueError(f"Class «{name}» not found in active vocabulary.")

        counts = self.get_sample_counts(name)
        total_samples = counts["total"]

        # If empty (0 samples), straightforward clean removal
        if total_samples == 0:
            del self._classes[cls_obj.id]
            self._deleted_ids.add(cls_obj.id)
            self._cleanup_empty_directories(name)
            self.save()
            return ("deleted", 0)

        # Has samples: Check permanent vs safe archive
        if not permanent_delete_samples:
            # Safe Archive: remove from active list, preserve files 100%
            del self._classes[cls_obj.id]
            cls_obj.is_active = False
            cls_obj.updated_at = time.time()
            self._archived_classes[cls_obj.id] = cls_obj
            self.save()
            return ("archived", total_samples)

        # Explicit permanent deletion of samples
        del self._classes[cls_obj.id]
        if cls_obj.id in self._archived_classes:
            del self._archived_classes[cls_obj.id]
        self._deleted_ids.add(cls_obj.id)

        sanitized = sanitize_folder_name(name)
        for cat in ["words", "dynamic", "alphabet"]:
            p = self.dataset_dir / cat / sanitized
            if p.exists():
                try:
                    shutil.rmtree(p)
                except Exception as e:
                    logger.error(f"Failed to delete {p}: {e}")

        self.save()
        return ("permanently_deleted", total_samples)

    def unarchive_class(self, name: str) -> Optional[GestureClass]:
        """Restore an archived class back to active vocabulary."""
        key = _canonical_key(name)
        for arch_id, arch_cls in list(self._archived_classes.items()):
            if _canonical_key(arch_cls.name) == key:
                del self._archived_classes[arch_id]
                arch_cls.is_active = True
                arch_cls.updated_at = time.time()
                self._classes[arch_cls.id] = arch_cls
                self.save()
                return arch_cls
        return None

    # ── Queries & Inspection ─────────────────────────────────────────────

    def get_class(self, name_or_id: str) -> Optional[GestureClass]:
        """Find active gesture class by ID, display name, or folder name."""
        if name_or_id in self._classes:
            return self._classes[name_or_id]

        key = _canonical_key(name_or_id)
        for c in self._classes.values():
            if _canonical_key(c.name) == key or c.id == name_or_id:
                return c
        return None

    def list_active_classes(self) -> list[GestureClass]:
        """Return list of all active GestureClass objects sorted alphabetically."""
        return sorted(self._classes.values(), key=lambda c: c.name)

    def list_archived_classes(self) -> list[GestureClass]:
        """Return list of all archived GestureClass objects."""
        return sorted(self._archived_classes.values(), key=lambda c: c.name)

    def get_sample_counts(self, name: str) -> dict[str, int]:
        """Count actual valid recorded sample directories on disk for a given class."""
        sanitized = sanitize_folder_name(name)
        static_cnt = self._count_samples_in_disk_dir("words", sanitized)
        dynamic_cnt = self._count_samples_in_disk_dir("dynamic", sanitized)
        alphabet_cnt = self._count_samples_in_disk_dir("alphabet", sanitized)

        return {
            "static": static_cnt,
            "dynamic": dynamic_cnt,
            "alphabet": alphabet_cnt,
            "total": static_cnt + dynamic_cnt + alphabet_cnt,
        }

    def _count_samples_in_disk_dir(self, cat: str, folder_name: str) -> int:
        samples_dir = self.dataset_dir / cat / folder_name / "samples"
        if not samples_dir.exists():
            return 0
        count = 0
        for d in samples_dir.iterdir():
            if d.is_dir() and (d / "landmarks.npy").exists():
                count += 1
        return count

    def filter_classes(
        self,
        query: str = "",
        status_filter: str = "all",  # "all", "with_samples", "empty"
        category_filter: str = "all",  # "all", "static", "dynamic", "both", "alphabet"
        sort_alphabetical: bool = True,
    ) -> list[tuple[GestureClass, dict[str, int]]]:
        """Filter and sort active gesture classes with real-time sample counts."""
        query_clean = normalize_gesture_name(query).lower()
        results: list[tuple[GestureClass, dict[str, int]]] = []

        for cls_obj in self._classes.values():
            # Query match
            if query_clean:
                clean_cls_name = normalize_gesture_name(cls_obj.name).lower()
                if query_clean not in clean_cls_name and query_clean not in cls_obj.description.lower():
                    continue

            # Category filter
            if category_filter != "all":
                if category_filter == "both" and cls_obj.category != "both":
                    continue
                elif category_filter == "static" and cls_obj.category not in ("static", "both"):
                    continue
                elif category_filter == "dynamic" and cls_obj.category not in ("dynamic", "both"):
                    continue
                elif category_filter == "alphabet" and cls_obj.category != "alphabet":
                    continue

            counts = self.get_sample_counts(cls_obj.name)

            # Status filter (with samples vs empty)
            if status_filter == "with_samples" and counts["total"] == 0:
                continue
            elif status_filter == "empty" and counts["total"] > 0:
                continue

            results.append((cls_obj, counts))

        if sort_alphabetical:
            results.sort(key=lambda item: item[0].name)

        return results

    # ── Helpers ──────────────────────────────────────────────────────────

    def _ensure_dataset_directories(self, name: str, category: str) -> None:
        """Ensure appropriate dataset samples directories exist for the class."""
        sanitized = sanitize_folder_name(name)
        if category in ("static", "both"):
            (self.dataset_dir / "words" / sanitized / "samples").mkdir(parents=True, exist_ok=True)
        if category in ("dynamic", "both"):
            (self.dataset_dir / "dynamic" / sanitized / "samples").mkdir(parents=True, exist_ok=True)
        if category == "alphabet":
            (self.dataset_dir / "alphabet" / sanitized / "samples").mkdir(parents=True, exist_ok=True)

    def _cleanup_empty_directories(self, name: str) -> None:
        """Remove empty class directories on disk."""
        sanitized = sanitize_folder_name(name)
        for cat in ["words", "dynamic", "alphabet"]:
            p = self.dataset_dir / cat / sanitized
            if p.exists():
                samples_dir = p / "samples"
                has_samples = samples_dir.exists() and any(samples_dir.iterdir())
                if not has_samples:
                    shutil.rmtree(p, ignore_errors=True)
