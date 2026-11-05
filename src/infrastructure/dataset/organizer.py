"""DatasetOrganizer — automatic folder structure and dataset management."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional


class DatasetOrganizer:
    """Creates and maintains standardized dataset directory structure.

    Structure:
        data/datasets/
        ├── alphabet/
        │   ├── alif/
        │   │   ├── samples/
        │   │   │   ├── abc12345/
        │   │   │   │   ├── landmarks.npy
        │   │   │   │   ├── metadata.json
        │   │   │   │   └── (motion_data.npy)  (optional)
        │   │   │   └── def67890/
        │   │   └── class_info.json
        │   ├── baa/
        │   └── ...
        ├── words/
        │   ├── hello/
        │   └── ...
        └── dynamic/
            ├── how_are_you/
            └── ...
    """

    def __init__(self, base_dir: str | Path = "data/datasets") -> None:
        self._base_dir = Path(base_dir)

    # ── Directory operations ────────────────────────────────────────

    def ensure_structure(self, dataset_types: Optional[list[str]] = None) -> None:
        """Create the full dataset directory skeleton if it doesn't exist."""
        types = dataset_types or ["alphabet", "words", "dynamic"]
        for t in types:
            (self._base_dir / t).mkdir(parents=True, exist_ok=True)

    def create_class(self, dataset_type: str, class_name: str) -> Path:
        """Create a new class directory within a dataset type."""
        class_dir = self._base_dir / dataset_type / class_name
        (class_dir / "samples").mkdir(parents=True, exist_ok=True)
        return class_dir

    def get_class_dir(self, dataset_type: str, class_name: str) -> Path:
        return self._base_dir / dataset_type / class_name

    def get_samples_dir(self, dataset_type: str, class_name: str) -> Path:
        return self.get_class_dir(dataset_type, class_name) / "samples"

    # ── Class info management ───────────────────────────────────────

    def write_class_info(self, dataset_type: str, class_name: str, info: dict[str, Any]) -> None:
        """Write metadata JSON about a class (e.g., Arabic label, description)."""
        class_dir = self.get_class_dir(dataset_type, class_name)
        with open(class_dir / "class_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def read_class_info(self, dataset_type: str, class_name: str) -> dict[str, Any]:
        class_dir = self.get_class_dir(dataset_type, class_name)
        info_file = class_dir / "class_info.json"
        if info_file.exists():
            with open(info_file, encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ── Listing ─────────────────────────────────────────────────────

    def list_classes(self, dataset_type: str) -> list[str]:
        """List all class directories for a given dataset type."""
        type_dir = self._base_dir / dataset_type
        if not type_dir.exists():
            return []
        return sorted(
            d.name for d in type_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def list_sample_dirs(self, dataset_type: str, class_name: str) -> list[Path]:
        """List all sample directories for a class."""
        samples_dir = self.get_samples_dir(dataset_type, class_name)
        if not samples_dir.exists():
            return []
        return sorted(
            d for d in samples_dir.iterdir()
            if d.is_dir()
        )

    def count_samples(self, dataset_type: str, class_name: str) -> int:
        return len(self.list_sample_dirs(dataset_type, class_name))

    def class_statistics(self, dataset_type: str) -> dict[str, int]:
        """Return dict of class_name → sample count for a dataset type."""
        stats: dict[str, int] = {}
        for class_name in self.list_classes(dataset_type):
            stats[class_name] = self.count_samples(dataset_type, class_name)
        return stats

    # ── Maintenance ─────────────────────────────────────────────────

    def rename_class(self, dataset_type: str, old_name: str, new_name: str) -> bool:
        """Rename a class directory."""
        old_path = self.get_class_dir(dataset_type, old_name)
        new_path = self.get_class_dir(dataset_type, new_name)
        if not old_path.exists() or new_path.exists():
            return False
        old_path.rename(new_path)
        return True

    def delete_class(self, dataset_type: str, class_name: str) -> bool:
        """Delete a class and all its samples."""
        class_dir = self.get_class_dir(dataset_type, class_name)
        if not class_dir.exists():
            return False
        shutil.rmtree(class_dir)
        return True

    def move_sample(
        self,
        sample_dir: Path,
        target_type: str,
        target_class: str,
    ) -> Optional[Path]:
        """Move a sample to a different class."""
        if not sample_dir.exists():
            return None
        target_dir = self.get_samples_dir(target_type, target_class)
        target_dir.mkdir(parents=True, exist_ok=True)
        new_path = target_dir / sample_dir.name
        sample_dir.rename(new_path)
        return new_path
