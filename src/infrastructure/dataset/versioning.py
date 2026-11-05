"""DatasetVersionManager — dataset versioning with manifests and checksums.

Supports:
    - Creating version manifests (class counts, checksums, timestamps)
    - Verifying integrity against manifests
    - Diffing two manifests to detect changes
    - Locking dataset versions for reproducibility
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path


class DatasetVersionManager:
    """Manages dataset version manifests with integrity verification."""

    SCHEMA_VERSION = "1.0"

    def __init__(self, manifests_dir: str | Path = "data/datasets/manifests") -> None:
        self._manifests_dir = Path(manifests_dir)
        self._manifests_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

    def create_manifest(self, dataset_path: str | Path) -> dict:
        """Walk dataset directory and build a version manifest."""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        classes: dict[str, dict] = {}
        total_samples = 0
        checksums: list[str] = []

        for class_dir in sorted(dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            samples_dir = class_dir / "samples"
            if not samples_dir.exists():
                classes[class_name] = {"count": 0, "checksum": ""}
                continue

            sample_ids: list[str] = []
            for sample_dir in sorted(samples_dir.iterdir()):
                if sample_dir.is_dir():
                    lm_file = sample_dir / "landmarks.npy"
                    if lm_file.exists():
                        sample_ids.append(str(sample_dir.name))
                        checksums.append(self._file_checksum(lm_file))

            classes[class_name] = {
                "count": len(sample_ids),
                "samples": sample_ids,
                "checksum": hashlib.md5("".join(sample_ids).encode()).hexdigest(),
            }
            total_samples += len(sample_ids)

        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "dataset_path": str(dataset_path.resolve()),
            "total_classes": len(classes),
            "total_samples": total_samples,
            "classes": classes,
            "root_checksum": hashlib.md5("".join(sorted(checksums)).encode()).hexdigest(),
            "created_at": time.time(),
        }
        return manifest

    def save_manifest(self, manifest: dict, name: str) -> Path:
        path = self._manifests_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        self._logger.info(f"Manifest saved: {path} ({manifest['total_samples']} samples)")
        return path

    def load_manifest(self, name: str) -> dict:
        path = self._manifests_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        with open(path) as f:
            return json.load(f)

    def verify_manifest(self, dataset_path: str | Path, manifest: dict) -> bool:
        """Verify dataset integrity against a manifest."""
        current = self.create_manifest(dataset_path)
        expected = manifest.get("root_checksum", "")
        actual = current.get("root_checksum", "")
        return expected == actual

    def diff_manifests(self, old_name: str, new_name: str) -> dict:
        """Compare two manifests and report differences."""
        old = self.load_manifest(old_name)
        new = self.load_manifest(new_name)

        old_classes = set(old.get("classes", {}).keys())
        new_classes = set(new.get("classes", {}).keys())

        added = new_classes - old_classes
        removed = old_classes - new_classes
        common = old_classes & new_classes

        changed = []
        for cls in sorted(common):
            old_count = old["classes"][cls]["count"]
            new_count = new["classes"][cls]["count"]
            if old_count != new_count:
                changed.append({"class": cls, "old_count": old_count, "new_count": new_count})

        return {
            "old_samples": old.get("total_samples", 0),
            "new_samples": new.get("total_samples", 0),
            "added_classes": sorted(added),
            "removed_classes": sorted(removed),
            "changed_classes": changed,
            "schema_match": old.get("schema_version") == new.get("schema_version"),
        }

    def list_manifests(self) -> list[str]:
        return sorted([p.stem for p in self._manifests_dir.glob("*.json")])

    @staticmethod
    def _file_checksum(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
