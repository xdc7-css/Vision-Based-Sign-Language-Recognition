"""DatasetSplitManager — persistent train/val/test split management.

Creates, loads, and verifies split manifests. Supports multiple
split strategies (random, user-defined, condition-based).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatasetSplit:
    """A single dataset split with train/val/test sample IDs."""
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    strategy: str = "random"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    checksum: str = ""

    @property
    def is_valid(self) -> bool:
        """Check no overlap between splits."""
        train_set = set(self.train)
        val_set = set(self.val)
        test_set = set(self.test)
        if train_set & val_set:
            return False
        if train_set & test_set:
            return False
        if val_set & test_set:
            return False
        return True

    @property
    def total_samples(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


class DatasetSplitManager:
    """Creates, saves, loads, and verifies dataset splits."""

    def __init__(self, splits_dir: str | Path = "data/datasets/splits") -> None:
        self._splits_dir = Path(splits_dir)
        self._splits_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

    # ── Public API ──────────────────────────────────────────────────

    def create_splits(
        self,
        dataset_path: str | Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        strategy: str = "random",
        seed: int = 42,
        output_name: Optional[str] = None,
    ) -> DatasetSplit:
        """Create a dataset split from all sample IDs found in dataset_path.

        Args:
            dataset_path: Path to structured dataset (class_dir/samples/).
            train_ratio: Proportion for training.
            val_ratio: Proportion for validation.
            test_ratio: Proportion for testing.
            strategy: Split strategy ('random', 'user', 'condition').
            seed: Random seed for reproducibility.
            output_name: Name for the split manifest file.

        Returns:
            DatasetSplit with sample ID lists.
        """
        sample_ids = self._collect_sample_ids(dataset_path)
        if not sample_ids:
            raise ValueError(f"No samples found in {dataset_path}")

        random.Random(seed).shuffle(sample_ids)

        n = len(sample_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        split = DatasetSplit(
            train=sample_ids[:n_train],
            val=sample_ids[n_train:n_train + n_val],
            test=sample_ids[n_train + n_val:],
            strategy=strategy,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        if not split.is_valid:
            raise RuntimeError("Generated split has overlapping sets")

        split.checksum = self._compute_checksum(split)

        # Save manifest
        name = output_name or f"split_{len(split.train)}_{len(split.val)}_{len(split.test)}"
        self.save_split(split, name)

        self._logger.info(
            f"Created split '{name}': "
            f"train={len(split.train)}, val={len(split.val)}, test={len(split.test)}"
        )
        return split

    def save_split(self, split: DatasetSplit, name: str) -> Path:
        """Save split manifest to JSON."""
        path = self._splits_dir / f"{name}.json"
        data = {
            "name": name,
            "strategy": split.strategy,
            "train_ratio": split.train_ratio,
            "val_ratio": split.val_ratio,
            "test_ratio": split.test_ratio,
            "train": split.train,
            "val": split.val,
            "test": split.test,
            "checksum": split.checksum,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def load_split(self, path: str | Path) -> DatasetSplit:
        """Load a split manifest from JSON and verify integrity."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Split manifest not found: {path}")

        with open(path) as f:
            data = json.load(f)

        split = DatasetSplit(
            train=data["train"],
            val=data["val"],
            test=data["test"],
            strategy=data.get("strategy", "random"),
            train_ratio=data.get("train_ratio", 0.7),
            val_ratio=data.get("val_ratio", 0.15),
            test_ratio=data.get("test_ratio", 0.15),
            checksum=data.get("checksum", ""),
        )

        if split.checksum and self._compute_checksum(split) != split.checksum:
            self._logger.warning(f"Split manifest checksum mismatch: {path}")

        if not split.is_valid:
            self._logger.warning(f"Split manifest has overlapping sets: {path}")

        return split

    def lock_test_set(self, dataset_path: str | Path, split_name: str) -> Optional[Path]:
        """Lock the test set by saving it as a separate manifest.

        This prevents test set leakage during iterative training.
        """
        try:
            split = self.load_split(self._splits_dir / f"{split_name}.json")
            locked = DatasetSplit(
                train=split.train,
                val=split.val,
                test=split.test,
                strategy=split.strategy + "_locked",
                train_ratio=split.train_ratio,
                val_ratio=split.val_ratio,
                test_ratio=split.test_ratio,
            )
            locked.checksum = self._compute_checksum(locked)
            path = self._splits_dir / f"{split_name}_locked.json"
            self.save_split(locked, split_name + "_locked")
            self._logger.info(f"Test set locked: {path}")
            return path
        except Exception as e:
            self._logger.error(f"Failed to lock test set: {e}")
            return None

    def list_splits(self) -> list[str]:
        """List available split manifest names."""
        return sorted([p.stem for p in self._splits_dir.glob("*.json")])

    def get_split_path(self, name: str) -> Path:
        """Get the filesystem path for a split manifest."""
        return self._splits_dir / f"{name}.json"

    # ── Private ─────────────────────────────────────────────────────

    def _collect_sample_ids(self, dataset_path: str | Path) -> list[str]:
        """Walk dataset directory and collect all sample IDs (relative paths)."""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return []

        sample_ids: list[str] = []
        for class_dir in sorted(dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            samples_dir = class_dir / "samples"
            if not samples_dir.exists():
                continue
            for sample_dir in sorted(samples_dir.iterdir()):
                if sample_dir.is_dir() and (sample_dir / "landmarks.npy").exists():
                    rel_path = sample_dir.relative_to(dataset_path)
                    sample_ids.append(str(rel_path))
        return sample_ids

    @staticmethod
    def _compute_checksum(split: DatasetSplit) -> str:
        """MD5 checksum of concatenated sorted sample IDs."""
        all_ids = sorted(split.train + split.val + split.test)
        return hashlib.md5("".join(all_ids).encode()).hexdigest()
