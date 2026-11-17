"""KArSL-502 Dataset Management and Preparation for Scientific Research.

Manages:
- Official KArSL-502 label lookup and metadata resolution
- Selection of representative 12-class research vocabulary subset
- Signer identity management (Signer01, Signer02, Signer03)
- Importer for raw KArSL video recordings / frame directories
- Multi-signer benchmark dataset generation and feature extraction
- Train/Test/Validation manifest generation with Leave-One-Signer-Out (LOSO) splits
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("arsl.research.dataset")

# Authentic 10 research sign classes from KArSL-502 (Numbers 0-9)
SELECTED_RESEARCH_CLASSES: list[dict[str, Any]] = [
    {"sign_id": "0001", "index": 0, "name_ar": "صفر", "name_en": "0", "category": "number"},
    {"sign_id": "0002", "index": 1, "name_ar": "واحد", "name_en": "1", "category": "number"},
    {"sign_id": "0003", "index": 2, "name_ar": "اثنان", "name_en": "2", "category": "number"},
    {"sign_id": "0004", "index": 3, "name_ar": "ثلاثة", "name_en": "3", "category": "number"},
    {"sign_id": "0005", "index": 4, "name_ar": "أربعة", "name_en": "4", "category": "number"},
    {"sign_id": "0006", "index": 5, "name_ar": "خمسة", "name_en": "5", "category": "number"},
    {"sign_id": "0007", "index": 6, "name_ar": "ستة", "name_en": "6", "category": "number"},
    {"sign_id": "0008", "index": 7, "name_ar": "سبعة", "name_en": "7", "category": "number"},
    {"sign_id": "0009", "index": 8, "name_ar": "ثمانية", "name_en": "8", "category": "number"},
    {"sign_id": "0010", "index": 9, "name_ar": "تسعة", "name_en": "9", "category": "number"},
]

SIGNERS: list[str] = ["Signer01", "Signer02", "Signer03"]
SEQ_LEN: int = 30  # Fixed temporal frames per sample
FEAT_DIM: int = 736  # 184 keypoints * 4 (x, y, z, visibility)


@dataclass
class DatasetSample:
    sample_id: str
    sign_id: str
    sign_name_ar: str
    sign_name_en: str
    class_index: int
    signer_id: str
    repetition: int
    num_frames: int
    feature_dim: int
    data_source: str


class KArSLDatasetManager:
    """Manages KArSL-502 research data, imports, and cross-signer splits."""

    def __init__(self, base_dir: Path | str = "data/research/karsl") -> None:
        self.base_dir = Path(base_dir)
        self.meta_dir = self.base_dir / "metadata"
        self.raw_dir = self.base_dir / "raw"
        self.feat_dir = self.base_dir / "features"
        self.splits_dir = self.base_dir / "splits"
        self.eval_dir = self.base_dir / "eval"

        for d in [self.meta_dir, self.raw_dir, self.feat_dir, self.splits_dir, self.eval_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.labels_file = Path("data/models/karsl/labels.json")
        self.classes = SELECTED_RESEARCH_CLASSES
        self._ensure_metadata_saved()

    def _ensure_metadata_saved(self) -> None:
        """Persist official subset class definitions."""
        subset_file = self.meta_dir / "karsl_subset_classes.json"
        with open(subset_file, "w", encoding="utf-8") as f:
            json.dump(self.classes, f, ensure_ascii=False, indent=2)

    def get_class_names_ar(self) -> list[str]:
        return [c["name_ar"] for c in self.classes]

    def get_class_names_en(self) -> list[str]:
        return [c["name_en"] for c in self.classes]

    def get_status(self) -> dict[str, Any]:
        """Return dataset readiness, provenance, and counts."""
        features_file = self.feat_dir / "karsl_features.npz"
        manifest_file = self.meta_dir / "dataset_manifest.json"
        splits_file = self.splits_dir / "loso_splits.json"

        is_prepared = features_file.exists() and manifest_file.exists()
        sample_count = 0
        signer_counts = {s: 0 for s in SIGNERS}
        class_counts: dict[str, int] = {}
        data_source = "unknown"
        is_authentic = False

        if is_prepared:
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                sample_count = len(manifest)
                for item in manifest:
                    s_id = item.get("signer_id", "Unknown")
                    if s_id in signer_counts:
                        signer_counts[s_id] += 1
                    c_name = item.get("sign_name_ar", "Unknown")
                    class_counts[c_name] = class_counts.get(c_name, 0) + 1
                    data_source = item.get("data_source", data_source)
                is_authentic = (data_source == "karsl_502_authentic_video")
            except Exception as e:
                logger.warning(f"Error reading manifest: {e}")

        raw_videos = list(self.raw_dir.glob("**/*.mp4")) + list(self.raw_dir.glob("**/*.avi"))

        return {
            "is_prepared": is_prepared,
            "is_authentic": is_authentic,
            "data_source": data_source,
            "provenance": (
                "KArSL-502 Official Dataset (Sidig et al., ACM IMWUT 2021)"
                if is_authentic
                else "SYNTHETIC / DEVELOPMENT (Artificially Generated)"
            ),
            "total_samples": sample_count,
            "num_classes": len(self.classes),
            "classes": [c["name_ar"] for c in self.classes],
            "signers": SIGNERS,
            "samples_per_signer": signer_counts,
            "class_counts": class_counts,
            "has_raw_videos": len(raw_videos) > 0,
            "raw_video_count": len(raw_videos),
            "has_splits": splits_file.exists(),
            "feature_dim": FEAT_DIM,
            "seq_len": SEQ_LEN,
        }

    def prepare_dataset(
        self,
        samples_per_sign_per_signer: int = 15,
        force_rebuild: bool = False,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Prepare research dataset from authentic raw videos if present, else fallback."""
        features_file = self.feat_dir / "karsl_features.npz"
        manifest_file = self.meta_dir / "dataset_manifest.json"

        if features_file.exists() and manifest_file.exists() and not force_rebuild:
            logger.info("Research dataset already prepared. Returning cached status.")
            return self.get_status()

        # If authentic videos are present, extract from them
        raw_manifest_csv = self.base_dir / "manifests" / "real_10_signs.csv"
        raw_videos = list(self.raw_dir.glob("**/*.mp4"))
        if raw_manifest_csv.exists() and len(raw_videos) >= 20:
            logger.info(f"Extracting features from {len(raw_videos)} authentic KArSL videos...")
            from src.research.extract_real_karsl_features import main as extract_main
            extract_main()
            return self.get_status()

        logger.warning("No authentic raw videos found. Preparing synthetic development dataset...")
        return self.prepare_synthetic_development_dataset(
            samples_per_sign_per_signer=samples_per_sign_per_signer,
            random_seed=random_seed,
        )

    def prepare_synthetic_development_dataset(
        self,
        samples_per_sign_per_signer: int = 15,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Development fallback only: generates synthetic kinematic trajectories."""
        features_file = self.feat_dir / "karsl_features.npz"
        manifest_file = self.meta_dir / "dataset_manifest.json"

        logger.info(
            f"Generating multi-signer synthetic development dataset: {len(self.classes)} signs x "
            f"{len(SIGNERS)} signers x {samples_per_sign_per_signer} repetitions."
        )

        rng = np.random.RandomState(random_seed)

        all_sequences: list[np.ndarray] = []
        all_labels: list[int] = []
        all_signers: list[str] = []
        manifest_records: list[dict[str, Any]] = []

        total_samples = len(self.classes) * len(SIGNERS) * samples_per_sign_per_signer
        logger.info(f"Preparing {total_samples} synthetic temporal feature sequences...")

        for class_idx, class_info in enumerate(self.classes):
            sign_id = class_info["sign_id"]
            name_ar = class_info["name_ar"]
            name_en = class_info["name_en"]

            base_phase = (class_idx * 2 * np.pi) / len(self.classes)
            base_freq = 0.8 + 0.15 * (class_idx % 4)

            for signer_idx, signer_id in enumerate(SIGNERS):
                signer_scale = 0.92 + 0.08 * signer_idx
                signer_tempo = 0.95 + 0.05 * (signer_idx - 1)
                signer_offset = np.array([0.02 * (signer_idx - 1), 0.01 * signer_idx, 0.0])

                for rep in range(samples_per_sign_per_signer):
                    sample_id = f"karsl_synth_{sign_id}_{signer_id}_rep{rep:02d}"

                    seq = self._generate_synthetic_dev_trajectory(
                        seq_len=SEQ_LEN,
                        feat_dim=FEAT_DIM,
                        class_idx=class_idx,
                        base_phase=base_phase,
                        base_freq=base_freq * signer_tempo,
                        scale=signer_scale,
                        offset=signer_offset,
                        rng=rng,
                    )

                    all_sequences.append(seq)
                    all_labels.append(class_idx)
                    all_signers.append(signer_id)

                    manifest_records.append(
                        asdict(
                            DatasetSample(
                                sample_id=sample_id,
                                sign_id=sign_id,
                                sign_name_ar=name_ar,
                                sign_name_en=name_en,
                                class_index=class_idx,
                                signer_id=signer_id,
                                repetition=rep,
                                num_frames=SEQ_LEN,
                                feature_dim=FEAT_DIM,
                                data_source="karsl_502_synthetic_dev",
                            )
                        )
                    )

        X_arr = np.array(all_sequences, dtype=np.float32)
        y_arr = np.array(all_labels, dtype=np.int64)
        s_arr = np.array(all_signers, dtype=object)

        np.savez_compressed(
            features_file,
            sequences=X_arr,
            labels=y_arr,
            signers=s_arr,
        )

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_records, f, ensure_ascii=False, indent=2)

        self._generate_splits(manifest_records)
        return self.get_status()

    def _generate_synthetic_dev_trajectory(
        self,
        seq_len: int,
        feat_dim: int,
        class_idx: int,
        base_phase: float,
        base_freq: float,
        scale: float,
        offset: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Create a smooth 3D kinematic trajectory conforming to MediaPipe 184-kps topology."""
        seq = np.zeros((seq_len, feat_dim), dtype=np.float32)
        t = np.linspace(0, 1, seq_len)

        # Bell-shaped velocity envelope (rest -> sign peak -> rest)
        envelope = np.sin(np.pi * t) ** 1.8

        # Hand movement modulation
        x_signal = np.sin(2 * np.pi * base_freq * t + base_phase) * envelope * scale
        y_signal = np.cos(2 * np.pi * base_freq * t + base_phase) * envelope * scale
        z_signal = 0.2 * np.sin(4 * np.pi * base_freq * t) * envelope

        # Construct 184 keypoints with visibility = 1.0
        # Layout: 6 pose kps (0..5), 136 face kps (6..141), 21 right hand (142..162), 21 left hand (163..183)
        for frame_idx in range(seq_len):
            frame_feats = np.zeros(feat_dim, dtype=np.float32)
            noise = rng.normal(0, 0.005, size=feat_dim).astype(np.float32)

            for kp_idx in range(184):
                offset_idx = kp_idx * 4

                if kp_idx < 6:
                    # Pose (shoulders, elbows, wrists)
                    side = 1.0 if kp_idx % 2 == 0 else -1.0
                    frame_feats[offset_idx] = side * 0.25 + offset[0] + x_signal[frame_idx] * 0.1
                    frame_feats[offset_idx + 1] = 0.4 + offset[1] + y_signal[frame_idx] * 0.1
                    frame_feats[offset_idx + 2] = offset[2]
                    frame_feats[offset_idx + 3] = 0.98  # visibility
                elif kp_idx < 142:
                    # Face (136 contour points)
                    phi = (kp_idx - 6) * (2 * np.pi / 136)
                    frame_feats[offset_idx] = 0.12 * np.cos(phi) + offset[0] * 0.2
                    frame_feats[offset_idx + 1] = 0.15 * np.sin(phi) + offset[1] * 0.2
                    frame_feats[offset_idx + 2] = offset[2] * 0.2
                    frame_feats[offset_idx + 3] = 0.95
                elif kp_idx < 163:
                    # Right Hand (primary active articulators for single-hand or dual-hand)
                    h_idx = kp_idx - 142
                    frame_feats[offset_idx] = 0.2 + x_signal[frame_idx] + (h_idx % 5) * 0.015
                    frame_feats[offset_idx + 1] = 0.5 + y_signal[frame_idx] + (h_idx // 5) * 0.02
                    frame_feats[offset_idx + 2] = z_signal[frame_idx]
                    frame_feats[offset_idx + 3] = 0.99
                else:
                    # Left Hand
                    h_idx = kp_idx - 163
                    # Active for dual-hand signs (e.g. السلام عليكم, أهلا وسهلا, صفر)
                    is_dual = class_idx in [0, 4, 10, 11]
                    dual_scale = 0.8 if is_dual else 0.1
                    frame_feats[offset_idx] = -0.2 - x_signal[frame_idx] * dual_scale + (h_idx % 5) * 0.015
                    frame_feats[offset_idx + 1] = 0.5 + y_signal[frame_idx] * dual_scale + (h_idx // 5) * 0.02
                    frame_feats[offset_idx + 2] = z_signal[frame_idx] * dual_scale
                    frame_feats[offset_idx + 3] = 0.99 if is_dual else 0.70

            seq[frame_idx] = frame_feats + noise

        return seq

    def _generate_splits(self, manifest: list[dict[str, Any]]) -> None:
        """Create Leave-One-Signer-Out (LOSO) and Stratified Split manifests."""
        loso_splits: dict[str, Any] = {}

        for test_signer in SIGNERS:
            train_signers = [s for s in SIGNERS if s != test_signer]
            train_idx = [i for i, item in enumerate(manifest) if item["signer_id"] in train_signers]
            test_idx = [i for i, item in enumerate(manifest) if item["signer_id"] == test_signer]

            loso_splits[f"test_{test_signer}"] = {
                "train_signers": train_signers,
                "test_signer": test_signer,
                "train_indices": train_idx,
                "test_indices": test_idx,
                "train_count": len(train_idx),
                "test_count": len(test_idx),
            }

        with open(self.splits_dir / "loso_splits.json", "w", encoding="utf-8") as f:
            json.dump(loso_splits, f, ensure_ascii=False, indent=2)

        # Stratified random split (70% train, 15% val, 15% test) for general benchmarks
        n = len(manifest)
        indices = np.random.RandomState(42).permutation(n).tolist()
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        strat_split = {
            "train_indices": indices[:n_train],
            "val_indices": indices[n_train : n_train + n_val],
            "test_indices": indices[n_train + n_val :],
            "train_count": n_train,
            "val_count": n_val,
            "test_count": len(indices) - (n_train + n_val),
        }

        with open(self.splits_dir / "stratified_split.json", "w", encoding="utf-8") as f:
            json.dump(strat_split, f, ensure_ascii=False, indent=2)

    def load_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load extracted feature array, labels, and signers."""
        features_file = self.feat_dir / "karsl_features.npz"
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found at {features_file}. Run prepare_dataset() first.")
        data = np.load(features_file, allow_pickle=True)
        return data["sequences"], data["labels"], data["signers"]

    def load_loso_splits(self) -> dict[str, Any]:
        """Load LOSO split specifications."""
        split_file = self.splits_dir / "loso_splits.json"
        with open(split_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_inner_train_val_split(
        self,
        loso_fold: str = "test_Signer03",
        inner_val_ratio: float = 0.2,
        val_fraction: Optional[float] = None,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Return strictly isolated inner-training and inner-validation partitions from training signers only.

        The held-out test signer is NEVER included in training or validation partitions.
        Zero data leakage guaranteed.
        """
        if val_fraction is not None:
            inner_val_ratio = val_fraction
        from sklearn.model_selection import StratifiedShuffleSplit

        X_all, y_all, signers_all = self.load_features()
        splits = self.load_loso_splits()
        if loso_fold not in splits:
            raise KeyError(f"Fold '{loso_fold}' not found in splits: {list(splits.keys())}")

        fold_info = splits[loso_fold]
        train_idx = np.array(fold_info["train_indices"])
        test_idx = np.array(fold_info["test_indices"])
        test_signer = fold_info.get("test_signer", fold_info.get("held_out_signer", "Unknown"))
        train_signers = fold_info.get("train_signers", [])

        # Invariant 1: Ensure zero overlap between outer train and outer test
        if len(np.intersect1d(train_idx, test_idx)) > 0:
            raise AssertionError("FATAL LEAKAGE: train_indices and test_indices overlap!")
        if test_signer in train_signers:
            raise AssertionError(f"FATAL LEAKAGE: test_signer {test_signer} present in train_signers!")

        X_train_full = X_all[train_idx]
        y_train_full = y_all[train_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_ratio, random_state=random_state)
        inner_train_rel, inner_val_rel = next(sss.split(X_train_full, y_train_full))

        inner_train_idx = train_idx[inner_train_rel]
        inner_val_idx = train_idx[inner_val_rel]

        # Invariant 2: Inner partitions must strictly exclude held-out test signer
        assert np.intersect1d(inner_train_idx, test_idx).size == 0, "Inner train contaminated with test indices!"
        assert np.intersect1d(inner_val_idx, test_idx).size == 0, "Inner val contaminated with test indices!"
        assert test_signer not in signers_all[inner_train_idx], f"Test signer {test_signer} found in inner train!"
        assert test_signer not in signers_all[inner_val_idx], f"Test signer {test_signer} found in inner val!"

        return {
            "test_signer": test_signer,
            "train_signers": train_signers,
            "inner_train_indices": inner_train_idx,
            "inner_val_indices": inner_val_idx,
            "outer_test_indices": test_idx,
            "X_inner_train": X_all[inner_train_idx],
            "y_inner_train": y_all[inner_train_idx],
            "X_inner_val": X_all[inner_val_idx],
            "y_inner_val": y_all[inner_val_idx],
        }


