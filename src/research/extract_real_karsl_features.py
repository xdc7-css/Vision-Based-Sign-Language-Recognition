"""Extract genuine multimodal MediaPipe landmarks from authentic KArSL-502 MP4 videos.

Pipeline:
1. Decode each MP4 video using OpenCV
2. Resize frames for optimal CPU MediaPipe inference
3. Extract 184 keypoints (Pose 6, Face 136, Left Hand 21, Right Hand 21) = 736 features per frame
4. Apply missing landmark mask (0 where unobserved)
5. Uniformly resample to fixed temporal length T=30
6. Save features to data/research/karsl/features/karsl_features.npz
7. Generate Leave-One-Signer-Out (LOSO) splits and updated dataset_manifest.json
"""

import os
import csv
import json
import logging
import time
from pathlib import Path
import cv2
import numpy as np

from src.infrastructure.recognition.karsl_recognizer import LandmarkerProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_real_karsl")

RAW_DIR = Path("data/research/karsl/raw")
FEAT_DIR = Path("data/research/karsl/features")
META_DIR = Path("data/research/karsl/metadata")
SPLITS_DIR = Path("data/research/karsl/splits")
MANIFEST_CSV = Path("data/research/karsl/manifests/real_10_signs.csv")

TARGET_SEQ_LEN = 30
FEAT_DIM = 736

SIGNERS = ["Signer01", "Signer02", "Signer03"]

CLASSES_10 = [
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

SIGN_ID_TO_INDEX = {c["sign_id"]: c["index"] for c in CLASSES_10}


def resample_sequence(kps: np.ndarray, target_len: int = TARGET_SEQ_LEN) -> np.ndarray:
    """Linear temporal interpolation of (T_raw, 736) sequence to (target_len, 736)."""
    seq_len = kps.shape[0]
    if seq_len == target_len:
        return kps.astype(np.float32)

    indices = np.linspace(0, seq_len - 1, target_len)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.clip(np.ceil(indices).astype(int), 0, seq_len - 1)
    alpha = (indices - idx_floor)[:, np.newaxis].astype(np.float32)

    sampled = (1.0 - alpha) * kps[idx_floor] + alpha * kps[idx_ceil]
    return sampled.astype(np.float32)


def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read manifest entries
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}")

    entries = []
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)

    logger.info(f"Loaded {len(entries)} video entries from manifest {MANIFEST_CSV}")

    # 2. Initialize LandmarkerProcessor
    processor = LandmarkerProcessor("data/models/karsl")

    all_sequences = []
    all_labels = []
    all_signers = []
    manifest_records = []

    start_total = time.time()
    for idx, entry in enumerate(entries):
        v_rel = entry["video_file"]
        v_path = Path(v_rel)
        if not v_path.exists():
            logger.warning(f"Video file missing: {v_path}")
            continue

        sign_id = entry["sign_id"]
        signer_id = entry["signer_id"]
        class_idx = SIGN_ID_TO_INDEX.get(sign_id, 0)
        sample_id = f"karsl_real_{sign_id}_{signer_id}_{Path(entry['filename']).stem}"

        cap = cv2.VideoCapture(str(v_path))
        raw_kps = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            h, w = frame.shape[:2]
            scale = 640.0 / w
            small = cv2.resize(frame, (640, int(h * scale)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            kps, _ = processor.extract_frame_keypoints(rgb)
            raw_kps.append(kps.flatten())
        cap.release()

        if not raw_kps:
            logger.warning(f"No frames read from {v_path}")
            continue

        kps_arr = np.array(raw_kps, dtype=np.float32)
        resampled_kps = resample_sequence(kps_arr, TARGET_SEQ_LEN)

        all_sequences.append(resampled_kps)
        all_labels.append(class_idx)
        all_signers.append(signer_id)

        manifest_records.append({
            "sample_id": sample_id,
            "sign_id": sign_id,
            "sign_name_ar": entry["name_ar"],
            "sign_name_en": entry["name_en"],
            "class_index": class_idx,
            "signer_id": signer_id,
            "filename": entry["filename"],
            "video_file": v_rel,
            "num_frames": TARGET_SEQ_LEN,
            "original_frames": len(raw_kps),
            "feature_dim": FEAT_DIM,
            "data_source": "karsl_502_authentic_video",
            "provenance": "KArSL-502 Official Dataset (Sidig et al., ACM IMWUT 2021)",
            "sha256": entry["sha256"],
        })

        if (idx + 1) % 20 == 0 or idx == len(entries) - 1:
            elapsed = time.time() - start_total
            fps_avg = (idx + 1) / elapsed
            logger.info(f"Processed {idx + 1}/{len(entries)} videos ({fps_avg:.1f} vids/s)")

    processor.close()

    # 3. Save feature arrays
    X_arr = np.array(all_sequences, dtype=np.float32)  # (N, 30, 736)
    y_arr = np.array(all_labels, dtype=np.int64)        # (N,)
    s_arr = np.array(all_signers, dtype=object)         # (N,)

    logger.info(f"Feature tensor shape: {X_arr.shape}, labels: {y_arr.shape}, signers: {s_arr.shape}")

    # Save to both standard path and real-specific path
    features_real = FEAT_DIR / "karsl_real_features.npz"
    features_main = FEAT_DIR / "karsl_features.npz"

    for p in [features_real, features_main]:
        np.savez_compressed(
            p,
            sequences=X_arr,
            labels=y_arr,
            signers=s_arr,
        )
        logger.info(f"Saved features to {p}")

    # 4. Save metadata manifests
    manifest_file = META_DIR / "dataset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved manifest to {manifest_file}")

    classes_file = META_DIR / "karsl_subset_classes.json"
    with open(classes_file, "w", encoding="utf-8") as f:
        json.dump(CLASSES_10, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved classes to {classes_file}")

    # 5. Generate LOSO splits (Signer01 vs Signer02 vs Signer03)
    loso_splits = {}
    for held_out in SIGNERS:
        train_indices = [i for i, s in enumerate(all_signers) if s != held_out]
        test_indices = [i for i, s in enumerate(all_signers) if s == held_out]
        loso_splits[f"test_{held_out}"] = {
            "held_out_signer": held_out,
            "train_signers": [s for s in SIGNERS if s != held_out],
            "train_count": len(train_indices),
            "test_count": len(test_indices),
            "train_indices": train_indices,
            "test_indices": test_indices,
        }

    splits_file = SPLITS_DIR / "loso_splits.json"
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(loso_splits, f, indent=2)
    logger.info(f"Saved LOSO splits to {splits_file}")


if __name__ == "__main__":
    main()

