"""Multimodal gesture training pipeline — trains MLP, SVM, and RF on dual-hand and face features."""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.domain.multimodal import (
    FEATURE_DIM_HANDS_FACE,
    MULTIMODAL_SCHEMA_VERSION,
    ModalityMode,
)
from src.infrastructure.vision.feature_extractor import MultimodalFeatureExtractor


logger = logging.getLogger(__name__)


def inspect_multimodal_dataset(
    dataset_path: str | Path = "data/datasets/words",
    allowed_classes: Optional[set[str] | list[str]] = None,
) -> dict[str, Any]:
    """Inspect dataset directory for multimodal (v2) vs legacy (v1) samples.

    Returns:
        dict with detailed counts of multimodal vs legacy samples and trainability status.
    """
    dataset_path = Path(dataset_path)
    if (dataset_path / "words").exists() and not any(
        (dataset_path / d).is_dir() and (dataset_path / d / "samples").exists()
        for d in dataset_path.iterdir()
        if (dataset_path / d).is_dir()
    ):
        dataset_path = dataset_path / "words"

    if not dataset_path.exists():
        return {
            "total_classes": 0,
            "multimodal_class_counts": {},
            "legacy_class_counts": {},
            "accepted_classes": {},
            "insufficient_classes": {},
            "legacy_only_classes": {},
            "empty_classes": {},
            "is_trainable": False,
            "reason": f"المجلد غير موجود: {dataset_path}",
        }

    target_allowed = None
    if allowed_classes is not None:
        target_allowed = {str(c).replace(" ", "_") for c in allowed_classes} | {str(c) for c in allowed_classes}

    mm_counts: dict[str, int] = {}
    legacy_counts: dict[str, int] = {}
    empty_classes: dict[str, int] = {}

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if target_allowed is not None and class_name not in target_allowed:
            continue

        samples_dir = class_dir / "samples"
        mm_count = 0
        leg_count = 0

        if samples_dir.exists():
            for sample_dir in samples_dir.iterdir():
                if not sample_dir.is_dir():
                    continue

                meta_file = sample_dir / "metadata.json"
                lm_file = sample_dir / "landmarks.npy"

                if not lm_file.exists():
                    continue

                is_mm = False
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            m = json.load(f)
                        if m.get("schema_version") == MULTIMODAL_SCHEMA_VERSION:
                            is_mm = True
                    except Exception:
                        pass

                # If not tagged in metadata, check array shape directly
                if not is_mm:
                    try:
                        arr = np.load(lm_file, allow_pickle=False)
                        if arr.ndim == 2 and arr.shape[1] in (134, 206, 230):
                            is_mm = True
                    except Exception:
                        pass

                if is_mm:
                    mm_count += 1
                else:
                    leg_count += 1

        mm_counts[class_name] = mm_count
        legacy_counts[class_name] = leg_count
        if mm_count == 0 and leg_count == 0:
            empty_classes[class_name] = 0

    accepted = {k: v for k, v in mm_counts.items() if v >= 2}
    insufficient = {k: v for k, v in mm_counts.items() if v == 1}
    legacy_only = {k: legacy_counts[k] for k, v in mm_counts.items() if v == 0 and legacy_counts[k] > 0}

    is_trainable = len(accepted) >= 2
    if is_trainable:
        reason = f"جاهز للتدريب المتعدد الوسائط: تم العثور على {len(accepted)} فئات مستوفية للشروط (عينتان فأكثر لكل فئة)."
    else:
        reason = (
            f"لا تتوفر بيانات متعددة الوسائط كافية:\n"
            f"• الفئات المستوفية (عينتان متعددتا الوسائط فأكثر): {len(accepted)} فئة.\n"
            f"• فئات تحوي عينات أحادية اليد فقط (Legacy v1): {len(legacy_only)} فئة.\n"
            f"يتطلب النموذج تدريب عينتين على الأقل لكل فئة بصيغة اليدين + الوجه (v2_multimodal)."
        )

    return {
        "total_classes": len(mm_counts),
        "multimodal_class_counts": mm_counts,
        "legacy_class_counts": legacy_counts,
        "accepted_classes": accepted,
        "qualifying_classes": accepted,
        "insufficient_classes": insufficient,
        "legacy_only_classes": legacy_only,
        "legacy_classes": legacy_only,
        "empty_classes": empty_classes,
        "is_trainable": is_trainable,
        "reason": reason,
        "multimodal_samples_count": sum(mm_counts.values()),
        "legacy_samples_count": sum(legacy_counts.values()),
    }


def load_multimodal_dataset(
    dataset_path: str | Path = "data/datasets/words",
    allowed_classes: Optional[set[str] | list[str]] = None,
    expected_dim: int = FEATURE_DIM_HANDS_FACE,
    min_classes: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load multimodal samples from disk, strictly filtering for v2 multimodal schema.

    Returns:
        X: (N, expected_dim) float32 features.
        y: (N,) string class labels.
        classes: sorted list of included class names.
    """
    dataset_path = Path(dataset_path)
    if (dataset_path / "words").exists() and not any(
        (dataset_path / d).is_dir() and (dataset_path / d / "samples").exists()
        for d in dataset_path.iterdir()
        if (dataset_path / d).is_dir()
    ):
        dataset_path = dataset_path / "words"

    inspection = inspect_multimodal_dataset(dataset_path, allowed_classes=allowed_classes)

    if allowed_classes is None:
        target_classes = sorted(inspection["accepted_classes"].keys())
    else:
        target_classes = sorted(
            set(allowed_classes).intersection(inspection["accepted_classes"].keys())
        )

    if len(target_classes) < min_classes:
        raise ValueError(
            f"Multimodal training requires at least {min_classes} distinct classes with >=2 multimodal samples each.\n"
            f"{inspection['reason']}"
        )

    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    for class_name in target_classes:
        class_dir = dataset_path / class_name
        samples_dir = class_dir / "samples"
        if not samples_dir.exists():
            continue

        for sample_dir in sorted(samples_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            lm_file = sample_dir / "landmarks.npy"
            meta_file = sample_dir / "metadata.json"

            if not lm_file.exists():
                continue

            # Verify schema
            is_valid_mm = False
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("schema_version") == MULTIMODAL_SCHEMA_VERSION:
                        is_valid_mm = True
                except Exception:
                    pass

            try:
                arr = np.load(lm_file, allow_pickle=False)
                if not is_valid_mm and arr.ndim == 2 and arr.shape[1] == expected_dim:
                    is_valid_mm = True

                if not is_valid_mm:
                    continue

                # Handle 1D, 2D, or 3D
                if arr.ndim == 1 and arr.shape[0] == expected_dim:
                    vec = arr
                elif arr.ndim == 2:
                    mid = arr.shape[0] // 2
                    vec = arr[mid]
                else:
                    continue

                if vec.shape[0] == expected_dim and not np.isnan(vec).any():
                    X_list.append(vec.astype(np.float32))
                    y_list.append(class_name)

            except Exception as e:
                logger.debug(f"Error loading sample {sample_dir}: {e}")

    if not X_list:
        raise ValueError(f"No valid multimodal samples found in {dataset_path} matching schema v2.")

    return np.array(X_list, dtype=np.float32), np.array(y_list), target_classes


def train_multimodal_model(
    dataset_path: str | Path | None = None,
    output_dir: str | Path = "data/models/multimodal",
    model_type: str = "mlp",
    epochs: int = 60,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    allowed_classes: Optional[set[str] | list[str]] = None,
    dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train a multimodal gesture classifier and persist complete checkpoint with metrics."""
    actual_path = dataset_dir if dataset_dir is not None else (dataset_path or "data/datasets/words")
    actual_path = Path(actual_path)
    if not (actual_path / "samples").exists() and (actual_path / "words").exists():
        actual_path = actual_path / "words"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load multimodal dataset
    X, y, classes = load_multimodal_dataset(actual_path, allowed_classes=allowed_classes)
    n_samples, feat_dim = X.shape

    # 2. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(classes)

    # 3. Stratified Train / Val split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    try:
        train_idx, val_idx = next(sss.split(X, y_encoded))
    except ValueError:
        # Fallback if class counts too small for stratified split
        perm = np.random.RandomState(42).permutation(n_samples)
        split_point = max(1, int(n_samples * (1 - val_split)))
        train_idx, val_idx = perm[:split_point], perm[split_point:]

    X_train, y_train = X[train_idx], y_encoded[train_idx]
    X_val, y_val = X[val_idx], y_encoded[val_idx]

    # 4. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 5. Build and train classifier
    start_time = time.perf_counter()
    best_clf: Any = None
    best_name = "mlp"

    if model_type.lower() == "svm":
        clf = SVC(kernel="rbf", C=10.0, probability=True, random_state=42)
        clf.fit(X_train_scaled, y_train)
        best_clf = clf
        best_name = "svm"
        if progress_callback:
            progress_callback(epochs, epochs, 0.01, 0.01)

    elif model_type.lower() == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        clf.fit(X_train_scaled, y_train)
        best_clf = clf
        best_name = "rf"
        if progress_callback:
            progress_callback(epochs, epochs, 0.01, 0.01)

    else:
        # MLP Classifier with step-by-step progress
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=1,
            warm_start=True,
            learning_rate_init=learning_rate,
            random_state=42,
        )

        all_classes_idx = np.arange(num_classes)
        train_loss = 0.0
        val_loss = 0.0

        for ep in range(1, epochs + 1):
            clf.partial_fit(X_train_scaled, y_train, classes=all_classes_idx)
            train_loss = float(clf.loss_)

            # Compute validation loss proxy
            val_probs = clf.predict_proba(X_val_scaled)
            eps = 1e-15
            val_probs = np.clip(val_probs, eps, 1 - eps)
            one_hot_val = np.zeros_like(val_probs)
            one_hot_val[np.arange(len(y_val)), y_val] = 1.0
            val_loss = float(-np.mean(np.sum(one_hot_val * np.log(val_probs), axis=1)))

            if progress_callback:
                progress_callback(ep, epochs, train_loss, val_loss)

        best_clf = clf
        best_name = "mlp"

    train_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 6. Evaluation on Validation Set
    y_pred = best_clf.predict(X_val_scaled)
    acc = float(accuracy_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_val, y_pred, labels=np.arange(num_classes)).tolist()

    # 7. Package and save checkpoint
    extractor = MultimodalFeatureExtractor(ModalityMode.TWO_HANDS_FACE)
    schema = extractor.get_schema(ModalityMode.TWO_HANDS_FACE)

    checkpoint = {
        "model": best_clf,
        "label_encoder": le,
        "scaler": scaler,
        "schema": schema,
        "feature_dim": feat_dim,
        "model_type": best_name,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "train_time_ms": train_elapsed_ms,
            "n_train_samples": len(train_idx),
            "n_val_samples": len(val_idx),
            "total_samples": n_samples,
            "num_classes": num_classes,
        },
        "classes": classes,
        "timestamp": time.time(),
    }

    model_path = output_dir / "static_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f)

    classes_path = output_dir / "classes.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint["metrics"], f, indent=2, ensure_ascii=False)

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": MULTIMODAL_SCHEMA_VERSION,
            "feature_dim": feat_dim,
            "classes": classes,
            "metrics": checkpoint["metrics"],
        }, f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "model_type": best_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "classes": classes,
        "num_classes": num_classes,
        "total_samples": n_samples,
        "metrics": checkpoint["metrics"],
    }
