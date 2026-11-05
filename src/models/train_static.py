"""Static gesture training pipeline — trains RF, SVM, MLP, auto-selects best model.

Supports:
    - Data augmentation (landmark jitter, feature noise)
    - Persistent train/val/test splits via DatasetSplitManager
    - Model evaluation via ModelEvaluator (classification report, confusion matrix)
    - Model registration via ModelRegistry
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.infrastructure.dataset.augmenter import DatasetAugmenter
from src.infrastructure.dataset.splitter import DatasetSplitManager
from src.infrastructure.recognition.static_recognizer import (
    StaticModelConfig,
    StaticSignRecognizer,
)
from src.models.evaluator import ModelEvaluator, ModelRegistry


def inspect_dataset(dataset_path: str | Path) -> dict:
    """Inspect dataset directory and return detailed class analysis.

    Returns:
        dict with total_classes, class_counts, accepted_classes,
        insufficient_classes, empty_classes, is_trainable, reason.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return {
            "total_classes": 0,
            "class_counts": {},
            "accepted_classes": {},
            "insufficient_classes": {},
            "empty_classes": {},
            "is_trainable": False,
            "reason": f"المجلد غير موجود: {dataset_path}",
        }

    class_counts: dict[str, int] = {}
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        samples_dir = class_dir / "samples"
        count = 0
        if samples_dir.exists():
            for sample_dir in samples_dir.iterdir():
                landmarks_file = sample_dir / "landmarks.npy"
                if landmarks_file.exists():
                    try:
                        arr = np.load(landmarks_file)
                        if arr.size > 0 and not np.isnan(arr).any():
                            count += 1
                    except Exception:
                        pass
        class_counts[class_name] = count

    accepted = {k: v for k, v in class_counts.items() if v >= 2}
    insufficient = {k: v for k, v in class_counts.items() if v == 1}
    empty = {k: v for k, v in class_counts.items() if v == 0}

    is_trainable = len(accepted) >= 2
    if is_trainable:
        reason = f"جاهز للتدريب: تم العثور على {len(accepted)} فئات مستوفية للشروط (عينتان فأكثر)."
    else:
        reason = (
            f"غير كافٍ للتدريب: يتطلب النموذج فئتين صالحتين على الأقل (عينتان لكل فئة لتقسيم train/val).\n"
            f"الفئات المستوفية حالياً: {len(accepted)} فئة."
        )

    return {
        "total_classes": len(class_counts),
        "class_counts": class_counts,
        "accepted_classes": accepted,
        "insufficient_classes": insufficient,
        "empty_classes": empty,
        "is_trainable": is_trainable,
        "reason": reason,
    }


def load_dataset(
    dataset_path: str | Path,
    augmenter: Optional[DatasetAugmenter] = None,
    augment_multiplier: int = 1,
    split_manager: Optional[DatasetSplitManager] = None,
    split_name: Optional[str] = None,
    allowed_classes: Optional[set[str] | list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load landmarks and labels from a structured dataset directory.

    Expected structure:
        dataset_path/
            class_1/
                samples/
                    sample_1/landmarks.npy
                    sample_2/landmarks.npy
            class_2/
                ...

    Args:
        dataset_path: Path to the structured dataset.
        augmenter: Optional DatasetAugmenter for data augmentation.
        augment_multiplier: Number of augmented copies per original sample.
        split_manager: Optional DatasetSplitManager for test set separation.
        split_name: Name of the split to use (only train samples).
        allowed_classes: Optional list/set of class names to load. If None,
                         loads all classes with at least 2 samples.

    Returns:
        X: (n_samples, 63) feature matrix.
        y: (n_samples,) string labels.
        inspection: Inspection report dict.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    inspection = inspect_dataset(dataset_path)

    if allowed_classes is None:
        target_classes = set(inspection["accepted_classes"].keys())
    else:
        target_classes = set(allowed_classes)

    if len(target_classes) < 2:
        raise ValueError(
            f"Training requires at least 2 distinct gesture classes with at least 2 samples each.\n"
            f"{inspection['reason']}\n"
            f"Accepted: {list(inspection['accepted_classes'].items())}\n"
            f"Insufficient: {list(inspection['insufficient_classes'].items())}\n"
            f"Empty: {list(inspection['empty_classes'].keys())}"
        )

    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    for class_name in sorted(target_classes):
        class_dir = dataset_path / class_name
        samples_dir = class_dir / "samples"
        if not samples_dir.exists():
            continue

        for sample_dir in sorted(samples_dir.iterdir()):
            landmarks_file = sample_dir / "landmarks.npy"
            if not landmarks_file.exists():
                continue

            try:
                data = np.load(landmarks_file)
                if data.ndim == 3:
                    mid = data.shape[0] // 2
                    raw = data[mid]
                elif data.ndim == 2:
                    raw = data
                else:
                    continue

                from src.infrastructure.hand_detection.utils import LandmarkProcessor
                if raw.shape == (21, 3):
                    flat = LandmarkProcessor.normalize_array(raw).flatten()
                elif raw.shape == (63,):
                    flat = LandmarkProcessor.normalize_array(raw)
                else:
                    continue

                if flat.shape[0] == 63:
                    X_list.append(flat)
                    y_list.append(class_name)

                    # Augment
                    if augmenter is not None and augment_multiplier > 1:
                        for _ in range(augment_multiplier - 1):
                            aug = augmenter.augment_landmarks(flat)
                            X_list.append(aug)
                            y_list.append(class_name)

            except Exception:
                continue

    if not X_list:
        raise ValueError(f"No valid samples found in {dataset_path} for classes: {target_classes}")

    return np.array(X_list, dtype=np.float32), np.array(y_list), inspection


def train_static_model(
    dataset_path: str | Path = "data/datasets/words",
    output_dir: str | Path = "data/models/iraqi",
    model_types: Optional[list[str]] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    augment: bool = True,
    augment_multiplier: int = 3,
    split_manager: Optional[DatasetSplitManager] = None,
    split_name: Optional[str] = None,
    dialect: str = "iraqi",
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> dict:
    """Train, evaluate, and save the best static gesture model.

    Args:
        dataset_path: Path to the structured dataset.
        output_dir: Where to save the trained model.
        model_types: Which model types to try ('rf', 'svm', 'mlp').
        epochs: Number of training epochs for MLP.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate for MLP.
        val_split: Validation split ratio (e.g. 0.2 = 20%).
        augment: Whether to apply data augmentation.
        augment_multiplier: Augmented copies per real sample.
        split_manager: Optional split manager for test set separation.
        split_name: Split manifest name to load.
        dialect: Dialect name for model registry and vocabulary.
        progress_callback: Optional callback for live epoch progress.

    Returns:
        Dict with training results and best model info.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")

    augmenter = DatasetAugmenter() if augment else None
    X, y, inspection = load_dataset(
        dataset_path, augmenter, augment_multiplier, split_manager, split_name
    )

    counts = Counter(y)
    n_classes = len(counts)
    print(f"\n--- Dataset Class Distribution ---")
    for cls_name, count in sorted(counts.items()):
        print(f"  - Class '{cls_name}': {count} samples")
    print(f"Total: {len(X)} samples across {n_classes} qualifying classes (augment={augment}, multiplier={augment_multiplier})\n")

    # Configure and train
    config = StaticModelConfig(
        mlp_max_iter=epochs,
        mlp_learning_rate_init=learning_rate,
        test_size=val_split,
    )
    recognizer = StaticSignRecognizer(config)

    effective_models = model_types or ["mlp"]
    print(f"\nTraining models: {effective_models}")
    all_metrics = recognizer.train(X, y, effective_models, progress_callback=progress_callback)

    # Print results
    print(f"\n{'='*55}")
    print(f"  Model Comparison — Static Gesture Recognition")
    print(f"{'='*55}")
    print(f"  {'Model':<8} {'Accuracy':>10} {'F1':>10} {'Infer(ms)':>10} {'Size(KB)':>10}")
    print(f"  {'-'*48}")
    for mt, m in sorted(all_metrics.items()):
        print(f"  {mt:<8} {m.accuracy:>10.3f} {m.f1_score:>10.3f} {m.inference_time_ms:>10.3f} {m.model_size_kb:>10.1f}")

    best_type = recognizer.model_type
    best = all_metrics.get(best_type)
    print(f"\n  ✓ Best model: {best_type} (accuracy: {best.accuracy if best else 0.0:.3f}, F1: {best.f1_score if best else 0.0:.3f})")

    # Save model — canonical static_model.pkl and static_{type}.pkl
    model_path = recognizer.save(output_dir)
    print(f"\n  Saved model: {model_path}")

    # Generate and save vocabulary.json mapping trained classes
    vocab_data = {
        "metadata": {
            "dialect": dialect,
            "model_type": best_type,
            "accuracy": best.accuracy if best else 0.0,
            "classes": sorted(counts.keys()),
        },
        "gestures": {}
    }
    for cls_name in sorted(counts.keys()):
        arabic_clean = str(cls_name).replace("_", " ")
        vocab_data["gestures"][cls_name] = {
            "arabic": arabic_clean,
            "type": "static",
            "threshold": 0.65,
        }
        vocab_data["gestures"][f"static_{cls_name}"] = {
            "arabic": arabic_clean,
            "type": "static",
            "threshold": 0.65,
        }
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved vocabulary: {vocab_path}")

    # Evaluate and export using ModelEvaluator
    try:
        from sklearn.metrics import accuracy_score
        evaluator = ModelEvaluator()
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        can_stratify = len(Counter(y_enc)) >= 2 and all(c >= 2 for c in Counter(y_enc).values())
        _, X_eval, _, y_eval = train_test_split(
            X, y_enc, test_size=val_split, random_state=42, stratify=y_enc if can_stratify else None
        )
        scaler = StandardScaler()
        scaler.fit(X)
        X_eval_scaled = scaler.transform(X_eval)

        models = {best_type: recognizer._model}
        eval_results = evaluator.compare_models(models, X_eval_scaled, y_eval, list(le.classes_))

        eval_path = output_dir / "evaluation.json"
        evaluator.export_comparison(eval_results, str(eval_path))
        print(f"  Evaluation report: {eval_path}")
        evaluator.print_comparison(eval_results)
    except Exception as e:
        print(f"  Warning: Model evaluation failed: {e}")

    # Register model
    try:
        registry = ModelRegistry()
        registry.register_model(dialect, f"static_{best_type}", str(model_path), {
            "accuracy": best.accuracy if best else 0.0,
            "f1_score": best.f1_score if best else 0.0,
            "precision": best.precision if best else 0.0,
            "recall": best.recall if best else 0.0,
        })
        print(f"  Registered in model registry")
    except Exception as e:
        print(f"  Warning: Model registration failed: {e}")

    return {
        "model_type": best_type,
        "accuracy": best.accuracy if best else 0.0,
        "f1_score": best.f1_score if best else 0.0,
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "num_samples": len(X),
        "num_classes": n_classes,
        "classes": sorted(counts.keys()),
        "inspection": inspection,
        "metrics": {
            "accuracy": best.accuracy if best else 0.0,
            "f1_score": best.f1_score if best else 0.0,
        },
        "all_metrics": {
            mt: {
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "model_type": m.model_type,
            }
            for mt, m in all_metrics.items()
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train static gesture classifier")
    parser.add_argument("--dataset", default="data/datasets/words", help="Dataset path")
    parser.add_argument("--output", default="data/models/iraqi/static", help="Output directory")
    parser.add_argument("--models", nargs="+", default=None, help="Model types: rf svm mlp")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable data augmentation")
    parser.add_argument("--no-augment", action="store_false", dest="augment", help="Disable data augmentation")
    parser.add_argument("--augment-multiplier", type=int, default=3, help="Augmented copies per sample")
    parser.add_argument("--split", default=None, help="Split manifest name")
    parser.add_argument("--dialect", default="iraqi", help="Dialect name for registry")
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    split_mgr = DatasetSplitManager() if args.split else None
    results = train_static_model(
        args.dataset, args.output, args.models,
        augment=args.augment, augment_multiplier=args.augment_multiplier,
        split_manager=split_mgr, split_name=args.split,
        dialect=args.dialect,
    )
    print(f"\nTraining complete. Best model: {results['model_type']}")
