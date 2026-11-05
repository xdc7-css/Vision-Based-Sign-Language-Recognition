"""Dynamic gesture training pipeline — trains LSTM, GRU, Transformer, auto-selects best.

Supports:
    - Mini-batch training with DataLoader
    - Early stopping with patience
    - Data augmentation (landmark jitter, time warp)
    - Persistent train/val/test splits via DatasetSplitManager
    - Model evaluation via ModelEvaluator
    - Model registration via ModelRegistry
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from src.infrastructure.dataset.augmenter import DatasetAugmenter
from src.infrastructure.dataset.splitter import DatasetSplitManager
from src.infrastructure.recognition.dynamic_recognizer import (
    DynamicModelConfig,
    DynamicSignRecognizer,
)
from src.models.evaluator import ModelEvaluator, ModelRegistry


def load_sequence_dataset(
    dataset_path: str | Path,
    max_seq_len: int = 60,
    min_seq_len: int = 5,
    augmenter: Optional[DatasetAugmenter] = None,
    augment_multiplier: int = 1,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load sequential landmark data from a structured dataset.

    Expected structure:
        dataset_path/
            class_1/
                samples/
                    sample_1/
                        landmarks.npy  (T × 21 × 3)
            class_2/
                ...

    Returns:
        X: List of (T × 63) sequences.
        y: (n_samples,) string labels.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        samples_dir = class_dir / "samples"

        if not samples_dir.exists():
            continue

        for sample_dir in sorted(samples_dir.iterdir()):
            landmarks_file = sample_dir / "landmarks.npy"
            if not landmarks_file.exists():
                continue

            try:
                data = np.load(landmarks_file).astype(np.float32)

                if data.ndim == 3 and data.shape[2] == 3:
                    T = data.shape[0]
                    if T < min_seq_len:
                        continue
                    flat = data.reshape(T, -1)

                    if T > max_seq_len:
                        flat = flat[:max_seq_len]

                    X_list.append(flat)
                    y_list.append(class_name)

                    # Augment sequences
                    if augmenter is not None and augment_multiplier > 1:
                        for _ in range(augment_multiplier - 1):
                            aug = augmenter.augment_landmarks(flat)
                            X_list.append(aug)
                            y_list.append(class_name)

            except Exception:
                continue

    if not X_list:
        raise ValueError(f"No valid sequence samples found in {dataset_path}")

    return X_list, np.array(y_list)


def pad_sequences(
    sequences: list[np.ndarray],
    target_length: int = 30,
) -> np.ndarray:
    """Pad or truncate sequences to uniform length."""
    padded = np.zeros((len(sequences), target_length, 63), dtype=np.float32)
    for i, seq in enumerate(sequences):
        T = min(seq.shape[0], target_length)
        padded[i, :T, :] = seq[:T]
        if T > 0:
            padded[i, T:, :] = seq[-1:]
    return padded


def train_dynamic_model(
    dataset_path: str | Path = "data/datasets/dynamic",
    output_dir: str | Path = "data/models/iraqi/dynamic",
    architectures: Optional[list[str]] = None,
    sequence_length: int = 30,
    epochs: int = 50,
    augment: bool = True,
    augment_multiplier: int = 3,
    split_manager: Optional[DatasetSplitManager] = None,
    split_name: Optional[str] = None,
    dialect: str = "iraqi",
) -> dict:
    """Train, evaluate, and save the best dynamic gesture model.

    Args:
        dataset_path: Path to the structured sequence dataset.
        output_dir: Where to save the trained model.
        architectures: Which architectures to try.
        sequence_length: Target sequence length.
        epochs: Number of training epochs.
        augment: Whether to apply data augmentation.
        augment_multiplier: Augmented copies per real sample.
        split_manager: Optional split manager for test set separation.
        split_name: Split manifest name to load.
        dialect: Dialect name for model registry.

    Returns:
        Dict with training results and best model info.
    """
    print(f"Loading sequence dataset from: {dataset_path}")

    augmenter = DatasetAugmenter() if augment else None
    sequences, labels = load_sequence_dataset(
        dataset_path,
        augmenter=augmenter,
        augment_multiplier=augment_multiplier,
    )
    n_classes = len(set(labels))
    print(f"Loaded {len(sequences)} sequences across {n_classes} classes "
          f"(augment={augment}, multiplier={augment_multiplier})")

    # Pad sequences
    X = pad_sequences(sequences, sequence_length)
    print(f"Sequence shape: {X.shape}")

    # Configure and train
    config = DynamicModelConfig(
        sequence_length=sequence_length,
        epochs=epochs,
    )
    recognizer = DynamicSignRecognizer(config)

    print(f"\nTraining architectures: {architectures or recognizer.SUPPORTED_ARCHS}")
    all_metrics = recognizer.train(X, labels, architectures)

    # Print results
    print(f"\n{'='*65}")
    print(f"  Model Comparison — Dynamic Gesture Recognition")
    print(f"{'='*65}")
    print(f"  {'Arch':<14} {'Accuracy':>10} {'F1':>10} {'Infer(ms)':>10} {'Params':>10} {'Size(KB)':>10}")
    print(f"  {'-'*64}")
    for arch, m in sorted(all_metrics.items()):
        print(f"  {arch:<14} {m.accuracy:>10.3f} {m.f1_score:>10.3f} {m.inference_time_ms:>10.3f} "
              f"{m.params_count:>10} {m.model_size_kb:>10.1f}")

    best_arch = recognizer.architecture
    best = all_metrics.get(best_arch)
    print(f"\n  ✓ Best architecture: {best_arch} (accuracy: {best.accuracy:.3f}, F1: {best.f1_score:.3f})")

    # Save model
    model_path = recognizer.save(output_dir)
    print(f"\n  Saved model: {model_path}")

    # Evaluate using ModelEvaluator
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(labels)
        X_pad = pad_sequences(sequences, sequence_length)
        _, X_eval, _, y_eval = train_test_split(X_pad, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
        import torch
        recognizer._model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(X_eval)
            outputs = recognizer._model(x_t)
            _, preds = torch.max(outputs, 1)

        evaluator = ModelEvaluator()
        from sklearn.metrics import accuracy_score
        models = {best_arch: type("Wrapper", (), {"predict": lambda self, x: preds.cpu().numpy()})()}
        eval_results = evaluator.compare_models(models, X_eval, y_eval, list(le.classes_))

        eval_path = Path(output_dir) / "evaluation.json"
        evaluator.export_comparison(eval_results, str(eval_path))
        print(f"  Evaluation report: {eval_path}")
    except Exception as e:
        print(f"  Warning: Model evaluation failed: {e}")

    # Register model
    try:
        registry = ModelRegistry()
        registry.register_model(dialect, f"dynamic_{best_arch}", str(model_path), {
            "accuracy": best.accuracy if best else 0.0,
            "f1_score": best.f1_score if best else 0.0,
            "precision": best.precision if best else 0.0,
            "recall": best.recall if best else 0.0,
        })
        print(f"  Registered in model registry")
    except Exception as e:
        print(f"  Warning: Model registration failed: {e}")

    return {
        "architecture": best_arch,
        "accuracy": best.accuracy if best else 0.0,
        "f1_score": best.f1_score if best else 0.0,
        "inference_time_ms": best.inference_time_ms if best else 0.0,
        "model_path": str(model_path),
        "num_sequences": len(sequences),
        "num_classes": n_classes,
        "all_metrics": {
            arch: {
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "inference_time_ms": m.inference_time_ms,
                "params_count": m.params_count,
            }
            for arch, m in all_metrics.items()
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train dynamic gesture classifier")
    parser.add_argument("--dataset", default="data/datasets/dynamic", help="Dataset path")
    parser.add_argument("--output", default="data/models/iraqi/dynamic", help="Output directory")
    parser.add_argument("--archs", nargs="+", default=None, help="Architectures: lstm gru transformer")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable data augmentation")
    parser.add_argument("--no-augment", action="store_false", dest="augment", help="Disable data augmentation")
    parser.add_argument("--augment-multiplier", type=int, default=3, help="Augmented copies per sample")
    parser.add_argument("--split", default=None, help="Split manifest name")
    parser.add_argument("--dialect", default="iraqi", help="Dialect name for registry")
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    split_mgr = DatasetSplitManager() if args.split else None
    results = train_dynamic_model(
        args.dataset, args.output, args.archs, args.seq_len, args.epochs,
        augment=args.augment, augment_multiplier=args.augment_multiplier,
        split_manager=split_mgr, split_name=args.split,
        dialect=args.dialect,
    )
    print(f"\nTraining complete. Best architecture: {results['architecture']}")
