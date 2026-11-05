"""Standalone model evaluation entry point.

Usage:
    python -m src.models.evaluate --model model.pkl --test test_data.npy --test-labels test_labels.npy

Loads a model and test set, runs predictions, generates
classification report, confusion matrix, and latency report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.models.evaluator import ModelEvaluator


def evaluate_model(
    model_path: str | Path,
    test_data_path: str | Path,
    test_labels_path: Optional[str | Path] = None,
    output_dir: str | Path = "data/evaluations",
    model_name: Optional[str] = None,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Evaluate a trained model and generate full report.

    Auto-detects model format (.pkl / .pt / .onnx).

    Args:
        model_path: Path to model file.
        test_data_path: Path to test data (.npy, or directory).
        test_labels_path: Path to test labels (.npy, or None if data has labels).
        output_dir: Where to save evaluation artifacts.
        model_name: Name for the report files.
        class_names: Optional class name mapping.

    Returns:
        Dict with evaluation results.
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_name or model_path.stem

    print(f"Loading model: {model_path}")

    # Load model
    model, loaded_model_name = _load_model(model_path)

    # Load test data
    print(f"Loading test data: {test_data_path}")
    test_path = Path(test_data_path)
    if test_path.is_dir():
        X_test, y_test, names = _load_from_directory(test_path)
    else:
        X_test = np.load(test_path)
        if test_labels_path:
            y_test = np.load(str(test_labels_path))
        else:
            raise ValueError("test_labels_path must be provided for .npy test data")
        names = class_names or [str(i) for i in range(max(y_test) + 1)]

    print(f"Test set: {len(X_test)} samples, {len(names)} classes")

    # Inference latency measurement
    latencies: list[float] = []
    predictions: list[int] = []
    for i in range(len(X_test)):
        start = time.perf_counter()
        pred = int(model.predict(X_test[i:i+1])[0])
        latencies.append((time.perf_counter() - start) * 1000)
        predictions.append(pred)

    y_pred = np.array(predictions)
    latency_stats = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "samples": len(latencies),
    }
    print(f"Latency: {latency_stats['mean_ms']:.3f}ms ± {latency_stats['p90_ms']:.3f}ms (p90)")

    # Generate evaluation report
    evaluator = ModelEvaluator()
    evaluator.generate_classification_report(y_test, y_pred, names, output_dir, model_name)
    cm = evaluator.confusion_matrix(y_test, y_pred, names)

    # Build evaluation result
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    result = {
        "model_name": model_name,
        "model_path": str(model_path),
        "num_test_samples": len(X_test),
        "num_classes": len(names),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": cm,
        "class_names": names,
        "latency": latency_stats,
    }

    # Save overall result
    result_path = output_dir / f"{model_name}_evaluation.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\nResults:")
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1:        {result['f1_score']:.4f}")
    print(f"  Latency:   {latency_stats['mean_ms']:.3f}ms (p90: {latency_stats['p90_ms']:.3f}ms)")
    print(f"\nReports saved to: {output_dir}/")

    return result


def _load_model(path: Path) -> tuple[Any, str]:
    """Load model from .pkl, .pt, or .onnx."""
    suffix = path.suffix.lower()

    if suffix == ".pkl":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            model = data.get("model", data)
        else:
            model = data
        return model, "sklearn"

    elif suffix == ".pt":
        from src.infrastructure.recognition.dynamic_recognizer import DynamicSignRecognizer
        recognizer = DynamicSignRecognizer()
        recognizer.load(str(path))

        class Wrapper:
            def predict(self, x):
                import torch
                x_t = torch.FloatTensor(x)
                recognizer._model.eval()
                with torch.no_grad():
                    logits = recognizer._model(x_t)
                    _, preds = torch.max(logits, 1)
                return preds.cpu().numpy()
        return Wrapper(), "pytorch"

    elif suffix == ".onnx":
        from src.infrastructure.inference.onnx_engine import ONNXInferenceEngine
        engine = ONNXInferenceEngine()
        engine.load(str(path))

        class Wrapper:
            def predict(self, x):
                results = engine.predict_batch(x)
                return np.array([r.label_id for r in results])
        return Wrapper(), "onnx"

    else:
        raise ValueError(f"Unsupported model format: {suffix}")


def _load_from_directory(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load test data from structured directory (class/samples/landmarks.npy)."""
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    names: list[str] = []

    for class_dir in sorted(path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        names.append(class_name)
        label = len(names) - 1
        samples_dir = class_dir / "samples"
        if not samples_dir.exists():
            continue
        for sample_dir in sorted(samples_dir.iterdir()):
            lm_file = sample_dir / "landmarks.npy"
            if not lm_file.exists():
                continue
            try:
                data = np.load(lm_file)
                if data.ndim == 3:
                    flat = data[data.shape[0] // 2].flatten()
                elif data.ndim == 2:
                    flat = data.flatten()
                else:
                    continue
                if flat.shape[0] == 63:
                    X_list.append(flat)
                    y_list.append(label)
            except Exception:
                continue

    return np.array(X_list), np.array(y_list), names


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", required=True, help="Model path (.pkl, .pt, .onnx)")
    parser.add_argument("--test-data", required=True, help="Test data path (.npy or directory)")
    parser.add_argument("--test-labels", help="Test labels path (for .npy data)")
    parser.add_argument("--output", default="data/evaluations", help="Output directory")
    parser.add_argument("--name", default=None, help="Model name for reports")
    parser.add_argument("--classes", nargs="*", default=None, help="Class names")
    args = parser.parse_args()

    evaluate_model(
        args.model, args.test_data, args.test_labels,
        args.output, args.name, args.classes,
    )
