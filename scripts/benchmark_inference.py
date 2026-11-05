"""Inference latency benchmark — measures mean/median/p90/p99 latency for any model.

Usage:
    python scripts/benchmark_inference.py --model model.onnx --iterations 1000
    python scripts/benchmark_inference.py --model model.pkl --input-shape 63 --iterations 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def benchmark_model(
    model_path: str,
    input_shape: tuple[int, ...] = (63,),
    iterations: int = 500,
    warmup: int = 50,
    batch_size: int = 1,
    output: Optional[str] = None,
) -> dict:
    """Benchmark inference latency for a trained model.

    Args:
        model_path: Path to model file (.pkl, .pt, .onnx).
        input_shape: Expected input feature shape.
        iterations: Number of inference iterations.
        warmup: Number of warmup iterations.
        batch_size: Batch size for batched inference.
        output: Optional output JSON path.

    Returns:
        Dict with latency statistics.
    """
    model_path = Path(model_path)
    print(f"Benchmarking model: {model_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Iterations: {iterations}")
    print(f"  Warmup: {warmup}")
    print(f"  Batch size: {batch_size}")

    # Load model
    suffix = model_path.suffix.lower()

    if suffix == ".onnx":
        from src.infrastructure.inference.onnx_engine import ONNXInferenceEngine
        engine = ONNXInferenceEngine()
        assert engine.load(str(model_path)), "Failed to load ONNX model"
        provider = engine.metadata.get("provider", "cpu")
        print(f"  Provider: {provider}")

        dummy_input = np.random.randn(batch_size, *input_shape).astype(np.float32)

        def infer():
            engine.predict_batch(dummy_input)

    elif suffix == ".pkl":
        import pickle
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        model = data if not isinstance(data, dict) else data.get("model", data)
        dummy_input = np.random.randn(batch_size, *input_shape).astype(np.float32)

        def infer():
            model.predict(dummy_input)

    elif suffix == ".pt":
        from src.infrastructure.recognition.dynamic_recognizer import DynamicSignRecognizer
        recognizer = DynamicSignRecognizer()
        recognizer.load(str(model_path))
        recognizer._model.eval()

        import torch
        seq_len = 30
        dummy_input = torch.FloatTensor(np.random.randn(batch_size, seq_len, 63))

        def infer():
            with torch.no_grad():
                recognizer._model(dummy_input)
    else:
        raise ValueError(f"Unsupported model format: {suffix}")

    # Warmup
    for _ in range(warmup):
        infer()

    # Benchmark
    latencies: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        infer()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies_np = np.array(latencies)
    results = {
        "model_path": str(model_path),
        "input_shape": list(input_shape),
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup": warmup,
        "mean_ms": float(np.mean(latencies_np)),
        "median_ms": float(np.median(latencies_np)),
        "std_ms": float(np.std(latencies_np)),
        "p90_ms": float(np.percentile(latencies_np, 90)),
        "p95_ms": float(np.percentile(latencies_np, 95)),
        "p99_ms": float(np.percentile(latencies_np, 99)),
        "min_ms": float(np.min(latencies_np)),
        "max_ms": float(np.max(latencies_np)),
        "fps": 1000.0 / float(np.mean(latencies_np)) if latencies_np.mean() > 0 else 0,
    }

    print(f"\nResults:")
    print(f"  Mean:    {results['mean_ms']:.3f}ms")
    print(f"  Median:  {results['median_ms']:.3f}ms")
    print(f"  P90:     {results['p90_ms']:.3f}ms")
    print(f"  P99:     {results['p99_ms']:.3f}ms")
    print(f"  FPS:     {results['fps']:.1f}")
    print(f"  Std:     {results['std_ms']:.3f}ms")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model inference latency")
    parser.add_argument("--model", required=True, help="Model path (.pkl, .pt, .onnx)")
    parser.add_argument("--input-shape", type=int, nargs="+", default=[63], help="Input feature shape")
    parser.add_argument("--iterations", type=int, default=500, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    benchmark_model(
        args.model, tuple(args.input_shape),
        args.iterations, args.warmup, args.batch_size, args.output,
    )
