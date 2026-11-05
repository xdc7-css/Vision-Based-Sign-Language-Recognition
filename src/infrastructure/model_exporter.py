"""ModelExporter — export trained models to ONNX for production inference.

Supports:
    - scikit-learn models → ONNX (via skl2onnx)
    - PyTorch models → ONNX (via torch.onnx.export)
    - ONNX Runtime optimization (FP16, int8 quantization)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional


class ModelExporter:
    """Export trained models to ONNX format for production inference."""

    def __init__(self, output_dir: str = "data/models/exported") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

    def export_sklearn(self, model: Any, model_name: str, input_shape: tuple[int, ...]) -> Optional[str]:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            initial_type = [("float_input", FloatTensorType([None, *input_shape[1:]]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)

            path = self._output_dir / f"{model_name}.onnx"
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            self._logger.info(f"Exported sklearn model to {path}")
            return str(path)
        except ImportError:
            self._logger.warning("skl2onnx not installed, skipping ONNX export")
            return None
        except Exception as e:
            self._logger.error(f"Failed to export sklearn model: {e}")
            return None

    def export_pytorch(self, model: Any, model_name: str, dummy_input: Any) -> Optional[str]:
        try:
            import torch

            model.eval()
            path = self._output_dir / f"{model_name}.onnx"

            torch.onnx.export(
                model,
                dummy_input,
                str(path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            self._logger.info(f"Exported PyTorch model to {path}")
            return str(path)
        except Exception as e:
            self._logger.error(f"Failed to export PyTorch model: {e}")
            return None

    def quantize(self, onnx_path: str, output_path: Optional[str] = None) -> Optional[str]:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            out = output_path or onnx_path.replace(".onnx", "_quantized.onnx")
            quantize_dynamic(onnx_path, out, weight_type=QuantType.QInt8)
            self._logger.info(f"Quantized ONNX model: {onnx_path} → {out}")
            return out
        except ImportError:
            self._logger.warning("onnxruntime not installed, skipping quantization")
            return None
        except Exception as e:
            self._logger.error(f"Failed to quantize model: {e}")
            return None

    @staticmethod
    def optimize_for_cpu(onnx_path: str, output_path: Optional[str] = None) -> Optional[str]:
        try:
            import onnxruntime as ort

            out = output_path or onnx_path.replace(".onnx", "_optimized.onnx")
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.optimized_model_filepath = out
            _ = ort.InferenceSession(onnx_path, sess_opts)
            return out
        except Exception:
            return None

    @staticmethod
    def optimize_for_gpu(onnx_path: str, output_path: Optional[str] = None) -> Optional[str]:
        try:
            import onnxruntime as ort

            out = output_path or onnx_path.replace(".onnx", "_gpu_optimized.onnx")
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.optimized_model_filepath = out
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            _ = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
            return out
        except Exception:
            return None

    @staticmethod
    def validate_optimization(original_onnx: str, optimized_onnx: str, test_input: Any) -> bool:
        try:
            import numpy as np
            import onnxruntime as ort

            sess_orig = ort.InferenceSession(original_onnx)
            sess_opt = ort.InferenceSession(optimized_onnx)

            input_name = sess_orig.get_inputs()[0].name
            output_name = sess_orig.get_outputs()[0].name

            out_orig = sess_orig.run([output_name], {input_name: test_input})[0]
            out_opt = sess_opt.run([output_name], {input_name: test_input})[0]

            diff = np.max(np.abs(out_orig - out_opt))
            return diff < 1e-3
        except Exception:
            return False
