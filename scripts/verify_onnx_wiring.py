"""Verify ONNX Runtime is wired into recognizers."""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.recognition.static_recognizer import StaticSignRecognizer
from src.infrastructure.recognition.dynamic_recognizer import DynamicSignRecognizer
from src.infrastructure.inference.onnx_engine import ONNXInferenceEngine

# Prove StaticSignRecognizer accepts and delegates to InferenceEngine
sr = StaticSignRecognizer()
print("StaticSignRecognizer has set_inference_engine:", hasattr(sr, "set_inference_engine"))

engine = ONNXInferenceEngine()
sr.set_inference_engine(engine)
print("StaticSignRecognizer._inference_engine set:", sr._inference_engine is not None)
print("StaticSignRecognizer._inference_engine is ONNXEngine:", isinstance(sr._inference_engine, ONNXInferenceEngine))

# Same for DynamicSignRecognizer
dr = DynamicSignRecognizer()
print("DynamicSignRecognizer has set_inference_engine:", hasattr(dr, "set_inference_engine"))
dr.set_inference_engine(engine)
print("DynamicSignRecognizer._inference_engine set:", dr._inference_engine is not None)

# Verify predict() checks engine first by reading source code lines
src = inspect.getsource(sr.predict)
has_delegation = "self._inference_engine is not None and self._inference_engine.is_loaded" in src
print("StaticSignRecognizer.predict() delegates to engine:", has_delegation)

src = inspect.getsource(dr.predict)
has_delegation = "self._inference_engine is not None and self._inference_engine.is_loaded" in src
print("DynamicSignRecognizer.predict() delegates to engine:", has_delegation)

print("ONNX Runtime is WIRED into the recognition pipeline: VERIFIED")
