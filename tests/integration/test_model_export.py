"""Integration tests — model export utilities."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.infrastructure.model_exporter import ModelExporter


class TestModelExporter:

    @pytest.fixture
    def exporter(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield ModelExporter(output_dir=tmp)

    def test_export_sklearn_no_skl2onnx(self, exporter: ModelExporter):
        mock_model = Mock()
        result = exporter.export_sklearn(mock_model, "test_model", (1, 63))
        # Without skl2onnx, should return None without crashing
        assert result is None

    def test_export_pytorch_no_model(self, exporter: ModelExporter):
        result = exporter.export_pytorch(None, "test_model", None)
        assert result is None

    def test_quantize_no_onnxruntime(self, exporter: ModelExporter):
        result = exporter.quantize("nonexistent.onnx")
        assert result is None

    def test_output_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "exports"
            exporter = ModelExporter(output_dir=str(output))
            assert output.exists()
            assert output.is_dir()
