"""Integration tests — configuration system."""

import os
import tempfile
from pathlib import Path

import pytest

from src.infrastructure.config_manager import ConfigManager


class TestConfigManagerIntegration:

    @pytest.fixture
    def temp_config_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_default_config_loading(self):
        cm = ConfigManager(config_dir="config")
        config = cm.load()
        assert isinstance(config, dict)
        assert "app" in config
        assert "dialect" in config
        assert "recognition" in config
        assert "performance" in config
        assert "overlay" in config

    def test_default_values_present(self):
        cm = ConfigManager(config_dir="config")
        config = cm.load()
        assert cm.get("app.name") == "ArSL Translation Overlay"
        assert cm.get("dialect.active") == "iraqi"
        assert cm.get("performance.profile") == "balanced"

    def test_get_with_default(self):
        cm = ConfigManager(config_dir="config")
        config = cm.load()
        assert cm.get("nonexistent.key", "fallback") == "fallback"

    def test_set_and_get(self):
        cm = ConfigManager(config_dir="config")
        cm.load()
        cm.set("test.key", "value")
        assert cm.get("test.key") == "value"

    def test_nested_key_access(self):
        cm = ConfigManager(config_dir="config")
        cm.load()
        threshold = cm.get("recognition.hybrid.word_threshold")
        assert threshold == 0.85

    def test_env_override(self):
        os.environ["PERFORMANCE_PROFILE"] = "max_performance"
        cm = ConfigManager(config_dir="config")
        config = cm.load()
        assert cm.get("performance.profile") == "max_performance"
        del os.environ["PERFORMANCE_PROFILE"]

    def test_env_bool_override(self):
        os.environ["OVERLAY_ENABLED"] = "false"
        cm = ConfigManager(config_dir="config")
        config = cm.load()
        assert cm.get("overlay.enabled") is False
        del os.environ["OVERLAY_ENABLED"]

    def test_env_int_override(self):
        os.environ["OVERLAY_FONT_SIZE"] = "48"
        cm = ConfigManager(config_dir="config")
        config = cm.load()
        assert cm.get("overlay.font_size") == 48
        del os.environ["OVERLAY_FONT_SIZE"]
