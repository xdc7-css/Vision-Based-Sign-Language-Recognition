from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from src.domain.interfaces import ConfigRepository


DEFAULT_CONFIG: dict[str, Any] = {
    "pipeline": {
        "max_fps": 30,
        "frame_width": 640,
        "frame_height": 480,
    },
    "recognition": {
        "dialect": "iraqi",
        "confidence_threshold": {
            "static": 0.85,
            "dynamic": 0.75,
            "alphabet": 0.90,
        },
    },
    "overlay": {
        "enabled": True,
        "opacity": 0.85,
        "position": "bottom",
        "font_family": "Arial",
        "font_size": 36,
        "font_color": "#FFFFFF",
        "background_color": "#000000",
        "max_lines": 3,
    },
    "ui": {
        "language": "ar",
        "minimize_to_tray": True,
    },
    "logging": {
        "level": "info",
    },
}


class JsonConfigRepository(ConfigRepository):
    """Configuration persistence using JSON files."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path:
            self._config_path = Path(config_path)
        else:
            app_data = os.environ.get("APPDATA", str(Path.home() / ".config"))
            self._config_path = Path(app_data) / "ArSL-Overlay" / "config.json"

        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config: dict[str, Any] = {}
        self._load_or_create()

    def _load_or_create(self) -> None:
        if self._config_path.exists():
            try:
                with open(self._config_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                self._config = self._merge(DEFAULT_CONFIG, loaded)
            except (json.JSONDecodeError, OSError):
                self._config = dict(DEFAULT_CONFIG)
        else:
            self._config = dict(DEFAULT_CONFIG)
            self.save(self._config)

    @staticmethod
    def _merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = JsonConfigRepository._merge(result[key], value)
            else:
                result[key] = value
        return result

    def load(self) -> dict:
        return dict(self._config)

    def save(self, config: dict) -> None:
        self._config = dict(config)
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except OSError as e:
            msg = f"Failed to save config: {e}"
            raise RuntimeError(msg) from e

    def get(self, key: str, default=None):
        parts = key.split(".")
        value = self._config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self, key: str, value) -> None:
        parts = key.split(".")
        target = self._config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
        self.save(self._config)
