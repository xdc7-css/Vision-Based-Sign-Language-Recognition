"""ConfigManager — hierarchical YAML/JSON/environment configuration.

Read order:
    1. config/default.yaml   (shipped defaults)
    2. config/user.yaml       (user overrides, gitignored)
    3. Environment variables  (deployment overrides)

Values are merged recursively. Later sources override earlier ones.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


class ConfigManager:
    """Hierarchical configuration manager with env var overrides."""

    def __init__(self, config_dir: str = "config") -> None:
        self._config_dir = Path(config_dir)
        self._config: dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)

    def load(self) -> dict[str, Any]:
        if self._config:
            return self._config

        result: dict[str, Any] = {}

        # 1. Default config
        default_path = self._config_dir / "default.yaml"
        if default_path.exists():
            try:
                import yaml
                with open(default_path, encoding="utf-8") as f:
                    result = yaml.safe_load(f) or {}
                self._logger.info(f"Loaded default config from {default_path}")
            except ImportError:
                self._logger.warning("PyYAML not installed, using env-only config")
            except Exception as e:
                self._logger.error(f"Failed to load default config: {e}")

        # 2. User config overlay
        user_path = self._config_dir / "user.yaml"
        if user_path.exists():
            try:
                import yaml
                with open(user_path, encoding="utf-8") as f:
                    user_cfg = yaml.safe_load(f) or {}
                result = self._deep_merge(result, user_cfg)
                self._logger.info(f"Loaded user config from {user_path}")
            except Exception as e:
                self._logger.error(f"Failed to load user config: {e}")

        # 3. Environment variable overrides
        result = self._apply_env_overrides(result)

        self._config = result
        return result

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        target = self._config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def save_user_config(self) -> None:
        user_path = self._config_dir / "user.yaml"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
            with open(user_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self._logger.error(f"Failed to save user config: {e}")

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _apply_env_overrides(config: dict) -> dict:
        env_map = {
            "LOG_LEVEL": ("logging", "level"),
            "PERFORMANCE_PROFILE": ("performance", "profile"),
            "ACTIVE_DIALECT": ("dialect", "active"),
            "OVERLAY_ENABLED": ("overlay", "enabled"),
            "OVERLAY_X": ("overlay", "x"),
            "OVERLAY_Y": ("overlay", "y"),
            "OVERLAY_WIDTH": ("overlay", "width"),
            "OVERLAY_HEIGHT": ("overlay", "height"),
            "OVERLAY_FONT_SIZE": ("overlay", "font_size"),
            "WORD_THRESHOLD": ("recognition", "hybrid", "word_threshold"),
            "DYNAMIC_THRESHOLD": ("recognition", "hybrid", "dynamic_threshold"),
            "ALPHABET_THRESHOLD": ("recognition", "hybrid", "alphabet_threshold"),
        }

        for env_var, keys in env_map.items():
            value = os.environ.get(env_var)
            if value is not None:
                target = config
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]

                # Type conversion
                typed_value: Any = value
                if value.lower() in ("true", "false"):
                    typed_value = value.lower() == "true"
                else:
                    try:
                        if "." in value:
                            typed_value = float(value)
                        else:
                            typed_value = int(value)
                    except ValueError:
                        pass

                target[keys[-1]] = typed_value

        return config
