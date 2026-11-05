"""Dialect domain models — dialect registry and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class DialectId(Enum):
    IRAQI = "iraqi"
    SAUDI = "saudi"
    EGYPTIAN = "egyptian"
    GENERAL = "general"


# Human-readable display names
DIALECT_DISPLAY_NAMES: dict[DialectId, str] = {
    DialectId.IRAQI: "عراقي (Iraqi)",
    DialectId.SAUDI: "سعودي (Saudi)",
    DialectId.EGYPTIAN: "مصري (Egyptian)",
    DialectId.GENERAL: "عام (General)",
}


@dataclass
class DialectConfig:
    dialect_id: DialectId
    display_name: str = ""
    dataset_path: str = ""
    model_path: str = ""
    vocabulary_path: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = DIALECT_DISPLAY_NAMES.get(self.dialect_id, self.dialect_id.value)


@dataclass
class DialectRegistry:
    """Holds all available dialect configurations with runtime switching support."""
    dialects: dict[DialectId, DialectConfig] = field(default_factory=dict)
    _active_id: DialectId = DialectId.IRAQI

    @property
    def active_id(self) -> DialectId:
        return self._active_id

    @property
    def active(self) -> Optional[DialectConfig]:
        return self.dialects.get(self._active_id)

    def activate(self, dialect_id: DialectId) -> bool:
        if dialect_id in self.dialects and self.dialects[dialect_id].enabled:
            self._active_id = dialect_id
            return True
        return False

    def register(self, config: DialectConfig) -> None:
        self.dialects[config.dialect_id] = config

    def list_enabled(self) -> list[DialectConfig]:
        return [d for d in self.dialects.values() if d.enabled]
