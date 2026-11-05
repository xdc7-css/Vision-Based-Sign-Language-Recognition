"""Tests for DialectManager — runtime dialect switching, model loading, shared pipeline."""

import pytest
from unittest.mock import Mock, patch, PropertyMock

from src.domain.dialect import DialectId, DialectConfig, DialectRegistry
from src.infrastructure.dialect_manager import DialectManager, DEFAULT_DIALECTS


class TestDialectRegistry:

    def test_register_and_activate(self):
        reg = DialectRegistry()
        cfg = DialectConfig(dialect_id=DialectId.IRAQI)
        reg.register(cfg)

        assert reg.active_id == DialectId.IRAQI

        c2 = DialectConfig(dialect_id=DialectId.SAUDI)
        reg.register(c2)
        assert reg.activate(DialectId.SAUDI)
        assert reg.active_id == DialectId.SAUDI

    def test_activate_disabled_dialect_fails(self):
        reg = DialectRegistry()
        cfg = DialectConfig(dialect_id=DialectId.IRAQI, enabled=False)
        reg.register(cfg)
        assert not reg.activate(DialectId.IRAQI)

    def test_activate_missing_dialect_fails(self):
        reg = DialectRegistry()
        assert not reg.activate(DialectId.IRAQI)

    def test_list_enabled(self):
        reg = DialectRegistry()
        reg.register(DialectConfig(dialect_id=DialectId.IRAQI, enabled=True))
        reg.register(DialectConfig(dialect_id=DialectId.SAUDI, enabled=False))
        reg.register(DialectConfig(dialect_id=DialectId.EGYPTIAN, enabled=True))

        enabled = reg.list_enabled()
        assert len(enabled) == 2
        assert DialectId.IRAQI in [d.dialect_id for d in enabled]
        assert DialectId.EGYPTIAN in [d.dialect_id for d in enabled]

    def test_active_config(self):
        reg = DialectRegistry()
        cfg = DialectConfig(dialect_id=DialectId.IRAQI)
        reg.register(cfg)
        assert reg.active == cfg

    def test_display_name_default(self):
        cfg = DialectConfig(dialect_id=DialectId.IRAQI)
        assert "عراقي" in cfg.display_name


class TestDialectManager:

    def test_default_registration(self):
        dm = DialectManager(base_path=".")
        assert dm.active_dialect == DialectId.IRAQI
        assert len(dm.available_dialects) == 3

    def test_switch_dialect(self):
        dm = DialectManager(base_path=".")
        result = dm.switch_dialect(DialectId.SAUDI)
        # Should succeed even without models (will log warning but return True)
        assert result is True
        assert dm.active_dialect == DialectId.SAUDI

    def test_switch_to_unknown_dialect(self):
        dm = DialectManager(base_path=".")
        result = dm.switch_dialect(DialectId.GENERAL)
        # GENERAL is registered but has no model dir
        assert dm.active_dialect == DialectId.IRAQI  # should revert

    def test_available_dialects_contains_all_defaults(self):
        dm = DialectManager(base_path=".")
        ids = [d.dialect_id for d in dm.available_dialects]
        assert DialectId.IRAQI in ids
        assert DialectId.SAUDI in ids
        assert DialectId.EGYPTIAN in ids

    def test_discover_installed_dialects(self, tmp_path):
        dm = DialectManager(base_path=str(tmp_path))

        # Create some dialect model directories
        models_dir = tmp_path / "data" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "iraqi").mkdir()
        (models_dir / "saudi").mkdir()

        # Override base_path to point to tmp_path
        dm._base_path = tmp_path
        discovered = dm.discover_installed_dialects()
        assert DialectId.IRAQI in discovered
        assert DialectId.SAUDI in discovered
        assert DialectId.EGYPTIAN not in discovered  # not created

    def test_register_hook(self):
        dm = DialectManager(base_path=".")
        hook = lambda x: x
        dm.register_hook(DialectId.IRAQI, "preprocess", hook)
        assert DialectId.IRAQI in dm._preprocess_hooks

    def test_pipeline_setter(self):
        dm = DialectManager(base_path=".")
        mock_pipeline = Mock()
        dm.pipeline = mock_pipeline
        assert dm.pipeline == mock_pipeline

    def test_active_dialect_config_property(self):
        dm = DialectManager(base_path=".")
        cfg = dm.active_dialect_config
        assert cfg is not None
        assert cfg.dialect_id == DialectId.IRAQI

    def test_registry_property(self):
        dm = DialectManager(base_path=".")
        reg = dm.registry
        assert isinstance(reg, DialectRegistry)
        assert reg.active_id == DialectId.IRAQI
