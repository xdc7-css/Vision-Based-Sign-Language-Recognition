"""Comprehensive automated test suite for Gesture Classes Management system.

Covers:
1. Add a class (Static, Dynamic, Both, Alphabet).
2. Rename an empty class.
3. Rename a class with 100+ saved samples (transactional folder & metadata update).
4. Delete an empty default class (no resurrection on restart).
5. Archive a class containing samples (samples preserved 100%).
6. Permanently delete sample data only upon explicit confirmation.
7. Restart and verify changes persist across registry reloads.
8. Prevent duplicate class names (exact, spacing, canonical).
9. Handle Arabic Unicode class names with diacritics and hamza.
10. Preserve stable class IDs.
11. Verify training dataset consistency (respect active vs archived classes).
12. Verify saved model mapping protection (checkpoints preserved until retrained).
13. Verify search, filter, and counters refresh.
14. Real GUI integration test on DatasetCollectionTab.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.application.state import AppState
from src.application.use_cases import VideoSourceUseCases
from src.infrastructure.dataset.gesture_registry import (
    GestureRegistry,
    sanitize_folder_name,
)
from src.models.train_static import inspect_dataset, train_static_model
from src.presentation.tabs.dataset import DatasetCollectionTab


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def temp_env(tmp_path: Path):
    """Provides isolated registry and dataset directories for testing."""
    reg_file = tmp_path / "vocabulary" / "gesture_classes.json"
    data_dir = tmp_path / "datasets"
    (data_dir / "words").mkdir(parents=True)
    (data_dir / "dynamic").mkdir(parents=True)
    (data_dir / "alphabet").mkdir(parents=True)
    return reg_file, data_dir


# ── 1. Create / Add Class Tests ──────────────────────────────────────────


def test_add_class(temp_env):
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    # Add Static class
    cls1 = registry.add_class("طائرة", category="static", description="Plane gesture")
    assert cls1.name == "طائرة"
    assert cls1.category == "static"
    assert cls1.is_active
    assert not cls1.is_default
    assert (data_dir / "words" / "طائرة" / "samples").exists()

    # Add Dynamic class
    cls2 = registry.add_class("إلى اللقاء", category="dynamic", description="See you later")
    assert cls2.category == "dynamic"
    assert (data_dir / "dynamic" / "إلى_اللقاء" / "samples").exists()

    # Add Both class
    cls3 = registry.add_class("قراءة", category="both")
    assert cls3.category == "both"
    assert (data_dir / "words" / "قراءة" / "samples").exists()
    assert (data_dir / "dynamic" / "قراءة" / "samples").exists()


def test_prevent_duplicate_class_names(temp_env):
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    registry.add_class("حاسوب", category="static")

    # Exact duplicate
    with pytest.raises(ValueError, match="already exists"):
        registry.add_class("حاسوب", category="static")

    # Spacing duplicate
    with pytest.raises(ValueError, match="already exists"):
        registry.add_class("  حاسوب  ", category="dynamic")

    # Underscore duplicate
    with pytest.raises(ValueError, match="already exists"):
        registry.add_class("حاسوب_", category="static")


def test_arabic_unicode_names_and_stable_ids(temp_env):
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    names = [
        "السلام عليكم ورحمة الله",
        "أهلاً وسهلاً",
        "حرف الألف (أ)",
        "يَوْمٌ سَعِيد",
    ]
    created = []
    for name in names:
        cls_obj = registry.add_class(name)
        assert cls_obj.id is not None and len(cls_obj.id) > 0
        created.append(cls_obj)

    # Check IDs are unique and stable
    ids = [c.id for c in created]
    assert len(ids) == len(set(ids))

    # Save and reload
    registry.save()
    reloaded = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)
    for orig in created:
        found = reloaded.get_class(orig.name)
        assert found is not None
        assert found.id == orig.id


# ── 2. Update / Rename Class Tests ───────────────────────────────────────


def test_rename_empty_class(temp_env):
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    cls_orig = registry.add_class("كلمة قديمة", category="static")
    orig_id = cls_orig.id

    updated = registry.rename_class("كلمة قديمة", "كلمة جديدة")
    assert updated.name == "كلمة جديدة"
    assert updated.id == orig_id  # Stable ID preserved

    assert registry.get_class("كلمة قديمة") is None
    assert registry.get_class("كلمة جديدة") is not None


def test_rename_class_with_many_samples(temp_env):
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    old_name = "تحية"
    new_name = "سلام"
    registry.add_class(old_name, category="static")

    # Create 105 real sample folders with landmarks.npy and metadata.json
    samples_dir = data_dir / "words" / old_name / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for i in range(105):
        s_dir = samples_dir / f"sample_{i:03d}"
        s_dir.mkdir(parents=True)
        lm = np.ones((1, 21, 3), dtype=np.float32) * (i * 0.001)
        np.save(s_dir / "landmarks.npy", lm)
        meta = {
            "sample_id": f"sample_{i:03d}",
            "gesture_label": old_name,
            "timestamp": 1730000000.0 + i,
            "num_frames": 1,
        }
        with open(s_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    assert registry.get_sample_counts(old_name)["total"] == 105

    # Execute rename
    renamed_cls = registry.rename_class(old_name, new_name)
    assert renamed_cls.name == new_name

    # Check old path is gone and new path has all 105 samples
    old_dir = data_dir / "words" / old_name
    new_dir = data_dir / "words" / new_name
    assert not old_dir.exists()
    assert new_dir.exists()

    new_samples = list((new_dir / "samples").iterdir())
    assert len(new_samples) == 105

    # Verify each sample's metadata was updated to new name
    for s in new_samples:
        with open(s / "metadata.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        assert m["gesture_label"] == new_name
        # Verify landmarks are intact
        arr = np.load(s / "landmarks.npy")
        assert arr.shape == (1, 21, 3)

    assert registry.get_sample_counts(new_name)["total"] == 105


def test_rename_destination_conflict_protection(temp_env):
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    registry.add_class("فئة_أ", category="static")
    registry.add_class("فئة_ب", category="static")

    # Trying to rename to existing active class
    with pytest.raises(ValueError, match="already exists"):
        registry.rename_class("فئة_أ", "فئة_ب")


# ── 3. Delete & Archive Tests ────────────────────────────────────────────


def test_delete_empty_default_class_no_resurrect(temp_env):
    reg_file, data_dir = temp_env
    # First init migrates DEFAULT_VOCABULARY
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)
    assert registry.get_class("مرحبا") is not None

    # Delete empty default class
    action, count = registry.delete_class("مرحبا")
    assert action == "deleted"
    assert count == 0
    assert registry.get_class("مرحبا") is None

    # Simulate application restart
    restarted = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)
    assert restarted.get_class("مرحبا") is None  # MUST NOT RESURRECT


def test_archive_class_containing_samples(temp_env):
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    name = "فئة تحوي بيانات"
    folder = sanitize_folder_name(name)
    registry.add_class(name, category="static")

    # Create 5 samples
    s_dir = data_dir / "words" / folder / "samples"
    for i in range(5):
        sample = s_dir / f"sample_{i}"
        sample.mkdir(parents=True)
        np.save(sample / "landmarks.npy", np.zeros((1, 21, 3), dtype=np.float32))

    assert registry.get_sample_counts(name)["total"] == 5

    # Safe delete (Archive)
    action, affected = registry.delete_class(name, permanent_delete_samples=False)
    assert action == "archived"
    assert affected == 5

    # Class is removed from active list
    assert registry.get_class(name) is None
    # Class is in archived list
    archived_names = [c.name for c in registry.list_archived_classes()]
    assert name in archived_names

    # CRITICAL: All 5 sample files are still 100% on disk!
    assert (data_dir / "words" / folder / "samples").exists()
    assert len(list((data_dir / "words" / folder / "samples").iterdir())) == 5

    # Verify unarchive works
    restored = registry.unarchive_class(name)
    assert restored is not None
    assert restored.is_active
    assert registry.get_class(name) is not None


def test_permanent_delete_samples_with_confirmation(temp_env):
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    name = "فئة_للحذف_النهائي"
    registry.add_class(name, category="static")

    # Create samples
    s_dir = data_dir / "words" / name / "samples"
    for i in range(3):
        sample = s_dir / f"sample_{i}"
        sample.mkdir(parents=True)
        np.save(sample / "landmarks.npy", np.zeros((1, 21, 3), dtype=np.float32))

    # Explicit permanent delete
    action, affected = registry.delete_class(name, permanent_delete_samples=True)
    assert action == "permanently_deleted"
    assert affected == 3

    # Folder must be removed
    assert not (data_dir / "words" / name).exists()
    assert registry.get_class(name) is None

    # Restart check
    reloaded = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)
    assert reloaded.get_class(name) is None


# ── 4. Training Consistency & Model Mapping Protection ───────────────────


def test_training_dataset_consistency_with_registry(temp_env, monkeypatch):
    reg_file, data_dir = temp_env
    from unittest.mock import MagicMock
    monkeypatch.setattr("src.models.train_static.ModelRegistry", MagicMock)
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    # Create 3 classes on disk with samples: 'إشارة_أ', 'إشارة_ب', 'إشارة_مؤرشفة'
    words_dir = data_dir / "words"
    for cls_name in ["إشارة_أ", "إشارة_ب", "إشارة_مؤرشفة"]:
        cls_samples = words_dir / cls_name / "samples"
        cls_samples.mkdir(parents=True)
        for i in range(3):
            s = cls_samples / f"s_{i}"
            s.mkdir()
            lm = np.zeros((1, 21, 3), dtype=np.float32)
            lm[0, :, 0] = 0.5 if cls_name == "إشارة_أ" else 0.8
            np.save(s / "landmarks.npy", lm)

    # In registry, 'إشارة_أ' and 'إشارة_ب' are active, 'إشارة_مؤرشفة' is archived
    registry.add_class("إشارة_أ", category="static")
    registry.add_class("إشارة_ب", category="static")
    registry.add_class("إشارة_مؤرشفة", category="static")
    registry.delete_class("إشارة_مؤرشفة", permanent_delete_samples=False)  # archive

    # Get allowed classes from active registry
    active_classes = {c.name.replace(" ", "_") for c in registry.list_active_classes()}
    assert "إشارة_مؤرشفة" not in active_classes
    assert "إشارة_أ" in active_classes
    assert "إشارة_ب" in active_classes

    # Inspect dataset with active classes restriction
    inspection = inspect_dataset(words_dir, allowed_classes=active_classes)
    assert "إشارة_مؤرشفة" not in inspection["accepted_classes"]
    assert "إشارة_أ" in inspection["accepted_classes"]
    assert "إشارة_ب" in inspection["accepted_classes"]

    # Train model with active classes restriction
    out_dir = data_dir.parent / "models"
    res = train_static_model(
        dataset_path=words_dir,
        output_dir=out_dir,
        model_types=["mlp"],
        epochs=3,
        val_split=0.33,
        augment=False,
        allowed_classes={"إشارة_أ", "إشارة_ب"},
    )
    assert sorted(res["classes"]) == ["إشارة_أ", "إشارة_ب"]
    assert "إشارة_مؤرشفة" not in res["classes"]


def test_saved_model_checkpoint_protection(temp_env):
    """Ensure existing checkpoints are never overwritten merely by modifying classes in registry."""
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    model_dir = data_dir.parent / "models" / "iraqi"
    model_dir.mkdir(parents=True)
    fake_checkpoint = model_dir / "static_model.pkl"
    fake_checkpoint.write_bytes(b"EXISTING_CHECKPOINT_DATA_12345")
    classes_json = model_dir / "classes.json"
    classes_json.write_text('["نعم", "لا"]', encoding="utf-8")

    # Add and rename classes in registry
    registry.add_class("جديد", category="static")
    registry.rename_class("جديد", "جديد_معدل")

    # Checkpoint and classes.json must be 100% unchanged
    assert fake_checkpoint.read_bytes() == b"EXISTING_CHECKPOINT_DATA_12345"
    assert classes_json.read_text(encoding="utf-8") == '["نعم", "لا"]'


# ── 5. Search, Filter, and Organization Tests ────────────────────────────


def test_search_and_filter_classes(temp_env):
    reg_file, data_dir = temp_env
    registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)

    # Empty registry for test clarity
    registry._classes.clear()
    registry.add_class("شكرا", category="static")
    registry.add_class("شكرا جزيلا", category="dynamic")
    registry.add_class("أكل", category="both")
    registry.add_class("حرف الباء", category="alphabet")

    # Add samples to 'شكرا' only
    s_dir = data_dir / "words" / "شكرا" / "samples" / "s1"
    s_dir.mkdir(parents=True)
    np.save(s_dir / "landmarks.npy", np.zeros((1, 21, 3), dtype=np.float32))

    # 1. Search by Arabic query
    res_search = registry.filter_classes(query="شكرا")
    assert len(res_search) == 2

    # 2. Filter by status 'with_samples'
    res_samples = registry.filter_classes(status_filter="with_samples")
    assert len(res_samples) == 1
    assert res_samples[0][0].name == "شكرا"

    # 3. Filter by status 'empty'
    res_empty = registry.filter_classes(status_filter="empty")
    assert len(res_empty) == 3

    # 4. Filter by category 'dynamic' (includes dynamic and both)
    res_dyn = registry.filter_classes(category_filter="dynamic")
    names_dyn = {item[0].name for item in res_dyn}
    assert "شكرا جزيلا" in names_dyn
    assert "أكل" in names_dyn
    assert "شكرا" not in names_dyn

    # 5. Filter by category 'alphabet'
    res_alpha = registry.filter_classes(category_filter="alphabet")
    assert len(res_alpha) == 1
    assert res_alpha[0][0].name == "حرف الباء"


# ── 6. UI Integration Tests on DatasetCollectionTab ──────────────────────


def test_dataset_tab_gesture_classes_crud_ui(qapp, temp_env, monkeypatch):
    reg_file, data_dir = temp_env
    state = AppState()
    use_cases = VideoSourceUseCases(state)

    tab = DatasetCollectionTab(state, use_cases)
    # Point tab's registry to test environment
    test_registry = GestureRegistry(registry_path=reg_file, dataset_dir=data_dir)
    test_registry._classes.clear()
    test_registry.add_class("مرحبا", category="static")
    test_registry.add_class("شكرا", category="dynamic")
    tab._registry = test_registry
    tab._scan_and_populate_classes()

    assert tab._class_list.count() == 2

    # Verify buttons are enabled when an item is selected
    tab._class_list.setCurrentRow(0)
    assert tab._edit_class_btn.isEnabled()
    assert tab._delete_class_btn.isEnabled()
    assert tab._inspect_btn.isEnabled()

    # Verify selected label extraction
    assert tab._get_selected_class_label() in ("مرحبا", "شكرا")

    # Test search filter in UI
    tab._search_edit.setText("شكرا")
    assert tab._class_list.count() == 1
    assert "شكرا" in tab._class_list.item(0).text()

    tab._search_edit.clear()
    assert tab._class_list.count() == 2

    # Test category filter in UI
    idx = tab._type_filter_combo.findData("dynamic")
    tab._type_filter_combo.setCurrentIndex(idx)
    assert tab._class_list.count() == 1
    assert "شكرا" in tab._class_list.item(0).text()

    tab.close()
