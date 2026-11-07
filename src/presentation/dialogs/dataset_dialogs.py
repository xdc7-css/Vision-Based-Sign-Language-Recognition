"""Dialogs for Dataset Studio: Add class, Edit class, Delete/Archive class, and Inspect samples."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _canonical_key(text: str) -> str:
    """Normalize text for duplicate comparison: lowercase, whitespace and underscores collapsed."""
    clean = re.sub(r"[\s_]+", " ", text.strip()).lower()
    return clean.replace(" ", "_")


class AddClassDialog(QDialog):
    """Dialog to create a new gesture class with category and validation."""

    def __init__(
        self,
        existing_names: Optional[set[str] | QWidget] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        if isinstance(existing_names, QWidget) and parent is None:
            parent = existing_names
            existing_names = None

        super().__init__(parent)
        self.setWindowTitle("Add Gesture Class")
        self.setMinimumWidth(380)
        self._existing_names: set[str] = {
            _canonical_key(n) for n in (existing_names or set())
        }

        self.setStyleSheet("""
            QDialog {
                background-color: #181d24;
                color: #f8fafc;
            }
            QLabel {
                color: #cbd5e1;
                font-weight: 500;
            }
            QLineEdit, QComboBox {
                background-color: #0f1319;
                border: 1px solid #2b3441;
                border-radius: 4px;
                padding: 6px 10px;
                color: #f8fafc;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #0d9488;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        form = QFormLayout()
        form.setSpacing(10)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. شكرا or سيارة or حرف أ")
        form.addRow("Class Name (Arabic):", self._name_edit)

        self._type_combo = QComboBox()
        self._type_combo.addItem("Static Sign (Single Hand Shape)", "static")
        self._type_combo.addItem("Dynamic Sequence (Continuous Motion)", "dynamic")
        self._type_combo.addItem("Both (Static & Dynamic Support)", "both")
        self._type_combo.addItem("Alphabet Letter", "alphabet")
        form.addRow("Gesture Type:", self._type_combo)

        self._desc_edit = QLineEdit()
        self._desc_edit.setPlaceholderText("Optional description or English translation")
        form.addRow("Description:", self._desc_edit)

        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _validate_and_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a class name.")
            return

        key = _canonical_key(name)
        if key in self._existing_names:
            QMessageBox.warning(
                self,
                "Duplicate Class",
                f"Class «{name}» already exists in the active vocabulary.\nPlease choose a unique name.",
            )
            return

        self.accept()

    def get_data(self) -> tuple[str, str]:
        """Return (class_name, dataset_category) for backwards compatibility."""
        return self._name_edit.text().strip(), self._type_combo.currentData()

    def get_full_data(self) -> tuple[str, str, str]:
        """Return (class_name, dataset_category, description)."""
        return (
            self._name_edit.text().strip(),
            self._type_combo.currentData(),
            self._desc_edit.text().strip(),
        )

    @property
    def class_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def category(self) -> str:
        return self._type_combo.currentData()

    @property
    def description(self) -> str:
        return self._desc_edit.text().strip()


class EditClassDialog(QDialog):
    """Dialog to rename an existing gesture class and update its category."""

    def __init__(
        self,
        current_name: str,
        current_category: str = "static",
        sample_counts: Optional[dict[str, int]] = None,
        existing_names: Optional[set[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit Gesture Class — «{current_name}»")
        self.setMinimumWidth(400)
        self._current_name = current_name
        self._existing_names: set[str] = {
            _canonical_key(n) for n in (existing_names or set())
        }

        self.setStyleSheet("""
            QDialog {
                background-color: #181d24;
                color: #f8fafc;
            }
            QLabel {
                color: #cbd5e1;
                font-weight: 500;
            }
            QLineEdit, QComboBox {
                background-color: #0f1319;
                border: 1px solid #2b3441;
                border-radius: 4px;
                padding: 6px 10px;
                color: #f8fafc;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #0d9488;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # Informational banner
        counts = sample_counts or {"static": 0, "dynamic": 0, "total": 0}
        total = counts.get("total", 0)
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 6px;
            padding: 8px;
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(6, 6, 6, 6)

        status_text = f"Saved Data: {counts.get('static', 0)} static samples, {counts.get('dynamic', 0)} dynamic sequences"
        if total == 0:
            status_text += " (Class is currently empty)"
        status_label = QLabel(status_text)
        status_label.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 12px;")
        info_layout.addWidget(status_label)

        sub_label = QLabel("Renaming automatically updates all sample directories and metadata.")
        sub_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        info_layout.addWidget(sub_label)
        layout.addWidget(info_frame)

        form = QFormLayout()
        form.setSpacing(10)

        self._name_edit = QLineEdit(current_name)
        form.addRow("New Class Name:", self._name_edit)

        self._type_combo = QComboBox()
        self._type_combo.addItem("Static Sign (Single Hand Shape)", "static")
        self._type_combo.addItem("Dynamic Sequence (Continuous Motion)", "dynamic")
        self._type_combo.addItem("Both (Static & Dynamic Support)", "both")
        self._type_combo.addItem("Alphabet Letter", "alphabet")

        # Select current category
        cat_lower = current_category.lower()
        idx = self._type_combo.findData(cat_lower)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        form.addRow("Gesture Type:", self._type_combo)

        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _validate_and_accept(self) -> None:
        new_name = self._name_edit.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a non-empty name.")
            return

        new_key = _canonical_key(new_name)
        current_key = _canonical_key(self._current_name)

        if new_key != current_key and new_key in self._existing_names:
            QMessageBox.warning(
                self,
                "Duplicate Class",
                f"Class «{new_name}» already exists in the active vocabulary.\nPlease enter a unique name.",
            )
            return

        self.accept()

    def get_new_name(self) -> str:
        return self._name_edit.text().strip()

    def get_new_category(self) -> str:
        return self._type_combo.currentData()

    def get_data(self) -> tuple[str, str]:
        return self.get_new_name(), self.get_new_category()


class DeleteClassDialog(QDialog):
    """Dialog to delete or archive a gesture class with explicit safety controls."""

    def __init__(
        self,
        class_name: str,
        sample_counts: dict[str, int],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Delete / Archive Gesture Class")
        self.setMinimumWidth(440)
        self._class_name = class_name
        self._total_samples = sample_counts.get("total", 0)
        self._static_samples = sample_counts.get("static", 0)
        self._dynamic_samples = sample_counts.get("dynamic", 0)

        self.setStyleSheet("""
            QDialog {
                background-color: #181d24;
                color: #f8fafc;
            }
            QLabel {
                color: #cbd5e1;
            }
            QRadioButton {
                color: #f8fafc;
                font-size: 13px;
                font-weight: 500;
                spacing: 8px;
            }
            QRadioButton:hover {
                color: #38bdf8;
            }
            QCheckBox {
                color: #ef4444;
                font-weight: 600;
                font-size: 12px;
                spacing: 6px;
            }
            QPushButton#dangerButton {
                background-color: #991b1b;
                color: #ffffff;
                font-weight: 600;
                padding: 6px 14px;
                border-radius: 4px;
            }
            QPushButton#dangerButton:hover {
                background-color: #b91c1c;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(18, 18, 18, 18)

        if self._total_samples == 0:
            # Simple clean deletion of empty class
            header = QLabel(f"Remove empty class «{self._class_name}»?")
            header.setStyleSheet("font-size: 14px; font-weight: 700; color: #f8fafc;")
            layout.addWidget(header)

            desc = QLabel("This class contains no recorded training samples and will be removed from your active vocabulary.")
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
            layout.addWidget(desc)

            self._permanent_check = None
            self._archive_radio = None
        else:
            # Class has samples: safe archive vs permanent deletion
            header = QLabel(f"Remove class «{self._class_name}»")
            header.setStyleSheet("font-size: 15px; font-weight: 700; color: #f59e0b;")
            layout.addWidget(header)

            warning_box = QFrame()
            warning_box.setStyleSheet("""
                background-color: #271a06;
                border: 1px solid #78350f;
                border-radius: 6px;
                padding: 10px;
            """)
            warn_layout = QVBoxLayout(warning_box)
            warn_layout.setContentsMargins(8, 8, 8, 8)
            warn_msg = QLabel(
                f"⚠️ This class contains <b>{self._total_samples} recorded samples</b> "
                f"({self._static_samples} static, {self._dynamic_samples} dynamic)."
            )
            warn_msg.setStyleSheet("color: #fbbf24; font-size: 12px;")
            warn_layout.addWidget(warn_msg)
            layout.addWidget(warning_box)

            # Option A: Archive (Safe)
            self._archive_radio = QRadioButton("Archive Class (Safe — Recommended)")
            self._archive_radio.setChecked(True)
            self._archive_desc = QLabel(
                "Removes «" + self._class_name + "» from active vocabulary and training, "
                "but safely keeps all recorded sample files on disk."
            )
            self._archive_desc.setWordWrap(True)
            self._archive_desc.setStyleSheet("color: #94a3b8; font-size: 11px; margin-left: 24px;")
            layout.addWidget(self._archive_radio)
            layout.addWidget(self._archive_desc)

            # Option B: Permanent Delete
            self._delete_radio = QRadioButton("Permanently Delete Class AND Erase Samples")
            self._delete_desc = QLabel(
                "Permanently deletes «" + self._class_name + "» and deletes all "
                + str(self._total_samples) + " sample files from disk. This cannot be undone."
            )
            self._delete_desc.setWordWrap(True)
            self._delete_desc.setStyleSheet("color: #f87171; font-size: 11px; margin-left: 24px;")
            layout.addWidget(self._delete_radio)
            layout.addWidget(self._delete_desc)

            # Confirmation checkbox for permanent deletion
            self._permanent_check = QCheckBox(f"I confirm permanent deletion of all {self._total_samples} sample files")
            self._permanent_check.setVisible(False)
            self._permanent_check.stateChanged.connect(self._on_confirmation_changed)
            layout.addWidget(self._permanent_check)

            self._archive_radio.toggled.connect(self._on_radio_toggled)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._cancel_btn)

        self._ok_btn = QPushButton("Confirm")
        self._ok_btn.setObjectName("dangerButton")
        self._ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self._ok_btn)

        layout.addLayout(btn_layout)

    def _on_radio_toggled(self) -> None:
        if self._archive_radio is not None and self._permanent_check is not None:
            is_archive = self._archive_radio.isChecked()
            self._permanent_check.setVisible(not is_archive)
            if is_archive:
                self._ok_btn.setEnabled(True)
                self._ok_btn.setText("Archive Class")
            else:
                self._ok_btn.setEnabled(self._permanent_check.isChecked())
                self._ok_btn.setText("Delete Permanently")

    def _on_confirmation_changed(self) -> None:
        if self._permanent_check is not None:
            self._ok_btn.setEnabled(self._permanent_check.isChecked())

    def is_permanent_delete(self) -> bool:
        """Return True only if permanent deletion was explicitly selected and confirmed."""
        if self._total_samples == 0:
            return True  # Empty classes are safely removed
        if self._archive_radio is not None and self._archive_radio.isChecked():
            return False
        return self._permanent_check is not None and self._permanent_check.isChecked()


class SampleInspectorDialog(QDialog):
    """Dialog to inspect, review, and delete individual recorded samples for a class."""

    def __init__(
        self,
        class_label: str,
        base_dir: Path = Path("data/datasets"),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._class_label = class_label
        self._base_dir = base_dir
        self._samples_info: list[dict] = []

        self.setWindowTitle(f"Inspect Samples — {class_label}")
        self.resize(680, 420)
        self.setStyleSheet("""
            QDialog {
                background-color: #181d24;
                color: #f8fafc;
            }
            QLabel {
                color: #cbd5e1;
            }
            QTableWidget {
                background-color: #0f1319;
                border: 1px solid #2b3441;
                border-radius: 6px;
                color: #f8fafc;
                gridline-color: #2b3441;
            }
            QHeaderView::section {
                background-color: #1a222d;
                color: #94a3b8;
                font-weight: 600;
                padding: 6px;
                border: 1px solid #2b3441;
            }
        """)

        self._setup_ui()
        self._load_samples()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header_label = QLabel(f"Saved training samples for gesture: «{self._class_label}»")
        header_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #14b8a6;")
        layout.addWidget(header_label)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Sample ID", "Type", "Frames", "Handedness", "Recorded Date"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self._table, stretch=1)

        bottom_row = QHBoxLayout()
        self._summary_label = QLabel("Loading samples...")
        self._summary_label.setStyleSheet("color: #94a3b8; font-weight: 500;")
        bottom_row.addWidget(self._summary_label)
        bottom_row.addStretch()

        self._delete_btn = QPushButton("Delete Selected Sample")
        self._delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #7f1d1d;
                color: #fca5a5;
                border: 1px solid #991b1b;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #991b1b;
                color: #ffffff;
            }
        """)
        self._delete_btn.clicked.connect(self._delete_selected)
        bottom_row.addWidget(self._delete_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom_row.addWidget(close_btn)

        layout.addLayout(bottom_row)

    def _load_samples(self) -> None:
        """Scan samples on disk for this class across words and dynamic directories."""
        self._samples_info.clear()
        self._table.setRowCount(0)

        sanitized = self._class_label.strip().replace(" ", "_").replace("/", "_")

        for cat_dir, cat_name in [("words", "Static"), ("dynamic", "Dynamic"), ("alphabet", "Alphabet")]:
            class_samples_dir = self._base_dir / cat_dir / sanitized / "samples"
            if not class_samples_dir.exists():
                continue

            for sample_dir in sorted(class_samples_dir.iterdir(), reverse=True):
                if not sample_dir.is_dir():
                    continue

                meta_file = sample_dir / "metadata.json"
                num_frames = 1 if cat_dir in ("words", "alphabet") else 0
                handedness = "Right"
                date_str = "--"

                if meta_file.exists():
                    try:
                        with open(meta_file, encoding="utf-8") as f:
                            meta = json.load(f)
                            num_frames = meta.get("num_frames", num_frames)
                            handedness = meta.get("handedness", handedness)
                            ts = meta.get("timestamp", 0)
                            if ts:
                                date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                    except Exception:
                        pass

                self._samples_info.append({
                    "sample_id": sample_dir.name,
                    "type": cat_name,
                    "frames": num_frames,
                    "handedness": handedness,
                    "date": date_str,
                    "path": sample_dir,
                })

        # Populate table
        self._table.setRowCount(len(self._samples_info))
        for row, info in enumerate(self._samples_info):
            self._table.setItem(row, 0, QTableWidgetItem(info["sample_id"]))
            self._table.setItem(row, 1, QTableWidgetItem(info["type"]))
            self._table.setItem(row, 2, QTableWidgetItem(str(info["frames"])))
            self._table.setItem(row, 3, QTableWidgetItem(str(info["handedness"])))
            self._table.setItem(row, 4, QTableWidgetItem(info["date"]))

        static_cnt = sum(1 for s in self._samples_info if s["type"] == "Static")
        dynamic_cnt = sum(1 for s in self._samples_info if s["type"] == "Dynamic")
        alpha_cnt = sum(1 for s in self._samples_info if s["type"] == "Alphabet")
        self._summary_label.setText(
            f"Total: {len(self._samples_info)} samples  ({static_cnt} static, {dynamic_cnt} dynamic, {alpha_cnt} alphabet)"
        )

    def _delete_selected(self) -> None:
        """Delete the selected sample directory from disk."""
        row = self._table.currentRow()
        if row < 0 or row >= len(self._samples_info):
            QMessageBox.information(self, "Select Sample", "Please select a sample to delete.")
            return

        sample_info = self._samples_info[row]
        sample_path: Path = sample_info["path"]

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to permanently delete sample '{sample_info['sample_id']}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if confirm == QMessageBox.Yes:
            try:
                if sample_path.exists():
                    shutil.rmtree(sample_path)
                self._load_samples()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete sample: {e}")
