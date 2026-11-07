"""Dialogs for Dataset Studio: Add class, Edit class, and Inspect/Delete individual samples."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AddClassDialog(QDialog):
    """Dialog to create a new gesture class."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Gesture Class")
        self.setMinimumWidth(360)
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
                padding: 6px;
                color: #f8fafc;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. شكرا or ممتاز")
        form.addRow("Class Name (Arabic):", self._name_edit)

        self._type_combo = QComboBox()
        self._type_combo.addItem("Static Sign (Single Hand Shape)", "words")
        self._type_combo.addItem("Dynamic Sequence (Continuous Motion)", "dynamic")
        form.addRow("Gesture Type:", self._type_combo)

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
        self.accept()

    def get_data(self) -> tuple[str, str]:
        """Return (class_name, dataset_type)."""
        return self._name_edit.text().strip(), self._type_combo.currentData()


class EditClassDialog(QDialog):
    """Dialog to rename an existing gesture class."""

    def __init__(self, current_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Gesture Class")
        self.setMinimumWidth(360)
        self.setStyleSheet("""
            QDialog {
                background-color: #181d24;
                color: #f8fafc;
            }
            QLabel {
                color: #cbd5e1;
                font-weight: 500;
            }
            QLineEdit {
                background-color: #0f1319;
                border: 1px solid #2b3441;
                border-radius: 4px;
                padding: 6px;
                color: #f8fafc;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        form = QFormLayout()
        self._name_edit = QLineEdit(current_name)
        form.addRow("New Class Name:", self._name_edit)
        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _validate_and_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a non-empty name.")
            return
        self.accept()

    def get_new_name(self) -> str:
        return self._name_edit.text().strip()


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

        for cat_dir, cat_name in [("words", "Static"), ("dynamic", "Dynamic")]:
            class_samples_dir = self._base_dir / cat_dir / sanitized / "samples"
            if not class_samples_dir.exists():
                continue

            for sample_dir in sorted(class_samples_dir.iterdir(), reverse=True):
                if not sample_dir.is_dir():
                    continue

                meta_file = sample_dir / "metadata.json"
                num_frames = 1 if cat_dir == "words" else 0
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
        self._summary_label.setText(
            f"Total: {len(self._samples_info)} samples  ({static_cnt} static, {dynamic_cnt} dynamic)"
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
