"""KArSL-502 Vocabulary Viewer Dialog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class KArSLVocabularyDialog(QDialog):
    """Dialog displaying the complete 502-sign vocabulary of KArSL-502."""

    def __init__(
        self,
        vocabulary: Optional[list[dict[str, Any]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if vocabulary is None:
            vocabulary = self._load_default_vocabulary()
        self._vocab = vocabulary
        self.setWindowTitle("KArSL-502 — Supported Sign Dictionary (502 Signs)")
        self.resize(760, 580)
        self._setup_ui()
        self._populate(self._vocab)

    @staticmethod
    def _load_default_vocabulary() -> list[dict[str, Any]]:
        labels_path = Path("data/models/karsl/labels.json")
        if not labels_path.exists():
            return []
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ar_words = [str(w) for w in data.get("AR_WORDS", [])]
            en_words = [str(w) for w in data.get("EN_WORDS", [])]
            vocab = []
            for i in range(len(ar_words)):
                ar = ar_words[i]
                en = en_words[i] if i < len(en_words) else ""
                if i <= 30:
                    cat = "أرقام (Numbers)"
                elif i <= 69:
                    cat = "حروف أبجدية (Letters)"
                else:
                    cat = "كلمات (Words)"
                vocab.append({"index": i, "arabic": ar, "english": en, "category": cat})
            return vocab
        except Exception:
            return []

    def _setup_ui(self) -> None:
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px 12px;
                color: #ffffff;
                font-size: 14px;
            }
            QTableWidget {
                background-color: #252526;
                gridline-color: #333333;
                border: 1px solid #3c3c3c;
                color: #e0e0e0;
                font-size: 13px;
                selection-background-color: #094771;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #9cdcfe;
                font-weight: 600;
                border: 1px solid #3c3c3c;
                padding: 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Header info
        header_lbl = QLabel(
            "<h2>KArSL-502 Sign Vocabulary</h2>"
            "<p style='color: #9d9d9d;'>Pretrained Spatial-Temporal Transformer model supporting 502 distinct Arabic signs (Numbers, Letters, and Words).</p>"
        )
        layout.addWidget(header_lbl)

        # Search bar
        search_row = QHBoxLayout()
        search_lbl = QLabel("Search / بحث:")
        search_lbl.setStyleSheet("font-weight: 600;")
        search_row.addWidget(search_lbl)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Filter by Arabic word, English translation, or index...")
        self._search_input.textChanged.connect(self._filter_table)
        search_row.addWidget(self._search_input, stretch=1)
        layout.addLayout(search_row)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["#", "الإشارة بالعربية (Arabic)", "English Translation", "التصنيف (Category)"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, stretch=1)

        # Footer
        footer_row = QHBoxLayout()
        self._count_lbl = QLabel(f"Total: {len(self._vocab)} signs")
        self._count_lbl.setStyleSheet("color: #4ec9b0; font-weight: 600;")
        footer_row.addWidget(self._count_lbl)
        footer_row.addStretch()

        close_btn = QPushButton("Close / إغلاق")
        close_btn.setFixedWidth(120)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
        """)
        close_btn.clicked.connect(self.accept)
        footer_row.addWidget(close_btn)
        layout.addLayout(footer_row)

    def _populate(self, items: list[dict[str, Any]]) -> None:
        self._table.setRowCount(len(items))
        for row, item in enumerate(items):
            # Index
            idx_item = QTableWidgetItem(str(item.get("index", row)))
            idx_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 0, idx_item)

            # Arabic
            ar_text = str(item.get("arabic", ""))
            ar_item = QTableWidgetItem(ar_text)
            ar_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 1, ar_item)

            # English
            en_text = str(item.get("english", ""))
            en_item = QTableWidgetItem(en_text)
            en_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 2, en_item)

            # Category
            cat_text = str(item.get("category", ""))
            cat_item = QTableWidgetItem(cat_text)
            cat_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 3, cat_item)

        self._count_lbl.setText(f"Showing {len(items)} of {len(self._vocab)} signs")

    def _filter_table(self, query: str) -> None:
        query = query.strip().lower()
        if not query:
            self._populate(self._vocab)
            return

        filtered = [
            item for item in self._vocab
            if query in str(item.get("arabic", "")).lower()
            or query in str(item.get("english", "")).lower()
            or query in str(item.get("index", ""))
            or query in str(item.get("category", "")).lower()
        ]
        self._populate(filtered)
