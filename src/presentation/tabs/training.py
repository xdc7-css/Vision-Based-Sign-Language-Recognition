from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.application.state import AppState
from src.domain.entities import ModelInfo
from src.infrastructure.dialect_manager import DialectManager
from src.infrastructure.recognition.hybrid_pipeline import HybridRecognitionPipeline
from src.models.train_dynamic import inspect_dynamic_dataset, train_dynamic_model
from src.models.train_static import inspect_dataset, train_static_model
from src.presentation.widgets import SectionHeader, StatCard


class TrainingWorker(QThread):
    """Background worker thread for training models without blocking the UI."""

    progress_updated = Signal(int, int, float, float, float)  # epoch, total, loss, val_loss, elapsed_s
    finished_success = Signal(dict)
    finished_error = Signal(str, str)  # error msg, traceback

    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        dialect: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        val_split: float,
        augment: bool,
        is_dynamic: bool = False,
        model_type: str = "mlp",
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.dialect = dialect
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.augment = augment
        self.is_dynamic = is_dynamic
        self.model_type = model_type
        self._is_cancelled = False
        self._start_time = 0.0

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        self._start_time = time.perf_counter()

        def on_progress(ep: int, total: int, loss: float, val_loss: float) -> None:
            if self._is_cancelled:
                raise RuntimeError("تم إلغاء عملية التدريب من قِبل المستخدم.")
            elapsed = time.perf_counter() - self._start_time
            self.progress_updated.emit(ep, total, loss, val_loss, elapsed)

        try:
            if self.is_dynamic:
                arch = "lstm"
                if "gru" in self.model_type.lower():
                    arch = "gru"
                elif "transformer" in self.model_type.lower():
                    arch = "transformer"

                result = train_dynamic_model(
                    dataset_path=self.dataset_path,
                    output_dir=self.output_dir,
                    architectures=[arch],
                    epochs=self.epochs,
                    augment=self.augment,
                    dialect=self.dialect,
                    progress_callback=on_progress,
                )
            else:
                mt = "mlp"
                if "svm" in self.model_type.lower():
                    mt = "svm"
                elif "rf" in self.model_type.lower() or "forest" in self.model_type.lower():
                    mt = "rf"

                result = train_static_model(
                    dataset_path=self.dataset_path,
                    output_dir=self.output_dir,
                    model_types=[mt],
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    val_split=self.val_split,
                    augment=self.augment,
                    dialect=self.dialect,
                    progress_callback=on_progress,
                )
            self.finished_success.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.finished_error.emit(str(e), tb)


class ModelTrainingTab(QWidget):
    """Tab for training, validating, and exporting gesture recognition models."""

    def __init__(
        self,
        state: AppState,
        dialect_manager: Optional[DialectManager] = None,
        pipeline: Optional[HybridRecognitionPipeline] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._dialect_manager = dialect_manager
        self._pipeline = pipeline
        self._worker: Optional[TrainingWorker] = None

        self._setup_ui()
        self._connect_signals()
        self._sync_active_dialect()

    def _sync_active_dialect(self) -> None:
        if hasattr(self._state, "active_dialect") and self._state.active_dialect:
            d_name = (
                getattr(self._state.active_dialect, "name", "")
                or getattr(self._state.active_dialect, "display_name", "")
                or "Iraqi"
            )
            idx = self._dialect_combo.findText(d_name)
            if idx >= 0:
                self._dialect_combo.setCurrentIndex(idx)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        header = SectionHeader(
            "Model Training",
            "Train, validate, and export gesture recognition models — real-time progress",
        )
        layout.addWidget(header)

        # Training configuration
        config_section = QFrame()
        config_section.setObjectName("configSection")
        config_section.setStyleSheet("""
            QFrame#configSection {
                background-color: #181d24;
                border: 1px solid #2b3441;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        config_layout = QHBoxLayout(config_section)

        # Model type group
        model_group = QGroupBox("Model Architecture && Target")
        model_form = QFormLayout(model_group)

        self._model_type = QComboBox()
        self._model_type.addItems(["Static Gesture (MLP / ANN)", "Dynamic Gesture (LSTM)", "Alphabet (CTC)"])
        model_form.addRow("Architecture:", self._model_type)

        self._dialect_combo = QComboBox()
        self._dialect_combo.addItems(["Iraqi", "Saudi", "Egyptian"])
        self._dialect_combo.setCurrentIndex(0)
        model_form.addRow("Target Dialect:", self._dialect_combo)

        config_layout.addWidget(model_group)

        # Hyperparameters group
        hyper_group = QGroupBox("Hyperparameters")
        hyper_form = QFormLayout(hyper_group)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 1000)
        self._epochs_spin.setValue(40)
        self._epochs_spin.setSuffix(" epochs")
        hyper_form.addRow("Epochs:", self._epochs_spin)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 256)
        self._batch_spin.setValue(32)
        hyper_form.addRow("Batch Size:", self._batch_spin)

        self._lr_spin = QSpinBox()
        self._lr_spin.setRange(1, 6)
        self._lr_spin.setValue(3)
        self._lr_spin.setPrefix("1e-")
        hyper_form.addRow("Learning Rate:", self._lr_spin)

        self._val_split = QSpinBox()
        self._val_split.setRange(5, 40)
        self._val_split.setValue(20)
        self._val_split.setSuffix(" %")
        hyper_form.addRow("Validation Split:", self._val_split)

        config_layout.addWidget(hyper_group)

        # Augmentation group
        aug_group = QGroupBox("Data Augmentation")
        aug_form = QVBoxLayout(aug_group)

        self._aug_rotate = QCheckBox("Rotation (±15°)")
        self._aug_rotate.setChecked(True)
        aug_form.addWidget(self._aug_rotate)

        self._aug_scale = QCheckBox("Scale (±10%)")
        self._aug_scale.setChecked(True)
        aug_form.addWidget(self._aug_scale)

        self._aug_flip = QCheckBox("Horizontal Flip")
        self._aug_flip.setChecked(False)
        aug_form.addWidget(self._aug_flip)

        self._aug_jitter = QCheckBox("Landmark Jitter")
        self._aug_jitter.setChecked(True)
        aug_form.addWidget(self._aug_jitter)

        aug_form.addStretch()
        config_layout.addWidget(aug_group)

        layout.addWidget(config_section)

        # Training controls + progress
        train_section = QFrame()
        train_section.setObjectName("trainSection")
        train_section.setStyleSheet("""
            QFrame#trainSection {
                background-color: #181d24;
                border: 1px solid #2b3441;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        train_layout = QVBoxLayout(train_section)

        # Buttons
        btn_row = QHBoxLayout()

        self._train_btn = QPushButton("Start Training")
        self._train_btn.setObjectName("primaryButton")
        btn_row.addWidget(self._train_btn)

        self._stop_btn = QPushButton("Stop Training")
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._stop_btn)

        self._export_btn = QPushButton("Export ONNX Model")
        btn_row.addWidget(self._export_btn)

        btn_row.addStretch()
        train_layout.addLayout(btn_row)

        # Progress bar & epoch label
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Epoch Progress:"))

        self._epoch_progress = QProgressBar()
        self._epoch_progress.setRange(0, self._epochs_spin.value())
        self._epoch_progress.setValue(0)
        progress_layout.addWidget(self._epoch_progress, stretch=1)

        self._epoch_label = QLabel(f"0 / {self._epochs_spin.value()}")
        self._epoch_label.setMinimumWidth(80)
        self._epoch_label.setAlignment(Qt.AlignRight)
        progress_layout.addWidget(self._epoch_label)

        train_layout.addLayout(progress_layout)

        # Metrics stat cards
        metrics_layout = QHBoxLayout()
        self._loss_card = StatCard("Training Loss", "--")
        metrics_layout.addWidget(self._loss_card)
        self._val_loss_card = StatCard("Validation Loss", "--")
        metrics_layout.addWidget(self._val_loss_card)
        self._accuracy_card = StatCard("Accuracy", "--")
        metrics_layout.addWidget(self._accuracy_card)
        self._time_card = StatCard("Elapsed Time", "--")
        metrics_layout.addWidget(self._time_card)
        train_layout.addLayout(metrics_layout)

        # Dataset inspection report label
        self._dataset_info_label = QLabel("Dataset status: Ready to inspect.")
        self._dataset_info_label.setStyleSheet("color: #9cdcfe; font-size: 13px; padding-top: 8px; line-height: 1.4;")
        self._dataset_info_label.setWordWrap(True)
        train_layout.addWidget(self._dataset_info_label)

        train_layout.addStretch()
        layout.addWidget(train_section, stretch=1)

    def _connect_signals(self) -> None:
        self._train_btn.clicked.connect(self._inspect_and_train)
        self._stop_btn.clicked.connect(self._stop_training)
        self._export_btn.clicked.connect(self._export_model)
        self._epochs_spin.valueChanged.connect(self._on_epochs_changed)
        if hasattr(self._state, "active_dialect_changed"):
            self._state.active_dialect_changed.connect(lambda _: self._sync_active_dialect())

    def _export_model(self) -> None:
        """Export trained model checkpoint and vocabulary manifest for deployment."""
        target_dialect = self._dialect_combo.currentText().lower()
        model_dir = Path(f"data/models/{target_dialect}")
        export_dir = Path("data/models/exported")
        export_dir.mkdir(parents=True, exist_ok=True)

        copied = []
        for f in model_dir.glob("*.*"):
            if f.suffix in [".pkl", ".json", ".pt", ".onnx"]:
                import shutil
                dst = export_dir / f.name
                shutil.copy2(f, dst)
                copied.append(f.name)

        if copied:
            QMessageBox.information(
                self,
                "تم تصدير النموذج",
                f"تم تصدير ملفات النموذج بنجاح إلى:\n{export_dir.resolve()}\n\nالملفات المصدرة:\n"
                + "\n".join(f"  • {name}" for name in copied),
            )
        else:
            QMessageBox.warning(
                self,
                "لا توجد نماذج للتصدير",
                f"لم يتم العثور على أي نماذج مدربة في:\n{model_dir}\nيرجى تدريب النموذج أولاً.",
            )

    def _on_epochs_changed(self, value: int) -> None:
        self._epoch_progress.setMaximum(value)
        self._epoch_label.setText(f"{self._epoch_progress.value()} / {value}")

    def _inspect_and_train(self) -> None:
        """Inspect dataset classes, display qualifying vs excluded classes, and launch training."""
        is_dynamic = "Dynamic" in self._model_type.currentText()
        target_dialect = self._dialect_combo.currentText().lower()
        dataset_base = Path("data/datasets/dynamic") if is_dynamic else Path("data/datasets/words")

        if not dataset_base.exists():
            QMessageBox.warning(self, "مجلد البيانات غير موجود", f"المجلد المطلوب {dataset_base} غير موجود.")
            return

        # Detailed inspection (dynamic vs static)
        if is_dynamic:
            inspection = inspect_dynamic_dataset(dataset_base)
        else:
            inspection = inspect_dataset(dataset_base)

        report_lines = [
            f"📊 حالة مجموعة البيانات ({dataset_base.name}):",
            f"  • إجمالي الفئات المفحوصة: {inspection['total_classes']}",
            "",
            "✅ الفئات المقبولة والمستوفية للشروط (عينتان فأكثر):",
        ]

        if inspection["accepted_classes"]:
            for cls_name, cnt in sorted(inspection["accepted_classes"].items()):
                report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: {cnt} عينة (مقبولة للتدريب)")
        else:
            report_lines.append("    (لا توجد أي فئة مستوفية للحد الأدنى)")

        report_lines.append("")
        report_lines.append("⚠️ الفئات المستبعدة (فارغة أو غير كافية):")
        for cls_name, cnt in sorted(inspection["insufficient_classes"].items()):
            report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: {cnt} عينة فقط (مستبعدة — الحد الأدنى عينتان للتقسيم)")
        for cls_name, cnt in sorted(inspection["empty_classes"].items()):
            report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: 0 عينات (مستبعدة — لا توجد عينات مسجلة)")

        report_text = "\n".join(report_lines)
        self._dataset_info_label.setText(report_text)

        if not inspection["is_trainable"]:
            QMessageBox.warning(
                self,
                "بيانات التدريب غير كافية",
                f"{report_text}\n\n"
                f"❌ لا يمكن بدء تدريب المصنف:\n"
                f"يتطلب النموذج فئتين صالحتين على الأقل (كل فئة تحوي عينتين فأكثر لتقسيم التدريب والتحقق).\n"
                f"الفئات المقبولة حالياً: {len(inspection['accepted_classes'])} فئة.\n\n"
                f"يرجى تسجيل عينات إضافية في تبويب Dataset قبل بدء التدريب."
            )
            return

        # Prepare parameters from UI
        epochs = self._epochs_spin.value()
        batch_size = self._batch_spin.value()
        lr_exponent = self._lr_spin.value()
        learning_rate = 10.0 ** (-lr_exponent)
        val_split = self._val_split.value() / 100.0
        augment = any([
            self._aug_rotate.isChecked(),
            self._aug_scale.isChecked(),
            self._aug_flip.isChecked(),
            self._aug_jitter.isChecked(),
        ])

        out_dir = Path(f"data/models/{target_dialect}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # UI state for active training
        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._epoch_progress.setValue(0)
        self._epoch_progress.setMaximum(epochs)
        self._epoch_label.setText(f"0 / {epochs}")
        self._loss_card.set_value("--")
        self._val_loss_card.set_value("--")
        self._accuracy_card.set_value("--")
        self._time_card.set_value("0.0 s")

        self._dataset_info_label.setText(f"{report_text}\n\n⏳ جارٍ التدريب على الفئات المقبولة...")

        # Start asynchronous worker
        self._worker = TrainingWorker(
            dataset_path=dataset_base,
            output_dir=out_dir,
            dialect=target_dialect,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_split=val_split,
            augment=augment,
            is_dynamic=is_dynamic,
            model_type=self._model_type.currentText(),
            parent=self,
        )
        self._worker.progress_updated.connect(self._on_training_progress)
        self._worker.finished_success.connect(lambda res: self._on_training_success(res, report_text, target_dialect))
        self._worker.finished_error.connect(lambda err, tb: self._on_training_error(err, tb, report_text))
        self._worker.start()

    def _stop_training(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._stop_btn.setEnabled(False)
            self._dataset_info_label.setText(self._dataset_info_label.text() + "\n\n⏹ جاري إيقاف التدريب...")

    def _on_training_progress(
        self, epoch: int, total: int, loss: float, val_loss: float, elapsed_s: float
    ) -> None:
        self._epoch_progress.setValue(epoch)
        self._epoch_label.setText(f"{epoch} / {total}")
        self._loss_card.set_value(f"{loss:.4f}")
        self._val_loss_card.set_value(f"{val_loss:.4f}")
        self._time_card.set_value(f"{elapsed_s:.1f} s")

    def _on_training_success(self, result: dict, report_text: str, dialect: str) -> None:
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

        acc = result.get("accuracy", 0.0)
        self._accuracy_card.set_value(f"{acc:.1%}")

        # Reload trained models into active pipeline via DialectManager
        if self._dialect_manager is not None:
            self._dialect_manager.reload_dialect()

        # Update app state
        model_info = ModelInfo(
            dialect_id=dialect,
            model_type=result.get("model_type", "mlp"),
            version="1.0.0",
            file_path=result.get("model_path", ""),
            is_loaded=True,
            accuracy=acc,
        )
        self._state.active_model = model_info

        classes_str = ", ".join([c.replace("_", " ") for c in result.get("classes", [])])
        success_msg = (
            f"{report_text}\n\n"
            f"🎉 اكتمل التدريب بنجاح!\n"
            f"  • مسار النموذج: {result.get('model_path')}\n"
            f"  • مسار القاموس: {result.get('vocab_path')}\n"
            f"  • الدقة المقاسة: {acc:.1%}\n"
            f"  • الفئات المدمجة: {classes_str}\n"
            f"  ✓ تم تحميل النموذج وتفعيله تلقائياً داخل نظام التعرف (Recognition)."
        )
        self._dataset_info_label.setText(success_msg)

        QMessageBox.information(
            self,
            "اكتمل التدريب",
            f"تم تدريب النموذج وحفظه وتفعيله بنجاح!\n\n"
            f"الدقة: {acc:.1%}\n"
            f"الفئات: {classes_str}\n"
            f"مسار النموذج: {result.get('model_path')}\n\n"
            f"النموذج جاهز ومحمّل الآن للاستخدام في تبويب Recognition."
        )

    def _on_training_error(self, error_msg: str, traceback_text: str, report_text: str) -> None:
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._state.error(f"Training failed: {error_msg}")

        self._dataset_info_label.setText(
            f"{report_text}\n\n"
            f"❌ خطأ أثناء التدريب:\n{error_msg}\n\n"
            f"{traceback_text}"
        )
        QMessageBox.critical(
            self,
            "خطأ في التدريب",
            f"فشلت عملية التدريب بسبب الخطأ التالي:\n\n{error_msg}\n\n"
            f"يمكن مراجعة التفاصيل الكاملة في أسفل التبويب."
        )
