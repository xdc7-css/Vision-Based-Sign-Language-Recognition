from __future__ import annotations

import json
import logging
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
    QInputDialog,
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
from src.infrastructure.dataset.gesture_registry import GestureRegistry
from src.infrastructure.dialect_manager import DialectManager
from src.infrastructure.recognition.hybrid_pipeline import HybridRecognitionPipeline
from src.models.train_dynamic import inspect_dynamic_dataset, train_dynamic_model
from src.models.train_multimodal import inspect_multimodal_dataset, train_multimodal_model
from src.models.train_static import inspect_dataset, train_static_model
from src.presentation.widgets import StatCard


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
        allowed_classes: Optional[set[str] | list[str]] = None,
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
        self.allowed_classes = allowed_classes
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
            if "multimodal" in self.model_type.lower():
                result = train_multimodal_model(
                    dataset_path=self.dataset_path,
                    output_dir=self.output_dir,
                    model_type="mlp" if "mlp" in self.model_type.lower() else "svm",
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    val_split=self.val_split,
                    progress_callback=on_progress,
                    allowed_classes=self.allowed_classes,
                )
            elif self.is_dynamic:
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
                    allowed_classes=self.allowed_classes,
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
                    allowed_classes=self.allowed_classes,
                )
            self.finished_success.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.finished_error.emit(str(e), tb)


class ModelTrainingTab(QWidget):
    """Clean, student-friendly 'Train Model' tab for training and validating sign recognition models."""

    navigate_to_recognition = Signal()

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
        self._registry = GestureRegistry()
        self._worker: Optional[TrainingWorker] = None
        self._logger = logging.getLogger(__name__)

        self._setup_ui()
        self._connect_signals()
        self._sync_active_dialect()
        self.refresh_model_sync_status()

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
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(14)

        # ── Header Banner ────────────────────────────────────────────────
        header_banner = QFrame()
        header_banner.setObjectName("trainHeaderBanner")
        header_banner.setStyleSheet("""
            QFrame#trainHeaderBanner {
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
                padding: 10px 14px;
            }
        """)
        h_layout = QHBoxLayout(header_banner)
        h_layout.setContentsMargins(10, 4, 10, 4)

        header_text_box = QVBoxLayout()
        header_text_box.setSpacing(2)
        title_lbl = QLabel("Model Training Studio")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 700; color: #F3F4F6;")
        header_text_box.addWidget(title_lbl)

        sub_lbl = QLabel("Train neural networks from your recorded sign gestures with genuine validation metrics")
        sub_lbl.setStyleSheet("font-size: 12px; color: #94A3B8;")
        header_text_box.addWidget(sub_lbl)
        h_layout.addLayout(header_text_box)

        h_layout.addStretch()

        self._sync_status_badge = QLabel("Checkpoint: Checking...")
        self._sync_status_badge.setStyleSheet("""
            background-color: #101419;
            color: #94A3B8;
            border: 1px solid #2B3442;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
        """)
        h_layout.addWidget(self._sync_status_badge)

        self._export_btn = QPushButton("Export Model Files")
        self._export_btn.setToolTip("Export trained checkpoints and vocabulary manifest for deployment")
        h_layout.addWidget(self._export_btn)

        layout.addWidget(header_banner)

        # ── 3 Summary Overview Cards ─────────────────────────────────────
        overview_box = QHBoxLayout()
        overview_box.setSpacing(10)

        self._trainable_signs_card = StatCard("Trainable Signs", "Checking...")
        overview_box.addWidget(self._trainable_signs_card)

        self._valid_samples_card = StatCard("Multimodal Samples", "Checking...")
        overview_box.addWidget(self._valid_samples_card)

        self._readiness_card = StatCard("Training Readiness", "Ready to Train")
        overview_box.addWidget(self._readiness_card)

        layout.addLayout(overview_box)

        # ── Main Training Action Card ────────────────────────────────────
        main_train_card = QFrame()
        main_train_card.setObjectName("mainTrainCard")
        main_train_card.setStyleSheet("""
            QFrame#mainTrainCard {
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        train_vlayout = QVBoxLayout(main_train_card)
        train_vlayout.setContentsMargins(14, 14, 14, 14)
        train_vlayout.setSpacing(12)

        # Prominent Action Buttons Bar
        action_bar = QHBoxLayout()
        action_bar.setSpacing(10)

        self._train_btn = QPushButton("🚀  Start Training Model")
        self._train_btn.setObjectName("actionButton")
        self._train_btn.setMinimumHeight(44)
        self._train_btn.setMinimumWidth(220)
        action_bar.addWidget(self._train_btn)

        self._stop_btn = QPushButton("⏹  Stop Training")
        self._stop_btn.setObjectName("dangerButton")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setMinimumHeight(44)
        action_bar.addWidget(self._stop_btn)

        action_bar.addStretch()

        self._test_in_recog_btn = QPushButton("🎯  Test Model in Recognition →")
        self._test_in_recog_btn.setObjectName("successButton")
        self._test_in_recog_btn.setMinimumHeight(44)
        self._test_in_recog_btn.setVisible(False)
        self._test_in_recog_btn.clicked.connect(self._on_test_in_recog_clicked)
        action_bar.addWidget(self._test_in_recog_btn)

        train_vlayout.addLayout(action_bar)

        # Progress bar & epoch label
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Training Progress:"))

        self._epoch_progress = QProgressBar()
        self._epoch_progress.setRange(0, 40)
        self._epoch_progress.setValue(0)
        self._epoch_progress.setFixedHeight(10)
        progress_layout.addWidget(self._epoch_progress, stretch=1)

        self._epoch_label = QLabel("0 / 40")
        self._epoch_label.setMinimumWidth(80)
        self._epoch_label.setAlignment(Qt.AlignRight)
        progress_layout.addWidget(self._epoch_label)
        train_vlayout.addLayout(progress_layout)

        # Metrics stat cards
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(8)
        self._loss_card = StatCard("Training Loss", "--")
        metrics_layout.addWidget(self._loss_card)
        self._val_loss_card = StatCard("Validation Loss", "--")
        metrics_layout.addWidget(self._val_loss_card)
        self._accuracy_card = StatCard("Validation Accuracy", "--")
        metrics_layout.addWidget(self._accuracy_card)
        self._time_card = StatCard("Elapsed Time", "--")
        metrics_layout.addWidget(self._time_card)
        train_vlayout.addLayout(metrics_layout)

        # Dataset inspection report label
        self._dataset_info_label = QLabel("Dataset status: Ready to train.")
        self._dataset_info_label.setStyleSheet("color: #94A3B8; font-size: 13px; padding-top: 4px; line-height: 1.4;")
        self._dataset_info_label.setWordWrap(True)
        train_vlayout.addWidget(self._dataset_info_label)

        layout.addWidget(main_train_card)

        # ── Collapsible Advanced Hyperparameters Section ─────────────────
        self._adv_toggle_btn = QPushButton("▸ Show Advanced Hyperparameters & Architecture")
        self._adv_toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #3B82F6;
                text-align: left;
                font-size: 13px;
                font-weight: 600;
                padding: 4px 0;
            }
            QPushButton:hover {
                color: #60A5FA;
            }
        """)
        self._adv_toggle_btn.clicked.connect(self._toggle_advanced)
        layout.addWidget(self._adv_toggle_btn)

        self._adv_container = QFrame()
        self._adv_container.setVisible(False)
        self._adv_container.setStyleSheet("""
            QFrame {
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        adv_outer = QHBoxLayout(self._adv_container)
        adv_outer.setSpacing(14)

        # Model Architecture Group
        model_group = QGroupBox("Architecture & Dialect")
        model_form = QFormLayout(model_group)

        self._model_type = QComboBox()
        self._model_type.addItems([
            "Multimodal Iraqi — Hands + Face (MLP)",
            "Static Gesture (MLP / ANN - Legacy)",
            "Dynamic Gesture (LSTM)",
            "Alphabet (CTC)",
        ])
        model_form.addRow("Architecture:", self._model_type)

        self._dialect_combo = QComboBox()
        self._dialect_combo.addItems(["Iraqi", "Saudi", "Egyptian"])
        self._dialect_combo.setCurrentIndex(0)
        model_form.addRow("Target Dialect:", self._dialect_combo)

        adv_outer.addWidget(model_group)

        # Hyperparameters Group
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

        adv_outer.addWidget(hyper_group)

        # Augmentation Group
        aug_group = QGroupBox("Data Augmentation")
        aug_layout = QVBoxLayout(aug_group)

        self._aug_rotate = QCheckBox("Rotation (±15°)")
        self._aug_rotate.setChecked(True)
        aug_layout.addWidget(self._aug_rotate)

        self._aug_scale = QCheckBox("Scale (±10%)")
        self._aug_scale.setChecked(True)
        aug_layout.addWidget(self._aug_scale)

        self._aug_flip = QCheckBox("Horizontal Flip")
        self._aug_flip.setChecked(False)
        aug_layout.addWidget(self._aug_flip)

        self._aug_jitter = QCheckBox("Landmark Jitter")
        self._aug_jitter.setChecked(True)
        aug_layout.addWidget(self._aug_jitter)

        aug_layout.addStretch()
        adv_outer.addWidget(aug_group)

        layout.addWidget(self._adv_container)

        # ── Academic Research Benchmark Card ─────────────────────────────
        layout.addWidget(self._create_research_box())

        # ── Independent Human Evaluation Card ────────────────────────────
        layout.addWidget(self._create_independent_eval_box())

        layout.addStretch()

    def _create_research_box(self) -> QFrame:
        card = QFrame()
        card.setObjectName("researchCard")
        card.setStyleSheet("""
            QFrame#researchCard {
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(14, 14, 14, 14)
        vbox.setSpacing(10)

        # Header
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        t_lbl = QLabel("🔬  Academic Research Benchmark — KArSL-502 (SVM vs 1D Temporal CNN)")
        t_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #F3F4F6;")
        title_box.addWidget(t_lbl)

        s_lbl = QLabel("Supervisor: Dr. Hiba Hakim Abdulzahra • Dept. of Computer Engineering, University of Basrah • Protocol: Leave-One-Signer-Out (LOSO)")
        s_lbl.setStyleSheet("font-size: 11px; color: #94A3B8;")
        title_box.addWidget(s_lbl)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()

        self._res_eval_badge = QLabel("LOSO: Ready")
        self._res_eval_badge.setStyleSheet("color: #10B981; font-weight: 600; padding: 4px 10px; background: #064e3b; border-radius: 4px; font-size: 11px;")
        hdr_layout.addWidget(self._res_eval_badge)
        vbox.addLayout(hdr_layout)

        # Actions row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._prep_ds_btn = QPushButton("1. Prepare KArSL Subset")
        self._prep_ds_btn.clicked.connect(self._on_prep_research_dataset)
        btn_row.addWidget(self._prep_ds_btn)

        self._train_svm_btn = QPushButton("2. Train SVM (RBF)")
        self._train_svm_btn.clicked.connect(self._on_train_research_svm)
        btn_row.addWidget(self._train_svm_btn)

        self._train_cnn_btn = QPushButton("3. Train 1D CNN")
        self._train_cnn_btn.clicked.connect(self._on_train_research_cnn)
        btn_row.addWidget(self._train_cnn_btn)

        self._compare_btn = QPushButton("4. Run LOSO Comparison")
        self._compare_btn.setObjectName("actionButton")
        self._compare_btn.clicked.connect(self._on_run_research_comparison)
        btn_row.addWidget(self._compare_btn)

        btn_row.addStretch()
        vbox.addLayout(btn_row)

        # Metrics Summary Table / Text
        self._research_results_label = QLabel(
            "Evaluation Protocol: Train on Signer01 + Signer02 → Test on unseen Signer03 (LOSO).\n"
            "Dataset: Authentic KArSL-502 (Sidig et al., ACM IMWUT 2021) — 10 signs (0-9) across 3 signers (240 videos).\n"
            "Click '4. Run LOSO Comparison' to evaluate SVM and 1D Temporal CNN on authentic benchmark data."
        )
        self._research_results_label.setStyleSheet("color: #F3F4F6; font-size: 12px; background: #101419; border: 1px solid #2B3442; border-radius: 6px; padding: 10px; line-height: 1.5;")
        self._research_results_label.setWordWrap(True)
        vbox.addWidget(self._research_results_label)

        # Quick Test in Recognition Buttons
        test_row = QHBoxLayout()
        test_row.setSpacing(8)
        self._test_svm_btn = QPushButton("🎯  Test Research SVM in Recognition →")
        self._test_svm_btn.clicked.connect(lambda: self._test_research_model_in_recog("research_svm"))
        test_row.addWidget(self._test_svm_btn)

        self._test_cnn_btn = QPushButton("🏆  Test Research CNN in Recognition →")
        self._test_cnn_btn.setObjectName("successButton")
        test_row.addWidget(self._test_cnn_btn)
        self._test_cnn_btn.clicked.connect(lambda: self._test_research_model_in_recog("research_cnn"))

        test_row.addStretch()
        vbox.addLayout(test_row)

        return card

    def _create_independent_eval_box(self) -> QFrame:
        card = QFrame()
        card.setObjectName("independentEvalCard")
        card.setStyleSheet("""
            QFrame#independentEvalCard {
                background-color: #151a21;
                border: 1px solid #2B3442;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(14, 14, 14, 14)
        vbox.setSpacing(10)

        # Header
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        t_lbl = QLabel("👥  Independent Human Recognition Evaluation (Unseen Volunteers)")
        t_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #F3F4F6;")
        title_box.addWidget(t_lbl)

        s_lbl = QLabel(
            "Protocol: 10 Arabic Digits (0–9) × 5 Repetitions • Dual Frozen Models (SVM vs CNN) • Zero Retraining"
        )
        s_lbl.setStyleSheet("font-size: 11px; color: #94A3B8;")
        title_box.addWidget(s_lbl)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()

        self._ind_status_badge = QLabel("Status: PENDING")
        self._ind_status_badge.setStyleSheet(
            "color: #F59E0B; font-weight: 600; padding: 4px 10px; background: #451a03; border-radius: 4px; font-size: 11px;"
        )
        hdr_layout.addWidget(self._ind_status_badge)
        vbox.addLayout(hdr_layout)

        # Info label explaining error disentanglement and storage isolation
        info_lbl = QLabel(
            "ℹ️ Scientific Protocol: Newly recruited volunteers record 10 Arabic digits (0001–0010) with 5 repetitions.\n"
            "All recordings are saved isolated under data/research/independent_evaluation/ and are NEVER retrained on.\n"
            "MediaPipe tracking dropouts are disentangled from model errors, and resting frames are tested for false positives."
        )
        info_lbl.setStyleSheet(
            "color: #94A3B8; font-size: 11px; background: #101419; border: 1px solid #2B3442; border-radius: 6px; padding: 8px; line-height: 1.4;"
        )
        info_lbl.setWordWrap(True)
        vbox.addWidget(info_lbl)

        # Controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self._vol_combo = QComboBox()
        self._vol_combo.setMinimumWidth(180)
        ctrl_row.addWidget(QLabel("Volunteer:"))
        ctrl_row.addWidget(self._vol_combo)

        self._reg_vol_btn = QPushButton("➕  Register New Volunteer")
        self._reg_vol_btn.clicked.connect(self._on_register_volunteer_clicked)
        ctrl_row.addWidget(self._reg_vol_btn)

        self._run_ind_eval_btn = QPushButton("🚀  Run Frozen Evaluation")
        self._run_ind_eval_btn.setObjectName("actionButton")
        self._run_ind_eval_btn.clicked.connect(self._on_run_independent_eval_clicked)
        ctrl_row.addWidget(self._run_ind_eval_btn)

        ctrl_row.addStretch()
        vbox.addLayout(ctrl_row)

        # Results Label
        self._ind_results_label = QLabel("Awaiting volunteer recording sessions. Status is strictly pending.")
        self._ind_results_label.setStyleSheet(
            "color: #F3F4F6; font-size: 12px; background: #101419; border: 1px solid #2B3442; border-radius: 6px; padding: 10px; line-height: 1.5;"
        )
        self._ind_results_label.setWordWrap(True)
        vbox.addWidget(self._ind_results_label)

        self._refresh_independent_eval_ui()
        return card

    def _refresh_independent_eval_ui(self) -> None:
        try:
            from src.research.independent_evaluation import IndependentEvaluationManager
            mgr = IndependentEvaluationManager()
            vols = mgr.list_volunteers()
            self._vol_combo.clear()
            if vols:
                for v in vols:
                    self._vol_combo.addItem(f"{v['name']} ({v.get('total_recordings', 0)}/50)", v["volunteer_id"])
            else:
                self._vol_combo.addItem("No volunteers registered", "")

            st = mgr.get_status()
            if st.get("is_evaluated", False):
                self._ind_status_badge.setText(f"✅ Evaluated ({st.get('total_tested_recordings', 0)} videos)")
                self._ind_status_badge.setStyleSheet(
                    "color: #10B981; font-weight: 600; padding: 4px 10px; background: #064e3b; border-radius: 4px; font-size: 11px;"
                )
                tracking_rate = st.get("tracking_success_rate", 0.0) * 100
                svm_acc = st.get("conditioned_accuracy", {}).get("svm", 0.0) * 100
                cnn_acc = st.get("conditioned_accuracy", {}).get("cnn", 0.0) * 100
                fp_svm = st.get("idle_evaluation", {}).get("svm_false_positive_rate", 0.0) * 100
                fp_cnn = st.get("idle_evaluation", {}).get("cnn_false_positive_rate", 0.0) * 100
                self._ind_results_label.setText(
                    f"Independent Volunteer Benchmark Results:\n"
                    f"• Landmark Tracking Success Rate: {tracking_rate:.1f}%\n"
                    f"• Model A (SVM): Conditioned Acc {svm_acc:.1f}% | Idle False Positive Rate: {fp_svm:.1f}%\n"
                    f"• Model B (CNN): Conditioned Acc {cnn_acc:.1f}% | Idle False Positive Rate: {fp_cnn:.1f}%\n"
                    f"• Tested on {st.get('volunteers_count', 0)} unseen human volunteers."
                )
            else:
                self._ind_status_badge.setText("⚠️ Status: PENDING")
                self._ind_status_badge.setStyleSheet(
                    "color: #F59E0B; font-weight: 600; padding: 4px 10px; background: #451a03; border-radius: 4px; font-size: 11px;"
                )
                self._ind_results_label.setText(
                    "Independent Evaluation Status: PENDING_VOLUNTEER_RECORDINGS\n"
                    "In accordance with research integrity standards, performance remains unmeasured "
                    "until real human volunteers complete recording sessions. Zero synthetic data is substituted."
                )
        except Exception as e:
            self._logger.debug(f"Independent eval UI refresh note: {e}")

    def _on_register_volunteer_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, "Register Volunteer", "Enter Volunteer Name or Alias:")
        if not ok or not name.strip():
            return
        notes, ok_n = QInputDialog.getText(self, "Volunteer Setup", "Enter Camera / Environment Notes (optional):")
        try:
            from src.research.independent_evaluation import IndependentEvaluationManager
            mgr = IndependentEvaluationManager()
            mgr.register_volunteer(name.strip(), notes.strip() if ok_n else "")
            QMessageBox.information(self, "Volunteer Registered", f"Volunteer '{name.strip()}' registered successfully.")
            self._refresh_independent_eval_ui()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to register volunteer: {e}")

    def _on_run_independent_eval_clicked(self) -> None:
        try:
            from src.research.independent_evaluation import IndependentEvaluationManager
            mgr = IndependentEvaluationManager()
            res = mgr.evaluate_independent_dataset()
            self._refresh_independent_eval_ui()
            if not res.get("is_evaluated", False):
                QMessageBox.information(
                    self,
                    "Evaluation Pending",
                    "No accepted volunteer recordings were found in data/research/independent_evaluation/.\n"
                    "Status remains strictly PENDING until volunteers record the 10 Arabic digits."
                )
            else:
                QMessageBox.information(
                    self,
                    "Evaluation Complete",
                    f"Independent volunteer evaluation complete!\n"
                    f"Tracking Rate: {res.get('tracking_success_rate', 0.0)*100:.1f}%\n"
                    f"SVM Acc: {res.get('conditioned_accuracy', {}).get('svm', 0.0)*100:.1f}%\n"
                    f"CNN Acc: {res.get('conditioned_accuracy', {}).get('cnn', 0.0)*100:.1f}%"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Evaluation failed: {e}")

    def _test_research_model_in_recog(self, engine_id: str) -> None:
        if hasattr(self._state, "active_engine"):
            self._state.active_engine = engine_id
        self.navigate_to_recognition.emit()

    def _on_prep_research_dataset(self) -> None:
        try:
            from src.research.karsl_dataset import KArSLDatasetManager
            dm = KArSLDatasetManager()
            status = dm.prepare_dataset()
            provenance = status.get("provenance", "KArSL-502")
            total = status.get("total_samples", 0)
            n_classes = status.get("num_classes", 10)
            QMessageBox.information(
                self,
                "KArSL Dataset",
                f"KArSL-502 research subset prepared successfully ({total} samples across {n_classes} classes).\n"
                f"Provenance: {provenance}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare dataset: {e}")

    def _on_train_research_svm(self) -> None:
        try:
            from src.research.karsl_dataset import KArSLDatasetManager
            from src.research.svm_pipeline import ResearchSVMPipeline
            dm = KArSLDatasetManager()
            X_all, y_all, _ = dm.load_features()
            splits = dm.load_loso_splits()["test_Signer03"]
            svm_pipe = ResearchSVMPipeline()
            res = svm_pipe.train(
                X_train_seq=X_all[splits["train_indices"]],
                y_train=y_all[splits["train_indices"]],
                X_val_seq=X_all[splits["test_indices"]],
                y_val=y_all[splits["test_indices"]],
            )
            QMessageBox.information(self, "SVM Trained", f"Model A (SVM) trained successfully!\nValidation Accuracy: {res['val_accuracy']*100:.1f}%\nF1-score: {res['val_macro_f1']*100:.1f}%")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train SVM: {e}")

    def _on_train_research_cnn(self) -> None:
        try:
            from src.research.karsl_dataset import KArSLDatasetManager
            from src.research.cnn_pipeline import ResearchCNNPipeline
            dm = KArSLDatasetManager()
            X_all, y_all, _ = dm.load_features()
            splits = dm.load_loso_splits()["test_Signer03"]
            cnn_pipe = ResearchCNNPipeline()
            res = cnn_pipe.train(
                X_train_seq=X_all[splits["train_indices"]],
                y_train=y_all[splits["train_indices"]],
                X_val_seq=X_all[splits["test_indices"]],
                y_val=y_all[splits["test_indices"]],
                epochs=35,
            )
            QMessageBox.information(self, "CNN Trained", f"Model B (1D CNN) trained successfully!\nValidation Accuracy: {res['val_accuracy']*100:.1f}%\nF1-score: {res['val_macro_f1']*100:.1f}%")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train CNN: {e}")

    def _on_run_research_comparison(self) -> None:
        try:
            from src.research.evaluator import ResearchScientificEvaluator
            evaluator = ResearchScientificEvaluator()
            res = evaluator.run_loso_comparison()
            winner = res["winner"]["best_model"]
            self._research_results_label.setText(
                f"LOSO Benchmark Complete on Unseen Signer03 (120 test samples):\n"
                f"• Model A (SVM): Accuracy {res['svm']['accuracy']*100:.2f}% | F1 {res['svm']['f1_macro']*100:.2f}% | Latency {res['svm']['inference_latency_ms']:.2f}ms\n"
                f"• Model B (1D CNN): Accuracy {res['cnn']['accuracy']*100:.2f}% | F1 {res['cnn']['f1_macro']*100:.2f}% | Latency {res['cnn']['inference_latency_ms']:.2f}ms\n"
                f"🏆 Scientific Winner: {winner} (Accuracy advantage: +{(res['cnn']['accuracy'] - res['svm']['accuracy'])*100:.1f}%)."
            )
            QMessageBox.information(self, "Scientific Benchmark Complete", f"LOSO Comparison Finished!\nWinner: {winner}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison failed: {e}")

    def _toggle_advanced(self) -> None:
        vis = not self._adv_container.isVisible()
        self._adv_container.setVisible(vis)
        self._adv_toggle_btn.setText("▾ Hide Advanced Hyperparameters & Architecture" if vis else "▸ Show Advanced Hyperparameters & Architecture")

    def _connect_signals(self) -> None:
        self._train_btn.clicked.connect(self._inspect_and_train)
        self._stop_btn.clicked.connect(self._stop_training)
        self._export_btn.clicked.connect(self._export_model)
        self._epochs_spin.valueChanged.connect(self._on_epochs_changed)
        self._dialect_combo.currentIndexChanged.connect(lambda _: self.refresh_model_sync_status())
        self._model_type.currentIndexChanged.connect(lambda _: self.refresh_model_sync_status())
        if hasattr(self._state, "active_dialect_changed"):
            self._state.active_dialect_changed.connect(lambda _: self._sync_active_dialect())

    def _on_test_in_recog_clicked(self) -> None:
        self.navigate_to_recognition.emit()

    def refresh_model_sync_status(self) -> None:
        """Verify whether the active dataset classes match the trained model checkpoint and update summary cards."""
        if not hasattr(self, "_sync_status_badge"):
            return
        try:
            active_registry = {c.name.replace(" ", "_") for c in self._registry.list_active_classes()}
            dataset_base = Path("data/datasets/words")
            inspection = inspect_multimodal_dataset(dataset_base, allowed_classes=active_registry)

            # Update 3 Summary Overview Cards
            trainable_count = len(inspection.get("accepted_classes", {}))
            multimodal_samples_count = sum(inspection.get("accepted_classes", {}).values())

            self._trainable_signs_card.set_value(f"{trainable_count} Signs")
            self._valid_samples_card.set_value(f"{multimodal_samples_count} Samples")

            if trainable_count >= 2:
                self._readiness_card.set_value("Ready to Train")
            else:
                self._readiness_card.set_value(f"Need ≥2 signs ({trainable_count} ready)")

            # Checkpoint synchronization badge
            classes_path = Path("data/models/multimodal/classes.json")
            if not classes_path.exists():
                classes_path = Path("data/models/iraqi/classes.json")

            if not classes_path.exists():
                self._sync_status_badge.setText("⚠️ Not Trained Yet")
                self._sync_status_badge.setStyleSheet(
                    "background-color: #451a03; color: #F59E0B; border: 1px solid #78350f; "
                    "border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 600;"
                )
                return

            with open(classes_path, "r", encoding="utf-8") as f:
                trained_classes = set(json.load(f))
            qualifying = set(inspection["accepted_classes"].keys())

            if trained_classes == qualifying and qualifying:
                self._sync_status_badge.setText(f"✅ Synchronized ({len(trained_classes)} signs)")
                self._sync_status_badge.setStyleSheet(
                    "background-color: #064e3b; color: #10B981; border: 1px solid #047857; "
                    "border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 600;"
                )
            else:
                self._sync_status_badge.setText(
                    f"⚠️ Retraining Recommended ({len(qualifying)} active vs {len(trained_classes)} in checkpoint)"
                )
                self._sync_status_badge.setStyleSheet(
                    "background-color: #451a03; color: #F59E0B; border: 1px solid #78350f; "
                    "border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 600;"
                )
        except Exception as e:
            self._logger.debug(f"Sync check note: {e}")

    def _export_model(self) -> None:
        """Export trained model checkpoint and vocabulary manifest for deployment."""
        target_dialect = self._dialect_combo.currentText().lower()
        model_dir = Path("data/models/multimodal")
        if not model_dir.exists():
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
                "Model Exported",
                f"Model files exported successfully to:\n{export_dir.resolve()}\n\nFiles:\n"
                + "\n".join(f"  • {name}" for name in copied),
            )
        else:
            QMessageBox.warning(
                self,
                "No Model to Export",
                f"No trained models found in:\n{model_dir}\nPlease train the model first.",
            )

    def _on_epochs_changed(self, value: int) -> None:
        self._epoch_progress.setMaximum(value)
        self._epoch_label.setText(f"{self._epoch_progress.value()} / {value}")

    def _inspect_and_train(self) -> None:
        is_multimodal = "Multimodal" in self._model_type.currentText()
        is_dynamic = "Dynamic" in self._model_type.currentText()
        target_dialect = self._dialect_combo.currentText().lower()
        dataset_base = Path("data/datasets/dynamic") if is_dynamic else Path("data/datasets/words")

        if not dataset_base.exists():
            QMessageBox.warning(self, "Directory Missing", f"Directory {dataset_base} not found.")
            return

        active_registry = {c.name.replace(" ", "_") for c in self._registry.list_active_classes()}

        if is_multimodal:
            inspection = inspect_multimodal_dataset(dataset_base, allowed_classes=active_registry)
            report_lines = [
                f"📊 Multimodal Dataset Status ({dataset_base.name} - Hands + Face):",
                f"  • Total classes inspected: {inspection['total_classes']}",
                "",
                "✅ Trainable Qualifying Signs (≥2 samples):",
            ]
            if inspection["accepted_classes"]:
                for cls_name, cnt in sorted(inspection["accepted_classes"].items()):
                    report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: {cnt} multimodal samples")
            else:
                report_lines.append("    (No signs have ≥2 multimodal samples)")

            report_lines.append("")
            report_lines.append("ℹ️ Legacy v1 signs (single-hand only — new multimodal samples needed):")
            for cls_name, cnt in sorted(inspection.get("legacy_only_classes", {}).items()):
                report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: {cnt} legacy single-hand samples")
            for cls_name, cnt in sorted(inspection.get("empty_classes", {}).items()):
                report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: 0 samples")

        elif is_dynamic:
            inspection = inspect_dynamic_dataset(dataset_base, allowed_classes=active_registry)
            report_lines = [
                f"📊 Dynamic Dataset Status ({dataset_base.name}):",
                f"  • Total classes inspected: {inspection['total_classes']}",
                "",
                "✅ Trainable Qualifying Signs (≥2 sequences):",
            ]
            if inspection["accepted_classes"]:
                for cls_name, cnt in sorted(inspection["accepted_classes"].items()):
                    report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: {cnt} sequences")
            else:
                report_lines.append("    (No signs have ≥2 sequences)")

            report_lines.append("")
            report_lines.append("⚠️ Excluded signs (empty or insufficient):")
            for cls_name, cnt in sorted(inspection["insufficient_classes"].items()):
                report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: {cnt} sequences (need ≥2)")
            for cls_name, cnt in sorted(inspection["empty_classes"].items()):
                report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: 0 sequences")

        else:
            inspection = inspect_dataset(dataset_base, allowed_classes=active_registry)
            report_lines = [
                f"📊 Dataset Status ({dataset_base.name}):",
                f"  • Total classes inspected: {inspection['total_classes']}",
                "",
                "✅ Trainable Qualifying Signs (≥2 samples):",
            ]
            if inspection["accepted_classes"]:
                for cls_name, cnt in sorted(inspection["accepted_classes"].items()):
                    report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: {cnt} samples")
            else:
                report_lines.append("    (No signs have ≥2 samples)")

            report_lines.append("")
            report_lines.append("⚠️ Excluded signs (empty or insufficient):")
            for cls_name, cnt in sorted(inspection["insufficient_classes"].items()):
                report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: {cnt} sample only (need ≥2)")
            for cls_name, cnt in sorted(inspection["empty_classes"].items()):
                report_lines.append(f"    - «{cls_name.replace('_', ' ')}»: 0 samples")

        report_text = "\n".join(report_lines)
        self._dataset_info_label.setText(report_text)

        if not inspection["is_trainable"]:
            if is_multimodal:
                fail_msg = (
                    f"{report_text}\n\n"
                    f"❌ Cannot start training multimodal classifier:\n"
                    f"Requires at least 2 signs with ≥2 samples each in multimodal format (Hands + Face).\n"
                    f"Currently qualifying signs: {len(inspection['accepted_classes'])} signs.\n\n"
                    f"💡 Please open the 'Teach Signs' tab and record new samples with Hands + Face."
                )
            else:
                fail_msg = (
                    f"{report_text}\n\n"
                    f"❌ Cannot start training classifier:\n"
                    f"Requires at least 2 signs with ≥2 samples each for training and validation split.\n"
                    f"Currently qualifying signs: {len(inspection['accepted_classes'])} signs.\n\n"
                    f"Please record additional samples in the 'Teach Signs' tab."
                )
            QMessageBox.warning(self, "Insufficient Training Data", fail_msg)
            return

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

        if is_multimodal:
            out_dir = Path("data/models/multimodal")
        else:
            out_dir = Path(f"data/models/{target_dialect}")
        out_dir.mkdir(parents=True, exist_ok=True)

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._test_in_recog_btn.setVisible(False)
        self._epoch_progress.setValue(0)
        self._epoch_progress.setMaximum(epochs)
        self._epoch_label.setText(f"0 / {epochs}")
        self._loss_card.set_value("--")
        self._val_loss_card.set_value("--")
        self._accuracy_card.set_value("--")
        self._time_card.set_value("0.0 s")

        self._dataset_info_label.setText(f"{report_text}\n\n⏳ Training neural network on qualified signs...")

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
            allowed_classes=active_registry,
            parent=self,
        )
        self._worker.progress_updated.connect(self._on_training_progress)
        self._worker.finished_success.connect(
            lambda res: self._on_training_success(res, report_text, "multimodal" if is_multimodal else target_dialect)
        )
        self._worker.finished_error.connect(lambda err, tb: self._on_training_error(err, tb, report_text))
        self._worker.start()

    def _stop_training(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._stop_btn.setEnabled(False)
            self._dataset_info_label.setText(self._dataset_info_label.text() + "\n\n⏹ Stopping training...")

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
        self._test_in_recog_btn.setVisible(True)

        acc = result.get("accuracy", 0.0)
        self._accuracy_card.set_value(f"{acc:.1%}")

        if self._dialect_manager is not None:
            self._dialect_manager.reload_dialect()

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
        self.refresh_model_sync_status()
        success_msg = (
            f"{report_text}\n\n"
            f"🎉 Training completed successfully!\n"
            f"  • Model checkpoint: {result.get('model_path')}\n"
            f"  • Vocabulary file: {result.get('vocab_path')}\n"
            f"  • Measured Validation Accuracy: {acc:.1%}\n"
            f"  • Trained Signs: {classes_str}\n"
            f"  ✓ Model loaded into memory. Click 'Test Model in Recognition' to begin live testing!"
        )
        self._dataset_info_label.setText(success_msg)

        QMessageBox.information(
            self,
            "Training Complete",
            f"Model trained, validated, and saved successfully!\n\n"
            f"Validation Accuracy: {acc:.1%}\n"
            f"Trained Signs: {classes_str}\n\n"
            f"Click 'Test Model in Recognition →' to test your newly trained signs live!",
        )

    def _on_training_error(self, error_msg: str, traceback_text: str, report_text: str) -> None:
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._test_in_recog_btn.setVisible(False)
        self._state.error(f"Training failed: {error_msg}")

        self._dataset_info_label.setText(
            f"{report_text}\n\n"
            f"❌ Training error:\n{error_msg}\n\n"
            f"{traceback_text}"
        )
        QMessageBox.critical(
            self,
            "Training Error",
            f"Training failed with error:\n\n{error_msg}",
        )
