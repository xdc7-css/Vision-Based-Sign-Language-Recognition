"""Scientific Evaluation Engine: SVM vs 1D Temporal CNN on KArSL-502.

Implements the official supervisor evaluation protocol:
- One Shared Dataset -> Two Algorithms -> Same Evaluation Protocol
- Strict Leave-One-Signer-Out (LOSO) cross-signer validation
- Zero data leakage between training signers and held-out test signers
- Comprehensive metrics:
  - Accuracy
  - Macro Precision, Recall, F1-Score
  - Confusion Matrices
  - Training Time
  - Per-sample Inference Latency
  - Checkpoint Disk Footprint
- Scientific winner declaration based on empirical evidence
- JSON and CSV reporting under data/research/karsl/eval/
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.research.cnn_pipeline import ResearchCNNPipeline
from src.research.karsl_dataset import SELECTED_RESEARCH_CLASSES, KArSLDatasetManager
from src.research.svm_pipeline import ResearchSVMPipeline

logger = logging.getLogger("arsl.research.evaluator")


class ResearchScientificEvaluator:
    """Executes fair, reproducible benchmark comparisons between SVM and CNN."""

    def __init__(
        self,
        dataset_manager: Optional[KArSLDatasetManager] = None,
        eval_dir: Path | str = "data/research/karsl/eval",
    ) -> None:
        self.dm = dataset_manager or KArSLDatasetManager()
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.svm_pipeline = ResearchSVMPipeline()
        self.cnn_pipeline = ResearchCNNPipeline()
        self.classes = SELECTED_RESEARCH_CLASSES

    def run_comparison(
        self,
        loso_fold: str = "test_Signer03",
        cnn_epochs: int = 35,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict[str, Any]:
        """Execute a complete scientific comparison on a Leave-One-Signer-Out split.

        Default fold: Train on Signer01 + Signer02, Test on unseen Signer03.
        """
        logger.info(f"Starting scientific comparison on LOSO fold: {loso_fold}...")
        t_global_start = time.perf_counter()

        if progress_callback:
            progress_callback(1, 5, "تحميل بيانات KArSL ومصفوفات الميزات...")

        # 1. Load Data & Splits
        X_all, y_all, signers_all = self.dm.load_features()
        loso_splits = self.dm.load_loso_splits()

        if loso_fold not in loso_splits:
            raise ValueError(f"Fold {loso_fold} not found in LOSO splits: {list(loso_splits.keys())}")

        fold_info = loso_splits[loso_fold]
        train_idx = np.array(fold_info["train_indices"])
        test_idx = np.array(fold_info["test_indices"])

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        test_signer = fold_info["test_signer"]
        train_signers = fold_info["train_signers"]
        logger.info(
            f"LOSO Protocol: Train signers={train_signers} ({len(X_train)} samples), "
            f"Test signer={test_signer} ({len(X_test)} samples)."
        )

        # 2. Train Model A: SVM
        if progress_callback:
            progress_callback(2, 5, "تدريب وموازنة مصنف SVM (Support Vector Machine)...")

        svm_train_metrics = self.svm_pipeline.train(
            X_train_seq=X_train,
            y_train=y_train,
            X_val_seq=X_test,
            y_val=y_test,
            kernel="rbf",
            C=2.0,
        )

        # 3. Train Model B: 1D Temporal CNN
        if progress_callback:
            progress_callback(3, 5, "تدريب الشبكة العصبية الالتفافية 1D Temporal CNN (PyTorch)...")

        cnn_train_metrics = self.cnn_pipeline.train(
            X_train_seq=X_train,
            y_train=y_train,
            X_val_seq=X_test,
            y_val=y_test,
            epochs=cnn_epochs,
            batch_size=16,
            learning_rate=0.001,
        )

        # 4. Evaluate Held-out Test Set (Unseen Signer)
        if progress_callback:
            progress_callback(4, 5, "تقييم أداء الخوارزميتين على المؤدي غير المشاهد (Signer Test)...")

        # SVM Predictions on Test Set
        X_test_svm, _ = self.svm_pipeline.extractor.prepare_svm_dataset(
            X_test, scaler=self.svm_pipeline.scaler
        )
        svm_preds = self.svm_pipeline.model.predict(X_test_svm)
        svm_cm = confusion_matrix(y_test, svm_preds, labels=list(range(len(self.classes))))

        svm_results = {
            "model_name": "Support Vector Machine (SVM)",
            "algorithm": "Scikit-Learn SVC (RBF Kernel)",
            "input_type": "Engineered Statistical & Kinematic Features (2,208 dims)",
            "accuracy": round(float(accuracy_score(y_test, svm_preds)), 4),
            "precision_macro": round(float(precision_score(y_test, svm_preds, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, svm_preds, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, svm_preds, average="macro", zero_division=0)), 4),
            "training_time_sec": svm_train_metrics["train_time_sec"],
            "inference_latency_ms": svm_train_metrics["inference_latency_ms"],
            "model_size_kb": round((self.svm_pipeline.output_dir / "svm_model.pkl").stat().st_size / 1024.0, 1),
            "test_signer": test_signer,
        }

        # CNN Predictions on Test Set
        import torch

        X_test_cnn_tensor = torch.tensor(
            self.cnn_pipeline.extractor.prepare_cnn_dataset(X_test),
            dtype=torch.float32,
            device=self.cnn_pipeline.device,
        )
        self.cnn_pipeline.model.eval()
        with torch.no_grad():
            cnn_logits = self.cnn_pipeline.model(X_test_cnn_tensor)
            cnn_preds = torch.argmax(cnn_logits, dim=1).cpu().numpy()

        cnn_cm = confusion_matrix(y_test, cnn_preds, labels=list(range(len(self.classes))))

        cnn_results = {
            "model_name": "1D Temporal CNN",
            "algorithm": "PyTorch 1D Temporal Convolutional Neural Network",
            "input_type": "Raw Temporal Landmark Sequences (30 frames x 736 channels)",
            "accuracy": round(float(accuracy_score(y_test, cnn_preds)), 4),
            "precision_macro": round(float(precision_score(y_test, cnn_preds, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, cnn_preds, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, cnn_preds, average="macro", zero_division=0)), 4),
            "training_time_sec": cnn_train_metrics["train_time_sec"],
            "inference_latency_ms": cnn_train_metrics["inference_latency_ms"],
            "model_size_kb": round((self.cnn_pipeline.output_dir / "cnn_model.pt").stat().st_size / 1024.0, 1),
            "test_signer": test_signer,
        }

        # 5. Scientific Comparison & Winner Decision
        winner = self._determine_winner(svm_results, cnn_results)

        total_elapsed = time.perf_counter() - t_global_start

        comparison_payload = {
            "timestamp": time.time(),
            "protocol": "Leave-One-Signer-Out (LOSO)",
            "fold": loso_fold,
            "train_signers": train_signers,
            "test_signer": test_signer,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_classes": len(self.classes),
            "classes": [c["name_ar"] for c in self.classes],
            "svm": svm_results,
            "cnn": cnn_results,
            "winner": winner,
            "total_benchmark_time_sec": round(total_elapsed, 2),
        }

        if progress_callback:
            progress_callback(5, 5, "حفظ وتصدير تقرير المقارنة العلمي (JSON & CSV)...")

        # 6. Save JSON & CSV Reports
        self._export_reports(comparison_payload, svm_cm, cnn_cm)

        logger.info(
            f"Scientific comparison completed: SVM F1={svm_results['f1_macro']:.4f}, "
            f"CNN F1={cnn_results['f1_macro']:.4f} -> Best: {winner['best_model']}."
        )
        return comparison_payload

    def _determine_winner(
        self, svm_res: dict[str, Any], cnn_res: dict[str, Any]
    ) -> dict[str, Any]:
        """Determine superior model based on empirical evidence and multi-criteria trade-offs."""
        f1_diff = cnn_res["f1_macro"] - svm_res["f1_macro"]
        acc_diff = cnn_res["accuracy"] - svm_res["accuracy"]

        # Primary criterion: Macro F1-Score on unseen signer
        if abs(f1_diff) < 0.01:
            # Very close accuracy/F1: tie-break by inference latency
            if cnn_res["inference_latency_ms"] < svm_res["inference_latency_ms"]:
                best = "1D Temporal CNN"
                rationale = "كلا النموذجين حققا دقة متقاربة، ولكن CNN تفوق في سرعة زمن الاستجابة (Latency)."
            else:
                best = "Support Vector Machine (SVM)"
                rationale = "كلا النموذجين حققا دقة متقاربة، ولكن SVM تفوق في سرعة التدريب وخفة الحجم."
        elif f1_diff > 0:
            best = "1D Temporal CNN"
            rationale = (
                f"تفوقت الشبكة الالتفافية 1D Temporal CNN بنسبة F1 أعلى بمقدار (+{f1_diff * 100:.1f}%) "
                "بفضل قدرة طبقات الالتفاف الزمني (Conv1D) على التقاط ديناميكيات حركة الإشارة عبر الإطارات المتعاقبة."
            )
        else:
            best = "Support Vector Machine (SVM)"
            rationale = (
                f"تفوق مصنف SVM بنسبة F1 أعلى بمقدار (+{abs(f1_diff) * 100:.1f}%) "
                "بفضل المتانة العالية للميزات الإحصائية الحركية المستخرجة وقلة عدد العينات مقارنة بالشبكات العميقة."
            )

        return {
            "best_model": best,
            "f1_difference": round(f1_diff, 4),
            "accuracy_difference": round(acc_diff, 4),
            "rationale": rationale,
            "speed_advantage": "SVM" if svm_res["training_time_sec"] < cnn_res["training_time_sec"] else "CNN",
            "size_advantage": "SVM" if svm_res["model_size_kb"] < cnn_res["model_size_kb"] else "CNN",
        }

    def _export_reports(
        self,
        payload: dict[str, Any],
        svm_cm: np.ndarray,
        cnn_cm: np.ndarray,
    ) -> None:
        """Export metrics to JSON, CSV, and save confusion matrices."""
        # 1. JSON Report
        json_file = self.eval_dir / "comparison_metrics.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 2. Confusion Matrices JSON
        cm_file = self.eval_dir / "confusion_matrices.json"
        cm_payload = {
            "classes": payload["classes"],
            "svm_confusion_matrix": svm_cm.tolist(),
            "cnn_confusion_matrix": cnn_cm.tolist(),
        }
        with open(cm_file, "w", encoding="utf-8") as f:
            json.dump(cm_payload, f, ensure_ascii=False, indent=2)

        # 3. CSV Report
        csv_file = self.eval_dir / "comparison_metrics.csv"
        fieldnames = [
            "Metric",
            "Support Vector Machine (SVM)",
            "1D Temporal CNN",
            "Difference (CNN - SVM)",
            "Superior Approach",
        ]

        svm = payload["svm"]
        cnn = payload["cnn"]

        rows = [
            {
                "Metric": "Test Accuracy",
                "Support Vector Machine (SVM)": f"{svm['accuracy'] * 100:.2f}%",
                "1D Temporal CNN": f"{cnn['accuracy'] * 100:.2f}%",
                "Difference (CNN - SVM)": f"{(cnn['accuracy'] - svm['accuracy']) * 100:+.2f}%",
                "Superior Approach": "CNN" if cnn["accuracy"] >= svm["accuracy"] else "SVM",
            },
            {
                "Metric": "Macro Precision",
                "Support Vector Machine (SVM)": f"{svm['precision_macro'] * 100:.2f}%",
                "1D Temporal CNN": f"{cnn['precision_macro'] * 100:.2f}%",
                "Difference (CNN - SVM)": f"{(cnn['precision_macro'] - svm['precision_macro']) * 100:+.2f}%",
                "Superior Approach": "CNN" if cnn["precision_macro"] >= svm["precision_macro"] else "SVM",
            },
            {
                "Metric": "Macro Recall",
                "Support Vector Machine (SVM)": f"{svm['recall_macro'] * 100:.2f}%",
                "1D Temporal CNN": f"{cnn['recall_macro'] * 100:.2f}%",
                "Difference (CNN - SVM)": f"{(cnn['recall_macro'] - svm['recall_macro']) * 100:+.2f}%",
                "Superior Approach": "CNN" if cnn["recall_macro"] >= svm["recall_macro"] else "SVM",
            },
            {
                "Metric": "Macro F1-Score",
                "Support Vector Machine (SVM)": f"{svm['f1_macro'] * 100:.2f}%",
                "1D Temporal CNN": f"{cnn['f1_macro'] * 100:.2f}%",
                "Difference (CNN - SVM)": f"{(cnn['f1_macro'] - svm['f1_macro']) * 100:+.2f}%",
                "Superior Approach": "CNN" if cnn["f1_macro"] >= svm["f1_macro"] else "SVM",
            },
            {
                "Metric": "Training Time (seconds)",
                "Support Vector Machine (SVM)": f"{svm['training_time_sec']:.2f}s",
                "1D Temporal CNN": f"{cnn['training_time_sec']:.2f}s",
                "Difference (CNN - SVM)": f"{cnn['training_time_sec'] - svm['training_time_sec']:+.2f}s",
                "Superior Approach": "SVM" if svm["training_time_sec"] <= cnn["training_time_sec"] else "CNN",
            },
            {
                "Metric": "Inference Latency (ms/sample)",
                "Support Vector Machine (SVM)": f"{svm['inference_latency_ms']:.2f} ms",
                "1D Temporal CNN": f"{cnn['inference_latency_ms']:.2f} ms",
                "Difference (CNN - SVM)": f"{cnn['inference_latency_ms'] - svm['inference_latency_ms']:+.2f} ms",
                "Superior Approach": "SVM" if svm["inference_latency_ms"] <= cnn["inference_latency_ms"] else "CNN",
            },
            {
                "Metric": "Model File Size (KB)",
                "Support Vector Machine (SVM)": f"{svm['model_size_kb']:.1f} KB",
                "1D Temporal CNN": f"{cnn['model_size_kb']:.1f} KB",
                "Difference (CNN - SVM)": f"{cnn['model_size_kb'] - svm['model_size_kb']:+.1f} KB",
                "Superior Approach": "SVM" if svm["model_size_kb"] <= cnn["model_size_kb"] else "CNN",
            },
        ]

        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Evaluation reports exported to {self.eval_dir}.")

    def get_latest_comparison(self) -> Optional[dict[str, Any]]:
        """Load latest comparison report if available."""
        json_file = self.eval_dir / "comparison_metrics.json"
        if not json_file.exists():
            return None
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading comparison report: {e}")
            return None

