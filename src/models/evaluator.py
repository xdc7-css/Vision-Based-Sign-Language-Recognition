"""ModelEvaluator — cross-validation, confusion matrices, and model comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ClassificationReport:
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    per_class_metrics: dict[str, ClassificationReport] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0


class ModelEvaluator:
    """Evaluates and compares multiple models. Generates comparison reports."""

    @staticmethod
    def cross_validate(model, X: np.ndarray, y: np.ndarray, folds: int = 5) -> list[float]:
        """Run k-fold cross-validation on sklearn-compatible models."""
        from sklearn.model_selection import cross_val_score
        try:
            scores = cross_val_score(model, X, y, cv=folds, scoring="accuracy", n_jobs=-1)
            return scores.tolist()
        except Exception:
            return []

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[list[int]]:
        """Compute confusion matrix."""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        return cm.tolist()

    @staticmethod
    def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, ClassificationReport]:
        """Compute per-class precision, recall, F1, support."""
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                        target_names=class_names, output_dict=True, zero_division=0)
        result = {}
        for name in class_names:
            if name in report:
                result[name] = ClassificationReport(
                    precision=report[name]["precision"],
                    recall=report[name]["recall"],
                    f1_score=report[name]["f1-score"],
                    support=int(report[name]["support"]),
                )
            else:
                result[name] = ClassificationReport(precision=0.0, recall=0.0, f1_score=0.0, support=0)
        return result

    @staticmethod
    def generate_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str],
        output_dir: str | Path,
        model_name: str = "model",
    ) -> dict:
        """Generate full classification report with JSON, CSV, and plot export."""
        from sklearn.metrics import classification_report, confusion_matrix
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                        target_names=class_names, output_dict=True, zero_division=0)

        # JSON report
        report_path = output_dir / f"{model_name}_classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Confusion matrix plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
            fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.6), max(8, len(class_names) * 0.6)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
            plt.tight_layout()
            cm_path = output_dir / f"{model_name}_confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close(fig)
        except ImportError:
            pass

        return report

    @staticmethod
    def compare_models(
        models: dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[list[str]] = None,
    ) -> dict[str, EvaluationResult]:
        """Evaluate multiple models and return comparison results."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        results: dict[str, EvaluationResult] = {}

        for name, model in models.items():
            y_pred = model.predict(X_test)
            cm = ModelEvaluator.confusion_matrix(y_test, y_pred, class_names or [])
            per_class = ModelEvaluator.per_class_metrics(y_test, y_pred, class_names or [])

            results[name] = EvaluationResult(
                model_name=name,
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
                recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
                f1_score=f1_score(y_test, y_pred, average="weighted", zero_division=0),
                confusion_matrix=cm,
                class_names=class_names or [],
                per_class_metrics=per_class,
            )

        return results

    @staticmethod
    def print_comparison(results: dict[str, EvaluationResult]) -> None:
        """Pretty-print model comparison results."""
        print(f"\n{'='*65}")
        print("  Model Comparison")
        print(f"{'='*65}")
        print(f"  {'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*52}")
        for name, r in sorted(results.items()):
            print(f"  {name:<12} {r.accuracy:>10.3f} {r.precision:>10.3f} {r.recall:>10.3f} {r.f1_score:>10.3f}")
        print(f"{'='*65}")

    @staticmethod
    def export_comparison(results: dict[str, EvaluationResult], path: str | Path) -> None:
        """Export comparison results to JSON."""
        data = {
            name: {
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1_score": r.f1_score,
                "confusion_matrix": r.confusion_matrix,
                "class_names": r.class_names,
                "per_class_metrics": {
                    cls: {"precision": m.precision, "recall": m.recall, "f1_score": m.f1_score, "support": m.support}
                    for cls, m in r.per_class_metrics.items()
                },
            }
            for name, r in results.items()
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class ModelRegistry:
    """Model metadata persistence — tracks available trained models."""

    def __init__(self, registry_path: str | Path = "data/models/manifest.json") -> None:
        self._path = Path(registry_path)
        self._registry: dict = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                self._registry = json.load(f)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register_model(self, dialect: str, model_type: str, path: str, metrics: dict) -> None:
        if dialect not in self._registry:
            self._registry[dialect] = {}
        self._registry[dialect][model_type] = {
            "path": path,
            "metrics": metrics,
            "version": "1.0.0",
        }
        self._save()

    def get_model(self, dialect: str, model_type: str) -> Optional[dict]:
        return self._registry.get(dialect, {}).get(model_type)

    def list_models(self, dialect: str) -> list[str]:
        return list(self._registry.get(dialect, {}).keys())

    @property
    def all_dialects(self) -> list[str]:
        return list(self._registry.keys())
