"""Evaluation metrics, decision-threshold selection and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works in CI and Docker without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(y_true, y_prob) -> float:
    """Probability cut-off that maximises F1 on the given (validation) data.

    The default 0.5 cut-off is arbitrary. With ~26% churners, a lower threshold usually
    catches far more churners at an acceptable cost in false alarms.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall have one more entry than thresholds; drop the final (recall=0) point.
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix(y_true, y_prob, threshold: float, path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=["Stays", "Churns"], ax=ax, colorbar=False
    )
    ax.set_title(f"{title}\nthreshold={threshold:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
