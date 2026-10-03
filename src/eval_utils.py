"""Shared evaluation helpers so every model reports metrics the same way."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from preprocessing import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "reports" / "results"
FIG_DIR = ROOT / "figures"


def evaluate(y_true, y_pred, model_name: str, save: bool = True) -> dict:
    """Compute macro-F1, accuracy, per-class report, confusion matrix; optionally persist."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    result = {
        "model": model_name,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": report["accuracy"],
        "per_class": {
            cls: {k: report[cls][k] for k in ("precision", "recall", "f1-score", "support")}
            for cls in CLASS_NAMES
        },
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n=== {model_name} ===")
    print(f"Macro-F1: {macro_f1:.4f}  Weighted-F1: {weighted_f1:.4f}  Accuracy: {report['accuracy']:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / f"{model_name}.json", "w") as f:
            json.dump(result, f, indent=2)
        plot_confusion_matrix(cm, model_name)

    return result


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}", pad=12)
    fig.tight_layout()
    fname = f"cm_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def plot_roc_pr(y_true, y_proba, model_name: str) -> float:
    """One-vs-rest ROC and Precision-Recall curves; y_proba shape (n_samples, n_classes)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    macro_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")

    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
        axes[0].plot(fpr, tpr, label=f"{cls} (AUC={auc_i:.2f})")

        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        axes[1].plot(rec, prec, label=f"{cls} (AUC={auc_i:.2f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC (one-vs-rest), macro-AUC={macro_auc:.3f}")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall (one-vs-rest)")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{model_name}", y=1.02)
    fig.tight_layout()
    fname = f"roc_pr_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return macro_auc


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
