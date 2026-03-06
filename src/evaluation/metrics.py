"""
Evaluation Metrics
===================
Compute accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
Produces publication-ready results tables and figures.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DEVICE, USE_AMP, FIGURE_DIR, CLASS_NAMES

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> Dict[str, np.ndarray]:
    """Run model on loader and collect all predictions & labels."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for views, labels in loader:
        views = views.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(views)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

    return {
        "labels": np.concatenate(all_labels),
        "preds": np.concatenate(all_preds),
        "probs": np.concatenate(all_probs),
    }


def compute_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute all standard classification metrics."""
    y_true = results["labels"]
    y_pred = results["preds"]
    y_prob = results["probs"][:, 1]  # probability of 'real' class

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1": f1_score(y_true, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    logger.info("\n--- Classification Report ---")
    logger.info("\n" + classification_report(
        y_true, y_pred, target_names=CLASS_NAMES
    ))

    for k, v in metrics.items():
        logger.info(f"  {k:>12}: {v:.4f}")

    return metrics


def plot_confusion_matrix(
    results: Dict[str, np.ndarray],
    save_name: str = "confusion_matrix.png",
) -> Path:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(results["labels"], results["preds"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = FIGURE_DIR / save_name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Confusion matrix -> {path}")
    return path


def plot_roc_curve(
    results: Dict[str, np.ndarray],
    save_name: str = "roc_curve.png",
) -> Path:
    """Plot ROC curve."""
    y_true = results["labels"]
    y_prob = results["probs"][:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = FIGURE_DIR / save_name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ROC curve -> {path}")
    return path


def plot_training_curves(
    history: Dict[str, list],
    save_name: str = "training_curves.png",
) -> Path:
    """Plot loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURE_DIR / save_name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Training curves -> {path}")
    return path


def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    experiment_name: str = "model",
    history: Optional[Dict[str, list]] = None,
) -> Dict[str, float]:
    """Run full evaluation pipeline: metrics + all plots."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Full Evaluation: {experiment_name}")
    logger.info(f"{'='*60}")

    results = collect_predictions(model, test_loader)
    metrics = compute_metrics(results)

    plot_confusion_matrix(results, f"{experiment_name}_confusion.png")
    plot_roc_curve(results, f"{experiment_name}_roc.png")

    if history:
        plot_training_curves(history, f"{experiment_name}_curves.png")

    return metrics
