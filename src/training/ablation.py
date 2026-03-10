"""
Phase 5 — Ablation Studies
============================
View ablation, view dropout regularization, and model comparison.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, SEED, VIEW_ABLATION_CONFIGS, VIEW_DROPOUT_RATE,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    OUTPUT_DIR, FIGURE_DIR, seed_everything,
)
from src.dataset.dataloader import build_dataloaders
from src.models.baseline import MultiViewBaseline
from src.models.attention import MultiViewAttentionNet
from src.training.trainer import Trainer
from src.evaluation.metrics import full_evaluation

logger = logging.getLogger(__name__)

ABLATION_DIR = OUTPUT_DIR / "ablation"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# VIEW ABLATION
# ============================================================================

def run_view_ablation(
    num_views_list: List[int] = VIEW_ABLATION_CONFIGS,
    model_class: str = "baseline",
    num_epochs: int = NUM_EPOCHS,
) -> pd.DataFrame:
    """
    Train and evaluate with different numbers of views (1, 3, 6).

    For partial views, take the first N views.
    """
    seed_everything()
    results = []

    for nv in num_views_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"VIEW ABLATION: {nv} view(s) | model={model_class}")
        logger.info(f"{'='*60}")

        view_indices = list(range(1, nv + 1))
        train_loader, val_loader, test_loader = build_dataloaders(
            batch_size=BATCH_SIZE, view_indices=view_indices,
        )

        if model_class == "baseline":
            model = MultiViewBaseline(pretrained=True)
        else:
            model = MultiViewAttentionNet(pretrained=True)

        exp_name = f"ablation_{model_class}_{nv}views"
        trainer = Trainer(
            model, train_loader, val_loader,
            experiment_name=exp_name,
            num_epochs=num_epochs,
        )
        history = trainer.train()
        trainer.load_checkpoint()

        metrics = full_evaluation(model, test_loader, exp_name, history)
        metrics["num_views"] = nv
        metrics["model"] = model_class
        results.append(metrics)

        # Free GPU memory
        del model, trainer
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(ABLATION_DIR / "view_ablation.csv", index=False)
    logger.info(f"\nView ablation results:\n{df.to_string()}")
    return df


# ============================================================================
# VIEW DROPOUT ABLATION
# ============================================================================

def run_view_dropout_ablation(
    dropout_rates: List[float] = [0.0, 0.1, 0.15, 0.25],
    num_epochs: int = NUM_EPOCHS,
) -> pd.DataFrame:
    """Test different view dropout regularization strengths."""
    seed_everything()
    results = []

    for vd in dropout_rates:
        logger.info(f"\nVIEW DROPOUT ABLATION: rate={vd}")

        train_loader, val_loader, test_loader = build_dataloaders(
            batch_size=BATCH_SIZE, view_dropout=vd,
        )

        model = MultiViewAttentionNet(pretrained=True)
        exp_name = f"ablation_viewdropout_{vd:.2f}"
        trainer = Trainer(
            model, train_loader, val_loader,
            experiment_name=exp_name,
            num_epochs=num_epochs,
        )
        history = trainer.train()
        trainer.load_checkpoint()

        metrics = full_evaluation(model, test_loader, exp_name, history)
        metrics["view_dropout"] = vd
        results.append(metrics)

        del model, trainer
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(ABLATION_DIR / "view_dropout_ablation.csv", index=False)
    logger.info(f"\nView dropout ablation:\n{df.to_string()}")
    return df


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def run_model_comparison(num_epochs: int = NUM_EPOCHS) -> pd.DataFrame:
    """Compare different baselines vs Attention model."""
    seed_everything()
    results = []

    model_configs = [
        ("ResNet50_MeanPool", lambda: MultiViewBaseline(backbone_name="resnet50", pretrained=True)),
        ("MobileNetV2_MeanPool", lambda: MultiViewBaseline(backbone_name="mobilenet_v2", pretrained=True)),
        ("EfficientNetB0_MeanPool", lambda: MultiViewBaseline(backbone_name="efficientnet_b0", pretrained=True)),
        ("Proposed_Attention_Transformer", lambda: MultiViewAttentionNet(backbone_name="resnet50", pretrained=True)),
    ]

    for name, model_fn in model_configs:
        logger.info(f"\nMODEL COMPARISON: {name}")

        train_loader, val_loader, test_loader = build_dataloaders(
            batch_size=BATCH_SIZE,
        )

        model = model_fn()
        exp_name = f"comparison_{name}"
        trainer = Trainer(
            model, train_loader, val_loader,
            experiment_name=exp_name,
            num_epochs=num_epochs,
        )
        history = trainer.train()
        trainer.load_checkpoint()

        metrics = full_evaluation(model, test_loader, exp_name, history)
        metrics["model"] = name

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        metrics["total_params"] = total_params
        metrics["trainable_params"] = trainable_params

        results.append(metrics)

        del model, trainer
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(ABLATION_DIR / "model_comparison.csv", index=False)
    logger.info(f"\nModel comparison:\n{df.to_string()}")
    return df


# ============================================================================
# GENERATE ABLATION SUMMARY TABLE
# ============================================================================

def generate_ablation_summary():
    """Load all ablation results and produce a combined LaTeX-ready table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables = {}
    for csv_file in ABLATION_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)
        tables[csv_file.stem] = df

    # Combined figure
    if "model_comparison" in tables:
        df = tables["model_comparison"]
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        x = range(len(df))
        width = 0.15
        for i, col in enumerate(metrics_cols):
            ax.bar([xi + i * width for xi in x], df[col],
                   width=width, label=col.upper())
        ax.set_xticks([xi + 2 * width for xi in x])
        ax.set_xticklabels(df["model"])
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        fig.savefig(FIGURE_DIR / "model_comparison_bar.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    return tables
