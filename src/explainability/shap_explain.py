"""
Phase 6 — SHAP Explanations
==============================
Global feature attribution across the dataset using DeepSHAP / KernelSHAP.
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, EXPLAIN_DIR, NUM_VIEWS, USE_AMP,
    IMAGENET_MEAN, IMAGENET_STD,
)
from src.explainability.gradcam import denormalize

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


class MultiViewSHAP:
    """
    SHAP-based explanations for view-level attribution.

    Uses a simplified wrapper: treats the 6-view average feature as input
    to a linear attribution model, measuring each view's contribution.
    """

    def __init__(self, model: torch.nn.Module, num_views: int = NUM_VIEWS):
        self.model = model
        self.model.eval()
        self.num_views = num_views

    def compute_view_shapley_values(
        self,
        views: torch.Tensor,
        num_background: int = 50,
        background_loader=None,
    ) -> np.ndarray:
        """
        Compute Shapley values for each view's contribution to the prediction.

        Uses permutation-based approach: for each subset of views, measure
        how adding/removing each view changes the prediction confidence.

        Parameters
        ----------
        views : (1, V, 3, H, W) - single sample
        num_background : int - number of background samples

        Returns
        -------
        shapley_values : (V,) — attribution for each view
        """
        from itertools import combinations

        V = views.shape[1]
        shapley_values = np.zeros(V)

        # Baseline prediction with all views zeroed
        with torch.no_grad():
            views_device = views.to(DEVICE)
            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=USE_AMP):
                logits_full = self.model(views_device)
            prob_full = F.softmax(logits_full, dim=1)[0].cpu().numpy()
            pred_class = logits_full.argmax(dim=1).item()

        # For each view, compute marginal contribution across all subsets
        for view_idx in range(V):
            contributions = []
            other_views = [i for i in range(V) if i != view_idx]

            for subset_size in range(V):
                for subset in combinations(other_views, subset_size):
                    subset = set(subset)

                    # Prediction without view_idx
                    views_without = views.clone()
                    for v in range(V):
                        if v not in subset:
                            views_without[0, v] = 0.0
                    with torch.no_grad():
                        vw = views_without.to(DEVICE)
                        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                            logits_without = self.model(vw)
                        prob_without = F.softmax(logits_without, dim=1)[0, pred_class].cpu().item()

                    # Prediction with view_idx
                    views_with = views_without.clone()
                    views_with[0, view_idx] = views[0, view_idx]
                    with torch.no_grad():
                        vwi = views_with.to(DEVICE)
                        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                            logits_with = self.model(vwi)
                        prob_with = F.softmax(logits_with, dim=1)[0, pred_class].cpu().item()

                    contributions.append(prob_with - prob_without)

            shapley_values[view_idx] = np.mean(contributions)

        return shapley_values

    def compute_dataset_shapley(
        self,
        data_loader,
        max_samples: int = 50,
    ) -> np.ndarray:
        """
        Average Shapley values across multiple samples for global attribution.

        Returns
        -------
        avg_shapley : (V,) — mean view-level attribution
        """
        all_shapley = []
        count = 0

        for views, labels in data_loader:
            for i in range(views.shape[0]):
                if count >= max_samples:
                    break
                single = views[i:i+1]
                sv = self.compute_view_shapley_values(single)
                all_shapley.append(sv)
                count += 1
                if count % 10 == 0:
                    logger.info(f"  SHAP progress: {count}/{max_samples}")
            if count >= max_samples:
                break

        avg_shapley = np.mean(all_shapley, axis=0)
        logger.info(f"  Average SHAP values: {avg_shapley}")
        return avg_shapley


def visualize_shap_bar(
    shapley_values: np.ndarray,
    note_id: str = "global",
    save_dir: Optional[Path] = None,
) -> Path:
    """Bar plot of view-level Shapley values."""
    if save_dir is None:
        save_dir = EXPLAIN_DIR / "shap"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    views = [f"View {i+1}" for i in range(len(shapley_values))]
    colors = ["#ff6b6b" if v < 0 else "#4ecdc4" for v in shapley_values]
    ax.barh(views, shapley_values, color=colors)
    ax.set_xlabel("SHAP Value (contribution to prediction)")
    ax.set_title(f"View-Level SHAP Attribution — {note_id}")
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()

    path = save_dir / f"shap_{note_id}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  SHAP visualization -> {path}")
    return path


def visualize_dataset_shap(
    avg_shapley: np.ndarray,
    save_dir: Optional[Path] = None,
) -> Path:
    """Global dataset-level SHAP bar chart."""
    return visualize_shap_bar(avg_shapley, "dataset_global", save_dir)
