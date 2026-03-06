"""
Phase 6 — LIME Explanations
==============================
Local Interpretable Model-agnostic Explanations for individual predictions.
Generates per-view superpixel-based explanations.
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    EXPLAIN_DIR, NUM_VIEWS, USE_AMP,
)
from src.explainability.gradcam import denormalize

logger = logging.getLogger(__name__)

try:
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not installed. Run: pip install lime")


class MultiViewLIME:
    """
    LIME explanations for each view of a multi-view banknote model.

    For each view, treats the multi-view model as a single-image
    classifier by fixing other views and perturbing only the target view.
    """

    def __init__(self, model: torch.nn.Module, num_views: int = NUM_VIEWS):
        self.model = model
        self.model.eval()
        self.num_views = num_views

    def _make_predict_fn(
        self, views: torch.Tensor, view_idx: int
    ):
        """
        Create a prediction function for LIME that only varies one view.

        Parameters
        ----------
        views : (1, V, 3, H, W) — the full multi-view input
        view_idx : int — which view to explain (0-based)
        """
        mean = np.array(IMAGENET_MEAN)
        std = np.array(IMAGENET_STD)

        def predict_fn(images: np.ndarray) -> np.ndarray:
            """
            images : (N, H, W, 3) — perturbed versions of one view
            Returns probabilities (N, num_classes).
            """
            batch_size = images.shape[0]
            # Normalize images
            imgs_norm = (images.astype(np.float32) / 255.0 - mean) / std
            imgs_tensor = torch.tensor(
                imgs_norm, dtype=torch.float32
            ).permute(0, 3, 1, 2)  # (N, 3, H, W)

            # Build full multi-view batch: repeat the original views
            full_views = views.repeat(batch_size, 1, 1, 1, 1).clone()
            full_views[:, view_idx] = imgs_tensor

            # Determine device
            device = next(self.model.parameters()).device
            device_type = "cuda" if device.type == "cuda" else "cpu"

            # Forward pass
            with torch.no_grad():
                full_views = full_views.to(device)
                from torch.amp import autocast
                with autocast(device_type=device_type, enabled=USE_AMP if device_type == "cuda" else False):
                    logits = self.model(full_views)
                probs = F.softmax(logits, dim=1).cpu().numpy()

            return probs

        return predict_fn

    def explain(
        self,
        views: torch.Tensor,
        view_idx: int = 0,
        num_samples: int = 500,
        num_features: int = 10,
        top_labels: int = 2,
    ):
        """
        Generate LIME explanation for a specific view.

        Parameters
        ----------
        views : (1, V, 3, H, W)
        view_idx : int

        Returns
        -------
        explanation : lime.lime_image.ImageExplanation
        """
        if not LIME_AVAILABLE:
            raise RuntimeError("LIME not installed")

        # Convert view to uint8 RGB for LIME
        img_tensor = views[0, view_idx]  # (3, H, W)
        img_np = denormalize(img_tensor)
        img_uint8 = (img_np * 255).astype(np.uint8)

        explainer = lime_image.LimeImageExplainer()
        predict_fn = self._make_predict_fn(views, view_idx)

        explanation = explainer.explain_instance(
            img_uint8,
            predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features,
        )

        return explanation

    def explain_all_views(
        self,
        views: torch.Tensor,
        num_samples: int = 500,
    ) -> List:
        """Generate LIME explanations for all views."""
        explanations = []
        for v in range(self.num_views):
            logger.info(f"  LIME explaining view {v+1}/{self.num_views}")
            exp = self.explain(views, view_idx=v, num_samples=num_samples)
            explanations.append(exp)
        return explanations


def visualize_lime(
    views: torch.Tensor,
    explanations: List,
    predicted_class: int,
    note_id: str = "unknown",
    save_dir: Optional[Path] = None,
    class_names: List[str] = ["Fake", "Real"],
) -> Path:
    """
    Visualize LIME explanations for all views.
    """
    if save_dir is None:
        save_dir = EXPLAIN_DIR / "lime"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_views = len(explanations)
    fig, axes = plt.subplots(2, n_views, figsize=(3 * n_views, 6))

    for v, exp in enumerate(explanations):
        img = denormalize(views[0, v])

        # Original
        axes[0, v].imshow(img)
        axes[0, v].set_title(f"View {v+1}", fontsize=10)
        axes[0, v].axis("off")

        # LIME overlay
        try:
            temp, mask = exp.get_image_and_mask(
                predicted_class,
                positive_only=False,
                num_features=5,
                hide_rest=False,
            )
            axes[1, v].imshow(temp / 255.0 if temp.max() > 1 else temp)
        except Exception:
            axes[1, v].imshow(img)
        axes[1, v].set_title(f"LIME {v+1}", fontsize=10)
        axes[1, v].axis("off")

    fig.suptitle(
        f"LIME — Note: {note_id} | Pred: {class_names[predicted_class]}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = save_dir / f"lime_{note_id}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  LIME visualization -> {path}")
    return path
