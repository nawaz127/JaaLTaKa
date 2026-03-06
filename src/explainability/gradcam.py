"""
Phase 6 — Grad-CAM Explainability
====================================
Generate class-discriminative activation maps for each view of a banknote.
Uses the last convolutional layer of the ResNet50 backbone.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    EXPLAIN_DIR, NUM_VIEWS,
)

logger = logging.getLogger(__name__)


class MultiViewGradCAM:
    """
    Grad-CAM for multi-view models.

    Hooks into the last convolutional layer of the shared backbone
    to capture feature maps and gradients for each view independently.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Parameters
        ----------
        model : nn.Module
            A multi-view model with a `backbone` attribute.
        target_layer : nn.Module, optional
            Specific layer to hook. Defaults to last Conv2d in backbone.
        """
        self.model = model
        self.model.eval()

        # Find target layer
        if target_layer is None:
            target_layer = self._find_last_conv(model.backbone)
        self.target_layer = target_layer

        # Storage for hooks
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _find_last_conv(self, module: nn.Module) -> nn.Module:
        """Recursively find the last Conv2d layer."""
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No Conv2d layer found in backbone")
        return last_conv

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        views: torch.Tensor,
        target_class: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[np.ndarray], int, float]:
        """
        Generate Grad-CAM heatmaps for each view.

        Parameters
        ----------
        views : Tensor (1, 6, 3, H, W) — single sample batch
        target_class : int, optional
            Class to explain. If None, uses predicted class.
        device : torch.device, optional
            Device to run on. If None, uses model's device.

        Returns
        -------
        heatmaps : list of np.ndarray (H, W) — one per view
        predicted_class : int
        confidence : float
        """
        self.model.eval()
        
        # Determine device
        if device is None:
            device = next(self.model.parameters()).device
            
        views = views.to(device)
        B, V, C, H, W = views.shape
        assert B == 1, "Grad-CAM operates on single samples"

        heatmaps = []

        for v_idx in range(V):
            self.model.zero_grad()

            # Create input with all views but enable grad only for target view
            single_view_batch = views.clone().requires_grad_(True)

            # Forward pass through full model
            logits = self.model(single_view_batch)
            probs = F.softmax(logits, dim=1)

            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            confidence = probs[0, target_class].item()

            # Backward pass for target class
            self.model.zero_grad()

            # We need per-view activations: reshape and forward single view
            view_input = views[0, v_idx].unsqueeze(0)  # (1, 3, H, W)

            # Forward through backbone only
            self.activations = None
            self.gradients = None

            view_input.requires_grad_(True)
            feat = self.model.backbone(view_input)  # triggers hooks

            # Compute grad w.r.t. target class through the full pipeline
            # Use feature → project → classify path
            feat_flat = feat.view(1, -1)

            # For baseline model: pass through classifier directly
            if hasattr(self.model, 'projector'):
                # Attention model
                feat_proj = self.model.projector(feat_flat.unsqueeze(1))
                feat_proj = feat_proj + self.model.view_embeddings[:, v_idx:v_idx+1, :]
                out = self.model.classifier(feat_proj.squeeze(1))
            else:
                # Baseline model
                out = self.model.classifier(feat_flat)

            score = out[0, target_class]
            score.backward()

            if self.gradients is None:
                heatmaps.append(np.zeros((H, W)))
                continue

            # Grad-CAM: global average pool gradients → weights
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
            cam = F.relu(cam)

            # Resize to input size
            cam = F.interpolate(cam, size=(H, W), mode="bilinear",
                                align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            heatmaps.append(cam)

        return heatmaps, target_class, confidence

    def cleanup(self):
        """Remove hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ============================================================================
# VISUALIZATION
# ============================================================================

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def visualize_gradcam(
    views: torch.Tensor,
    heatmaps: List[np.ndarray],
    predicted_class: int,
    confidence: float,
    note_id: str = "unknown",
    save_dir: Optional[Path] = None,
    class_names: List[str] = ["Fake", "Real"],
) -> Path:
    """
    Create a figure showing original views with Grad-CAM overlays.

    Parameters
    ----------
    views : (1, 6, 3, H, W)
    heatmaps : list of (H, W) arrays
    """
    if save_dir is None:
        save_dir = EXPLAIN_DIR / "gradcam"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_views = views.shape[1]
    fig, axes = plt.subplots(2, n_views, figsize=(3 * n_views, 6))

    for v in range(n_views):
        img = denormalize(views[0, v])

        # Original image
        axes[0, v].imshow(img)
        axes[0, v].set_title(f"View {v+1}", fontsize=10)
        axes[0, v].axis("off")

        # Grad-CAM overlay
        axes[1, v].imshow(img)
        axes[1, v].imshow(heatmaps[v], cmap="jet", alpha=0.4)
        axes[1, v].set_title(f"Grad-CAM {v+1}", fontsize=10)
        axes[1, v].axis("off")

    fig.suptitle(
        f"Note: {note_id} | Pred: {class_names[predicted_class]} "
        f"({confidence:.2%})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    path = save_dir / f"gradcam_{note_id}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Grad-CAM visualization -> {path}")
    return path


# ============================================================================
# EXPLANATION STABILITY
# ============================================================================

def evaluate_explanation_stability(
    model: nn.Module,
    views: torch.Tensor,
    n_perturbations: int = 5,
    noise_std: float = 0.01,
) -> float:
    """
    Measure Grad-CAM stability under small input perturbations.

    Returns average cosine similarity between original and perturbed heatmaps.
    """
    gradcam = MultiViewGradCAM(model)
    original_heatmaps, _, _ = gradcam.generate(views)

    similarities = []
    for _ in range(n_perturbations):
        noise = torch.randn_like(views) * noise_std
        perturbed = views + noise

        perturbed_heatmaps, _, _ = gradcam.generate(perturbed)

        for orig, pert in zip(original_heatmaps, perturbed_heatmaps):
            o_flat = orig.flatten()
            p_flat = pert.flatten()
            cos_sim = np.dot(o_flat, p_flat) / (
                np.linalg.norm(o_flat) * np.linalg.norm(p_flat) + 1e-8
            )
            similarities.append(cos_sim)

    gradcam.cleanup()
    avg_stability = np.mean(similarities)
    logger.info(f"  Grad-CAM stability: {avg_stability:.4f}")
    return avg_stability
