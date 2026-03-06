"""
Phase 3 — Multi-View ResNet Baseline
======================================
Pretrained ResNet50 backbone with mean-pooling view fusion.

Architecture:
    Input (B, 6, 3, 224, 224)
  → Reshape to (B*6, 3, 224, 224)
  → ResNet50 backbone → (B*6, 2048)
  → Reshape to (B, 6, 2048)
  → Mean-pool over views → (B, 2048)
  → Classifier head → (B, 2)
"""

import torch
import torch.nn as nn
import torchvision.models as models

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    NUM_VIEWS, NUM_CLASSES, FEATURE_DIM, DROPOUT_RATE,
    CLASSIFIER_HIDDEN, BACKBONE,
)


class ClassifierHead(nn.Module):
    """Linear → ReLU → Dropout → Linear."""
    def __init__(self, in_features: int, hidden: int, num_classes: int,
                 dropout: float = DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiViewResNet(nn.Module):
    """
    Baseline multi-view model with ResNet50 backbone and mean pooling.

    Parameters
    ----------
    num_views : int
        Number of input views (default 6).
    num_classes : int
        Output classes (default 2).
    pretrained : bool
        Use ImageNet pretrained backbone.
    freeze_backbone : bool
        If True, freeze backbone weights (feature extraction only).
    """

    def __init__(
        self,
        num_views: int = NUM_VIEWS,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_views = num_views

        # Load backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove final FC layer — keep everything up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = FEATURE_DIM  # 2048 for ResNet50

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classifier head
        self.classifier = ClassifierHead(
            in_features=self.feature_dim,
            hidden=CLASSIFIER_HIDDEN,
            num_classes=num_classes,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-view features.

        Parameters
        ----------
        x : Tensor (B, V, 3, H, W)

        Returns
        -------
        features : Tensor (B, V, feature_dim)
        """
        B, V, C, H, W = x.shape
        # Merge batch and view dims
        x = x.view(B * V, C, H, W)             # (B*V, 3, H, W)
        features = self.backbone(x)              # (B*V, 2048, 1, 1)
        features = features.view(B * V, -1)      # (B*V, 2048)
        features = features.view(B, V, -1)       # (B, V, 2048)
        return features

    def fuse(self, features: torch.Tensor) -> torch.Tensor:
        """Mean pooling over views. (B, V, D) → (B, D)"""
        return features.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, V, 3, H, W)

        Returns
        -------
        logits : Tensor (B, num_classes)
        """
        features = self.extract_features(x)
        fused = self.fuse(features)
        logits = self.classifier(fused)
        return logits

    def get_view_features(self, x: torch.Tensor) -> torch.Tensor:
        """Public accessor for per-view features (used by explainability)."""
        return self.extract_features(x)
