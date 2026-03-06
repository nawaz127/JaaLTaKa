"""
Phase 4 — Attention-Based Multi-View Model
=============================================
Replaces mean pooling with a Transformer encoder and attention pooling
for learnable view-importance weighting.

Architecture:
    Input (B, 6, 3, 224, 224)
  → ResNet50 backbone → (B, 6, 2048)
  → Linear projection → (B, 6, D_model)
  → + Learnable view positional embeddings
  → Transformer Encoder (L layers, H heads)
  → Attention pooling → (B, D_model)
  → Classifier head → (B, 2)

Memory-optimized for RTX 4060 (8 GB VRAM).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    NUM_VIEWS, NUM_CLASSES, FEATURE_DIM, DROPOUT_RATE,
    CLASSIFIER_HIDDEN, TRANSFORMER_HEADS, TRANSFORMER_LAYERS,
    TRANSFORMER_DIM, TRANSFORMER_DROPOUT, VIEW_EMBED_DIM,
)


# ============================================================================
# ATTENTION POOLING
# ============================================================================

class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over a sequence of view features.

    Computes scalar importance weight for each view and returns
    weighted sum.  Also exposes attention weights for interpretability.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, V, D)

        Returns
        -------
        pooled : (B, D)
        weights : (B, V)  — attention weights (sum to 1 over views)
        """
        scores = self.attention(x).squeeze(-1)   # (B, V)
        weights = F.softmax(scores, dim=1)        # (B, V)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return pooled, weights


# ============================================================================
# MULTI-VIEW ATTENTION MODEL
# ============================================================================

class MultiViewAttentionNet(nn.Module):
    """
    Advanced multi-view model with:
      - ResNet50 feature extractor (shared across views)
      - Linear feature projection
      - Learnable view positional embeddings
      - Transformer encoder for cross-view reasoning
      - Attention pooling for view fusion
      - Classifier head
    """

    def __init__(
        self,
        num_views: int = NUM_VIEWS,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        d_model: int = TRANSFORMER_DIM,
        nhead: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        dropout: float = TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.num_views = num_views
        self.d_model = d_model

        # ---- Backbone ----
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone_dim = FEATURE_DIM  # 2048

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ---- Feature projection ----
        self.projector = nn.Sequential(
            nn.Linear(self.backbone_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- Learnable view embeddings (positional) ----
        self.view_embeddings = nn.Parameter(
            torch.randn(1, num_views, d_model) * 0.02
        )

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # ---- Attention Pooling ----
        self.attention_pool = AttentionPooling(d_model)

        # ---- Classifier Head ----
        self.classifier = nn.Sequential(
            nn.Linear(d_model, CLASSIFIER_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(CLASSIFIER_HIDDEN, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projection and classifier layers."""
        for m in [self.projector, self.classifier]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, V, 3, H, W) → (B, V, backbone_dim)"""
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)
        feats = self.backbone(x).view(B * V, -1)   # (B*V, 2048)
        return feats.view(B, V, -1)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ):
        """
        Parameters
        ----------
        x : (B, V, 3, H, W)
        return_attention : bool
            If True, also return per-view attention weights.

        Returns
        -------
        logits : (B, num_classes)
        attn_weights : (B, V)  [optional]
        """
        # 1. Per-view feature extraction
        feats = self.extract_features(x)            # (B, V, 2048)

        # 2. Project to transformer dim
        feats = self.projector(feats)                # (B, V, d_model)

        # 3. Add learnable view embeddings
        feats = feats + self.view_embeddings[:, :feats.size(1), :]

        # 4. Transformer encoder (cross-view reasoning)
        feats = self.transformer(feats)              # (B, V, d_model)

        # 5. Attention pooling
        fused, attn_weights = self.attention_pool(feats)  # (B, d_model)

        # 6. Classification
        logits = self.classifier(fused)

        if return_attention:
            return logits, attn_weights
        return logits

    def get_view_features(self, x: torch.Tensor) -> torch.Tensor:
        """Raw per-view features before projection (for Grad-CAM)."""
        return self.extract_features(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return only attention weights for interpretability."""
        _, attn = self.forward(x, return_attention=True)
        return attn
