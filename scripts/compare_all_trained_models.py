"""
Script: Compare All Trained Models
===================================
Evaluates previously trained baseline models and the proposed model
on the test set to produce a final comparison report.

Models:
1. ResNet50 + Mean Pooling
2. MobileNetV2 + Mean Pooling
3. EfficientNet-B0 + Mean Pooling
4. Proposed Attention Transformer (ResNet50 backbone)
"""

import sys
import logging
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    seed_everything, BATCH_SIZE, DEVICE, OUTPUT_DIR, FIGURE_DIR
)
from src.dataset.dataloader import build_dataloaders
from src.models.baseline import MultiViewBaseline
from src.models.attention import MultiViewAttentionNet
from src.evaluation.metrics import full_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Output directory for the final comparison
FINAL_REPORT_DIR = OUTPUT_DIR / "ablation"
FINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model(name, model, checkpoint_path, test_loader):
    logger.info(f"\nEvaluating {name}...")
    
    if not checkpoint_path.exists():
        logger.warning(f"  Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    # Handle both full checkpoint dict and state_dict only
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    # Run evaluation
    metrics = full_evaluation(model, test_loader, f"final_{name}")
    metrics["model"] = name
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    metrics["total_params"] = total_params
    
    return metrics

def main():
    seed_everything()
    
    # Load data
    _, _, test_loader = build_dataloaders(batch_size=BATCH_SIZE)
    
    # Define models to evaluate
    model_configs = [
        {
            "name": "ResNet50_Baseline",
            "model": MultiViewBaseline(backbone_name="resnet50"),
            "ckpt": OUTPUT_DIR / "checkpoints" / "baseline_resnet50_best.pth"
        },
        {
            "name": "MobileNetV2_Baseline",
            "model": MultiViewBaseline(backbone_name="mobilenet_v2"),
            "ckpt": OUTPUT_DIR / "checkpoints" / "baseline_mobilenet_v2_best.pth"
        },
        {
            "name": "EfficientNetB0_Baseline",
            "model": MultiViewBaseline(backbone_name="efficientnet_b0"),
            "ckpt": OUTPUT_DIR / "checkpoints" / "baseline_efficientnet_b0_best.pth"
        },
        {
            "name": "Proposed_Attention_Net",
            "model": MultiViewAttentionNet(backbone_name="resnet50"),
            "ckpt": OUTPUT_DIR / "checkpoints" / "attention_transformer_best.pth"
        }
    ]
    
    results = []
    for cfg in model_configs:
        metrics = evaluate_model(cfg["name"], cfg["model"], cfg["ckpt"], test_loader)
        if metrics:
            results.append(metrics)
            
    if not results:
        logger.error("No results generated. Check if checkpoints exist.")
        return

    # Create Summary Table
    df = pd.DataFrame(results)
    # Reorder columns for readability
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "total_params"]
    df = df[cols]
    
    csv_path = FINAL_REPORT_DIR / "final_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nFinal Comparison Table:\n{df.to_string(index=False)}")
    logger.info(f"\nSaved table to: {csv_path}")
    
    # Generate Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    x = np.arange(len(df))
    width = 0.15
    
    for i, m in enumerate(metrics_to_plot):
        ax.bar(x + i*width, df[m], width, label=m.upper())
        
    ax.set_title("Final Model Comparison (Test Set Performance)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylabel("Score")
    ax.set_ylim(0.8, 1.05) # Zoom in on the top range
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = FIGURE_DIR / "final_comparison_bar.png"
    fig.savefig(plot_path, dpi=300)
    logger.info(f"Saved comparison chart to: {plot_path}")

if __name__ == "__main__":
    main()
