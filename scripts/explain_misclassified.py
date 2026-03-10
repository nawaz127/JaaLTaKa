"""
Script: Explain Misclassified Samples with Grad-CAM
=====================================================
Targeted explainability for samples that the model got wrong.
"""

import sys
import argparse
import logging
from pathlib import Path
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import seed_everything, DEVICE, OUTPUT_DIR, EXPLAIN_DIR, CLASS_NAMES
from src.dataset.dataloader import build_dataloaders
from src.models.baseline import MultiViewBaseline
from src.models.attention import MultiViewAttentionNet
from src.explainability.gradcam import MultiViewGradCAM, visualize_gradcam

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Explain misclassified samples")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "mobilenet_v2", "efficientnet_b0"],
                        help="Backbone architecture")
    parser.add_argument("--type", type=str, default="baseline",
                        choices=["baseline", "attention"],
                        help="Model type")
    args = parser.parse_args()

    seed_everything()
    
    # 1. Load misclassified list
    exp_name = f"{args.type}_{args.backbone}"
    csv_path = OUTPUT_DIR / "logs" / f"{exp_name}_misclassified.csv"
    
    if not csv_path.exists():
        logger.error(f"No misclassified list found at {csv_path}. Run training/evaluation first.")
        return

    mis_df = pd.read_csv(csv_path)
    logger.info(f"Found {len(mis_df)} misclassified samples for {exp_name}.")

    # 2. Load Model
    if args.type == "baseline":
        model = MultiViewBaseline(backbone_name=args.backbone)
        ckpt_name = f"baseline_{args.backbone}_best.pth"
    else:
        model = MultiViewAttentionNet(backbone_name=args.backbone)
        ckpt_name = "attention_transformer_best.pth" if args.backbone == "resnet50" else f"attention_{args.backbone}_best.pth"

    ckpt_path = OUTPUT_DIR / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    logger.info(f"Loading weights from {ckpt_name}...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 3. Load Data
    # We need the full test loader to get the images
    _, _, test_loader = build_dataloaders(batch_size=1) # batch 1 for easy access

    # 4. Run Grad-CAM
    gradcam = MultiViewGradCAM(model)
    save_dir = EXPLAIN_DIR / "misclassified" / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # We iterate through the loader to find the specific note IDs
    found_count = 0
    for views, labels in test_loader:
        # Get note_id from dataset at the current index
        # This is a bit slow but ensures we match the correct images
        dataset = test_loader.dataset
        # The loader is not shuffled, so we can use a counter
        note_id = dataset.df.iloc[found_count]["note_id"]
        
        if note_id in mis_df["note_id"].values:
            logger.info(f"\nExplaining misclassified note: {note_id}")
            
            # Generate heatmaps
            heatmaps, pred_class, conf = gradcam.generate(views, device=DEVICE)
            
            # Visualize
            visualize_gradcam(
                views, heatmaps, pred_class, conf,
                note_id=note_id,
                save_dir=save_dir,
                class_names=CLASS_NAMES
            )
            
        found_count += 1
        if found_count >= len(dataset):
            break

    gradcam.cleanup()
    logger.info(f"\nDone! Explanations saved to {save_dir}")

if __name__ == "__main__":
    main()
