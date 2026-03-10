"""
Script: Run Evaluation Only
===========================
Loads an existing checkpoint and runs the full evaluation pipeline
to generate metrics, plots, and misclassified samples log.
"""

import sys
import argparse
import logging
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import seed_everything, DEVICE, OUTPUT_DIR, BATCH_SIZE
from src.dataset.dataloader import build_dataloaders
from src.models.baseline import MultiViewBaseline
from src.models.attention import MultiViewAttentionNet
from src.evaluation.metrics import full_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on a trained model")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "mobilenet_v2", "efficientnet_b0"],
                        help="Backbone architecture")
    parser.add_argument("--type", type=str, default="baseline",
                        choices=["baseline", "attention"],
                        help="Model type")
    args = parser.parse_args()

    seed_everything()
    exp_name = f"{args.type}_{args.backbone}"
    
    # 1. Load Model
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

    # 2. Load Data
    _, _, test_loader = build_dataloaders(batch_size=BATCH_SIZE)

    # 3. Run Full Evaluation
    # This will generate the missing _misclassified.csv file
    full_evaluation(model, test_loader, exp_name)
    
    logger.info(f"\nEvaluation complete for {exp_name}.")

if __name__ == "__main__":
    main()
