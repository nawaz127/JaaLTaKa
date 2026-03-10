"""
Script: Train Attention-Based MultiViewAttentionNet (Phase 4)
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).resolve().parent.parent
                            / "outputs" / "logs" / "train_attention.log"),
    ]
)

import argparse
from src.config import seed_everything, BATCH_SIZE, NUM_EPOCHS, DEVICE
from src.dataset.dataloader import build_dataloaders
from src.models.attention import MultiViewAttentionNet
from src.training.trainer import Trainer
from src.evaluation.metrics import full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train MultiView Attention Model")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "mobilenet_v2", "efficientnet_b0"],
                        help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    seed_everything()

    print(f"Device: {DEVICE}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {args.epochs}")

    # Build data
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
    )
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # Build model
    model = MultiViewAttentionNet(backbone_name=args.backbone, pretrained=True, freeze_backbone=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    exp_name = f"attention_{args.backbone}"

    # Train
    trainer = Trainer(
        model, train_loader, val_loader,
        experiment_name=exp_name,
        num_epochs=args.epochs,
    )
    history = trainer.train()

    # Load best checkpoint and evaluate
    trainer.load_checkpoint()
    metrics = full_evaluation(
        model, test_loader, exp_name, history
    )

    print("\n=== Final Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:>12}: {v:.4f}")


if __name__ == "__main__":
    main()
