"""
Script: Train Baseline MultiViewResNet (Phase 3)
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
                            / "outputs" / "logs" / "train_baseline.log"),
    ]
)

from src.config import seed_everything, BATCH_SIZE, NUM_EPOCHS, DEVICE
from src.dataset.dataloader import build_dataloaders
from src.models.baseline import MultiViewResNet
from src.training.trainer import Trainer
from src.evaluation.metrics import full_evaluation


def main():
    seed_everything()

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {NUM_EPOCHS}")

    # Build data
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
    )
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # Build model
    model = MultiViewResNet(pretrained=True, freeze_backbone=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # Train
    trainer = Trainer(
        model, train_loader, val_loader,
        experiment_name="baseline_resnet50",
        num_epochs=NUM_EPOCHS,
    )
    history = trainer.train()

    # Load best checkpoint and evaluate
    trainer.load_checkpoint()
    metrics = full_evaluation(model, test_loader, "baseline_resnet50", history)

    print("\n=== Final Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:>12}: {v:.4f}")


if __name__ == "__main__":
    main()
