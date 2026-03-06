"""
Training Engine
================
Full training loop with:
  - Mixed precision (AMP) for RTX 4060
  - Early stopping
  - Best model checkpointing
  - LR scheduling
  - TensorBoard logging
  - Deterministic training
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, USE_AMP, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
    CHECKPOINT_DIR, LOG_DIR, seed_everything, SEED,
)

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE,
                 mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    General-purpose trainer for multi-view banknote models.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    experiment_name : str
        Used for checkpoint and log naming.
    num_epochs : int
    lr, weight_decay : float
    device : torch.device
    use_amp : bool
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_name: str = "experiment",
        num_epochs: int = NUM_EPOCHS,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        device: torch.device = DEVICE,
        use_amp: bool = USE_AMP,
    ):
        seed_everything()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=SCHEDULER_PATIENCE,
            factor=SCHEDULER_FACTOR,
        )

        # AMP scaler
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        self.early_stopper = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE, mode="max",
        )

        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=str(LOG_DIR / experiment_name)
        )

        # State tracking
        self.best_val_acc = 0.0
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # TRAINING EPOCH
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> tuple:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (views, labels) in enumerate(self.train_loader):
            views = views.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(views)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # ------------------------------------------------------------------
    # VALIDATION EPOCH
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self) -> tuple:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for views, labels in self.val_loader:
            views = views.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type="cuda", enabled=self.use_amp):
                logits = self.model(views)
                loss = self.criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # ------------------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_acc: float,
                         tag: str = "best"):
        path = CHECKPOINT_DIR / f"{self.experiment_name}_{tag}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "history": self.history,
        }, path)
        logger.info(f"  Checkpoint saved -> {path}")

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            path = CHECKPOINT_DIR / f"{self.experiment_name}_best.pth"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"  Loaded checkpoint from {path} "
                     f"(val_acc={ckpt['val_acc']:.4f})")
        return ckpt

    # ------------------------------------------------------------------
    # FULL TRAINING LOOP
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, list]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {self.experiment_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  AMP: {self.use_amp}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"{'='*60}\n")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._val_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            self.writer.add_scalars("Loss", {
                "train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {
                "train": train_acc, "val": val_acc}, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            logger.info(
                f"Epoch {epoch:>3}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s"
            )

            # Best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc, tag="best")

            # Periodic checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_acc, tag=f"epoch{epoch}")

            # LR scheduler
            self.scheduler.step(val_acc)

            # Early stopping
            if self.early_stopper(val_acc):
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        self.writer.close()
        logger.info(f"\nBest validation accuracy: {self.best_val_acc:.4f}")
        return self.history
