"""
Phase 2 — Multi-View Data Pipeline
====================================
PyTorch Dataset and DataLoader that returns batches shaped (B, 6, 3, H, W).
Synchronized transforms across all six views of a banknote.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    COLOR_JITTER, ROTATION_DEGREES, RANDOM_FLIP_P,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, NUM_VIEWS,
    SPLITS_DIR, SEED, seed_everything,
)


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_train_transforms() -> transforms.Compose:
    """Training augmentations (applied identically to all 6 views)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=RANDOM_FLIP_P),
        transforms.RandomRotation(degrees=ROTATION_DEGREES),
        transforms.ColorJitter(**COLOR_JITTER),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Validation / test transforms (deterministic)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================================
# DATASET
# ============================================================================

class MultiViewBanknoteDataset(Dataset):
    """
    Returns (views_tensor, label) where views_tensor has shape (6, 3, H, W).

    Parameters
    ----------
    csv_path : Path or str
        CSV with columns note_id, note_path, img_1..img_6, label, split.
    transform : torchvision.transforms.Compose, optional
        Transform applied to each PIL image.
    num_views : int
        Number of views to load (for ablation; default=6).
    view_indices : list[int], optional
        Specific view indices (1-based) to load. Overrides num_views.
    view_dropout : float
        Probability of zeroing out a random view during training (0 = off).
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[transforms.Compose] = None,
        num_views: int = NUM_VIEWS,
        view_indices: Optional[List[int]] = None,
        view_dropout: float = 0.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.view_dropout = view_dropout

        # Determine which views to load
        if view_indices is not None:
            self.view_cols = [f"img_{i}" for i in view_indices]
        else:
            self.view_cols = [f"img_{i}" for i in range(1, num_views + 1)]
        self.target_num_views = NUM_VIEWS  # Always stack to 6 for model

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = int(row["label"])

        # For synchronized random transforms: fix the seed per sample access
        # so all views get the same random augmentation
        rng_seed = torch.randint(0, 2**31, (1,)).item()

        views = []
        for col in self.view_cols:
            img_path = row[col]
            img = Image.open(img_path).convert("RGB")

            if self.transform is not None:
                # Synchronize random state so all views get same augmentation
                torch.manual_seed(rng_seed)
                np.random.seed(rng_seed % (2**31))
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)

            views.append(img)

        # If fewer views loaded (ablation), pad with zeros to keep shape
        while len(views) < self.target_num_views:
            views.append(torch.zeros_like(views[0]))

        # Stack → (6, 3, H, W)
        views_tensor = torch.stack(views, dim=0)

        # View dropout regularization during training
        if self.view_dropout > 0 and self.transform is not None:
            mask = torch.bernoulli(
                torch.full((len(self.view_cols),), 1.0 - self.view_dropout)
            )
            # Ensure at least one view survives
            if mask.sum() == 0:
                mask[torch.randint(0, len(self.view_cols), (1,))] = 1.0
            for i in range(len(self.view_cols)):
                if mask[i] == 0:
                    views_tensor[i] = 0.0

        return views_tensor, label


# ============================================================================
# DATALOADER FACTORY
# ============================================================================

def _seed_worker(worker_id):
    """Worker init function — must be at module level for Windows pickling."""
    np.random.seed(SEED + worker_id)


def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_views: int = NUM_VIEWS,
    view_indices: Optional[List[int]] = None,
    view_dropout: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from saved CSV splits.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    seed_everything()

    train_ds = MultiViewBanknoteDataset(
        csv_path=SPLITS_DIR / "train.csv",
        transform=get_train_transforms(),
        num_views=num_views,
        view_indices=view_indices,
        view_dropout=view_dropout,
    )
    val_ds = MultiViewBanknoteDataset(
        csv_path=SPLITS_DIR / "val.csv",
        transform=get_eval_transforms(),
        num_views=num_views,
        view_indices=view_indices,
    )
    test_ds = MultiViewBanknoteDataset(
        csv_path=SPLITS_DIR / "test.csv",
        transform=get_eval_transforms(),
        num_views=num_views,
        view_indices=view_indices,
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    common = dict(
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **common,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **common,
    )

    return train_loader, val_loader, test_loader
