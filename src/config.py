"""
Global Configuration for JaalTaka Project
==========================================
All hyperparameters, paths, and settings in one place for reproducibility.
"""

import os
import torch
from pathlib import Path


# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT  # JaalTaka dataset is at project root
REAL_NOTES_DIR = DATA_ROOT / "real_notes"
FAKE_NOTES_DIR = DATA_ROOT / "fake_notes"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"
SPLITS_DIR = OUTPUT_DIR / "splits"
EXPORT_DIR = OUTPUT_DIR / "exports"
EXPLAIN_DIR = OUTPUT_DIR / "explanations"

# Create all output directories
for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR, SPLITS_DIR,
          EXPORT_DIR, EXPLAIN_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================
NUM_VIEWS = 6                    # Security regions per banknote
NUM_CLASSES = 2                  # Real (1) vs Fake (0)
CLASS_NAMES = ["Fake", "Real"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Split ratios (note-level stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================================
# HARDWARE
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(4, os.cpu_count() or 1)   # Safe for RTX 4060
PIN_MEMORY = True if torch.cuda.is_available() else False
USE_AMP = True                               # Mixed precision


# ============================================================================
# IMAGE TRANSFORMS
# ============================================================================
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation params
COLOR_JITTER = {"brightness": 0.2, "contrast": 0.2,
                "saturation": 0.2, "hue": 0.05}
ROTATION_DEGREES = 15
RANDOM_FLIP_P = 0.5


# ============================================================================
# TRAINING — BASELINE (Phase 3)
# ============================================================================
BATCH_SIZE = 16                  # RTX 4060 8GB: 16 safe with AMP
LEARNING_RATE = 3e-4             # Scaled for batch_size=16
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7      # Stop early if no improvement
SCHEDULER_PATIENCE = 3           # React faster to plateaus
SCHEDULER_FACTOR = 0.5

# Backbone
BACKBONE = "resnet50"
FEATURE_DIM = 2048               # ResNet50 output features
DROPOUT_RATE = 0.5
CLASSIFIER_HIDDEN = 512


# ============================================================================
# TRAINING — ATTENTION MODEL (Phase 4)
# ============================================================================
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 2
TRANSFORMER_DIM = 512             # Projection dim for transformer
TRANSFORMER_DROPOUT = 0.1
VIEW_EMBED_DIM = 512


# ============================================================================
# ABLATION (Phase 5)
# ============================================================================
VIEW_ABLATION_CONFIGS = [1, 3, 6]
VIEW_DROPOUT_RATE = 0.15          # Probability of dropping a view


# ============================================================================
# REPRODUCIBILITY
# ============================================================================
SEED = 42


def seed_everything(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
