"""
Script: Generate Research Outputs (Phase 10)
Creates all tables, figures, and summaries for journal submission.
"""

import sys
import json
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    seed_everything, OUTPUT_DIR, FIGURE_DIR, SPLITS_DIR,
    NUM_VIEWS, CLASS_NAMES,
)

logger = logging.getLogger(__name__)

REPORT_DIR = OUTPUT_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def generate_dataset_statistics():
    """Table 1: Dataset statistics."""
    print("\n=== Dataset Statistics ===")

    meta_path = SPLITS_DIR / "metadata_full.csv"
    if not meta_path.exists():
        print("  No metadata found. Run Phase 1 first.")
        return

    df = pd.read_csv(meta_path)

    stats = {
        "Total Notes": len(df),
        "Total Images": len(df) * NUM_VIEWS,
        "Real Notes": len(df[df.label == 1]),
        "Fake Notes": len(df[df.label == 0]),
        "Views per Note": NUM_VIEWS,
    }

    for split in ["train", "val", "test"]:
        sub = df[df.split == split]
        stats[f"{split.title()} Set"] = len(sub)
        stats[f"  - Real"] = len(sub[sub.label == 1])
        stats[f"  - Fake"] = len(sub[sub.label == 0])

    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save as JSON
    with open(REPORT_DIR / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Class distribution figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Overall distribution
    labels = ["Real", "Fake"]
    sizes = [len(df[df.label == 1]), len(df[df.label == 0])]
    axes[0].pie(sizes, labels=labels, autopct="%1.1f%%",
                colors=["#4ecdc4", "#ff6b6b"], startangle=90)
    axes[0].set_title("Overall Class Distribution")

    # Per-split distribution
    splits_data = {}
    for split in ["train", "val", "test"]:
        sub = df[df.split == split]
        splits_data[split] = {
            "real": len(sub[sub.label == 1]),
            "fake": len(sub[sub.label == 0]),
        }

    x = np.arange(len(splits_data))
    width = 0.35
    real_counts = [v["real"] for v in splits_data.values()]
    fake_counts = [v["fake"] for v in splits_data.values()]
    axes[1].bar(x - width/2, real_counts, width, label="Real", color="#4ecdc4")
    axes[1].bar(x + width/2, fake_counts, width, label="Fake", color="#ff6b6b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.title() for s in splits_data.keys()])
    axes[1].set_ylabel("Count")
    axes[1].set_title("Split Distribution")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "dataset_distribution.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIGURE_DIR / 'dataset_distribution.png'}")


def generate_architecture_diagram():
    """Generate text-based architecture summary."""
    print("\n=== Architecture Summary ===")

    arch_summary = """
    ┌─────────────────────────────────────────────────────────────┐
    │              JaalTaka Multi-View Architecture                │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Input: (B, 6, 3, 224, 224)                                │
    │     ↓                                                       │
    │  ┌──────────────────────────────────────┐                   │
    │  │   ResNet-50 Backbone (shared)         │                  │
    │  │   ImageNet pretrained                 │                  │
    │  │   Output: (B, 6, 2048)               │                  │
    │  └──────────────┬───────────────────────┘                   │
    │                 ↓                                            │
    │  ┌──────────────────────────────────────┐                   │
    │  │   Linear Projection (2048 → 512)     │                  │
    │  │   LayerNorm + ReLU + Dropout          │                  │
    │  └──────────────┬───────────────────────┘                   │
    │                 ↓                                            │
    │  ┌──────────────────────────────────────┐                   │
    │  │   + Learnable View Embeddings (6×512)│                   │
    │  └──────────────┬───────────────────────┘                   │
    │                 ↓                                            │
    │  ┌──────────────────────────────────────┐                   │
    │  │   Transformer Encoder                 │                  │
    │  │   2 layers, 8 heads, GELU, Pre-LN    │                  │
    │  │   Output: (B, 6, 512)                │                  │
    │  └──────────────┬───────────────────────┘                   │
    │                 ↓                                            │
    │  ┌──────────────────────────────────────┐                   │
    │  │   Attention Pooling                   │                  │
    │  │   Learned view-importance weights     │                  │
    │  │   Output: (B, 512)                   │                  │
    │  └──────────────┬───────────────────────┘                   │
    │                 ↓                                            │
    │  ┌──────────────────────────────────────┐                   │
    │  │   Classifier Head                     │                  │
    │  │   512 → 512 → ReLU → Dropout → 2    │                  │
    │  └──────────────┬───────────────────────┘                   │
    │                 ↓                                            │
    │  Output: (B, 2) — [Fake, Real]                             │
    └─────────────────────────────────────────────────────────────┘
    """
    print(arch_summary)

    with open(REPORT_DIR / "architecture_diagram.txt", "w", encoding="utf-8") as f:
        f.write(arch_summary)


def generate_results_tables():
    """Load and format all experimental results."""
    print("\n=== Experimental Results ===")

    ablation_dir = OUTPUT_DIR / "ablation"

    tables = {}
    for csv_file in ablation_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        tables[csv_file.stem] = df
        print(f"\n  --- {csv_file.stem} ---")
        print(df.to_string(index=False))

        # Export LaTeX
        latex = df.to_latex(index=False, float_format="%.4f")
        with open(REPORT_DIR / f"{csv_file.stem}.tex", "w", encoding="utf-8") as f:
            f.write(latex)

    return tables


def generate_full_report():
    """Generate complete research report."""
    print("\n" + "=" * 60)
    print("PHASE 10 — Research Output Generation")
    print("=" * 60)

    generate_dataset_statistics()
    generate_architecture_diagram()
    generate_results_tables()

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {REPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    seed_everything()
    generate_full_report()
