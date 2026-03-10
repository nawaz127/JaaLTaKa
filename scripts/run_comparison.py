"""
Script: Run Model Comparison (Phase 5 Extension)
===============================================
Compares multiple baselines (ResNet50, MobileNetV2, EfficientNet-B0)
against the Proposed Attention Transformer model.
"""

import sys
import logging
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).resolve().parent.parent
                            / "outputs" / "logs" / "model_comparison.log"),
    ]
)

from src.training.ablation import run_model_comparison, generate_ablation_summary


def main():
    parser = argparse.ArgumentParser(description="Run Model Comparison")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs per model (default 5 for quick comparison)")
    args = parser.parse_args()

    print(f"Starting model comparison with {args.epochs} epochs per model...")
    
    # Run comparison
    df = run_model_comparison(num_epochs=args.epochs)
    
    # Generate summary plots
    generate_ablation_summary()
    
    print("\nComparison complete. Results saved to outputs/ablation/model_comparison.csv")
    print("Summary plot saved to outputs/figures/model_comparison_bar.png")


if __name__ == "__main__":
    main()
