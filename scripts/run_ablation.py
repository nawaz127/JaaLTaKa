"""
Script: Run Ablation Studies (Phase 5)
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
                            / "outputs" / "logs" / "ablation.log"),
    ]
)

from src.config import seed_everything, NUM_EPOCHS
from src.training.ablation import (
    run_view_ablation,
    run_view_dropout_ablation,
    run_model_comparison,
    generate_ablation_summary,
)


def main():
    seed_everything()

    # Reduced epochs for ablation (can be overridden)
    ablation_epochs = min(30, NUM_EPOCHS)

    print("=" * 60)
    print("ABLATION STUDY 1: View Ablation")
    print("=" * 60)
    view_results = run_view_ablation(
        num_views_list=[1, 3, 6],
        model_class="attention",
        num_epochs=ablation_epochs,
    )

    print("\n" + "=" * 60)
    print("ABLATION STUDY 2: View Dropout Regularization")
    print("=" * 60)
    dropout_results = run_view_dropout_ablation(
        dropout_rates=[0.0, 0.1, 0.15, 0.25],
        num_epochs=ablation_epochs,
    )

    print("\n" + "=" * 60)
    print("ABLATION STUDY 3: Model Comparison")
    print("=" * 60)
    comparison_results = run_model_comparison(num_epochs=ablation_epochs)

    print("\n" + "=" * 60)
    print("Generating Summary")
    print("=" * 60)
    generate_ablation_summary()

    print("\nAll ablation studies complete.")


if __name__ == "__main__":
    main()
