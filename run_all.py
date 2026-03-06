"""
JaalTaka — Master Pipeline Script
=====================================
Run all phases sequentially or selectively.

Usage:
    python run_all.py                  # Run everything
    python run_all.py --phase 1        # Run only Phase 1
    python run_all.py --phase 1 2 3    # Run Phases 1, 2, 3
"""

import sys
import os
import warnings
import argparse
import logging
from pathlib import Path

# Suppress repetitive pandas dependency warnings from dataloader workers
warnings.filterwarnings("ignore", message=".*Pandas requires version.*")
warnings.filterwarnings("ignore", message=".*NumExpr.*")
os.environ["NUMEXPR_MAX_THREADS"] = str(min(8, os.cpu_count() or 1))

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/logs/pipeline.log"),
    ]
)
logger = logging.getLogger(__name__)


def phase1():
    """Dataset Engineering — build metadata and splits."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 -- Dataset Engineering")
    logger.info("=" * 70)
    from src.dataset.build_metadata import run_phase1
    run_phase1()


def phase2():
    """Multi-View Data Pipeline — verify dataloaders."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 -- Multi-View Data Pipeline (verification)")
    logger.info("=" * 70)
    from src.dataset.dataloader import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders()
    # Quick sanity check
    views, labels = next(iter(train_loader))
    logger.info(f"  Train batch shape: {views.shape}")
    logger.info(f"  Labels: {labels}")
    logger.info(f"  Train size: {len(train_loader.dataset)}")
    logger.info(f"  Val size:   {len(val_loader.dataset)}")
    logger.info(f"  Test size:  {len(test_loader.dataset)}")


def phase3():
    """Baseline Model Training."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3 -- Baseline Model Training")
    logger.info("=" * 70)
    from scripts.train_baseline import main
    main()


def phase4():
    """Attention Model Training."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4 -- Attention Model Training")
    logger.info("=" * 70)
    from scripts.train_attention import main
    main()


def phase5():
    """Ablation Studies."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5 -- Ablation Studies")
    logger.info("=" * 70)
    from scripts.run_ablation import main
    main()


def phase6():
    """Explainability Analysis."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6 -- Explainability Analysis")
    logger.info("=" * 70)
    from scripts.run_explainability import main
    main()


def phase7():
    """Model Compression."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7 -- Model Compression")
    logger.info("=" * 70)
    from scripts.export_model import main
    main()


def phase10():
    """Research Outputs."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 10 -- Research Outputs")
    logger.info("=" * 70)
    from scripts.generate_report import generate_full_report
    generate_full_report()


PHASES = {
    1: phase1,
    2: phase2,
    3: phase3,
    4: phase4,
    5: phase5,
    6: phase6,
    7: phase7,
    10: phase10,
}


def main():
    from src.config import seed_everything, DEVICE
    seed_everything()

    parser = argparse.ArgumentParser(description="JaalTaka Pipeline")
    parser.add_argument(
        "--phase", nargs="*", type=int, default=None,
        help="Phase numbers to run (e.g., 1 2 3). Default: all."
    )
    args = parser.parse_args()

    logger.info(f"Device: {DEVICE}")

    if args.phase is None:
        phases_to_run = sorted(PHASES.keys())
    else:
        phases_to_run = args.phase

    for p in phases_to_run:
        if p in PHASES:
            PHASES[p]()
        else:
            logger.warning(f"Phase {p} not implemented in this runner.")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
