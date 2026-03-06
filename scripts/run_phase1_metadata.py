"""
Script: Run Phase 1 — Build dataset metadata and splits.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import seed_everything
from src.dataset.build_metadata import run_phase1

if __name__ == "__main__":
    seed_everything()
    df = run_phase1()
    print(f"\nDone. {len(df)} notes processed.")
