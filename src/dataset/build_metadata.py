"""
Phase 1 — Dataset Metadata Builder
====================================
Scans the JaalTaka folder structure, validates images, builds a metadata
DataFrame, performs stratified note-level splits, and saves CSVs.

Output CSVs columns:
    note_id, note_path, img_1..img_6, label, split
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Allow running as script or module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    REAL_NOTES_DIR, FAKE_NOTES_DIR, NUM_VIEWS, SPLITS_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED, IMAGE_EXTENSIONS,
    seed_everything,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================

def _is_valid_image(path: Path) -> bool:
    """Check if an image file is readable and not corrupted."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _scan_notes(root_dir: Path, label: int) -> List[dict]:
    """
    Scan a class directory (real_notes / fake_notes) and return metadata
    for every valid note (i.e. notes with exactly NUM_VIEWS good images).
    """
    records = []
    if not root_dir.exists():
        logger.warning(f"Directory not found: {root_dir}")
        return records

    note_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    skipped_count = 0

    for note_dir in note_dirs:
        # Gather image files sorted by name
        images = sorted([
            f for f in note_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(images) != NUM_VIEWS:
            logger.warning(
                f"Skipping {note_dir.name}: expected {NUM_VIEWS} images, "
                f"found {len(images)}"
            )
            skipped_count += 1
            continue

        # Validate each image
        valid = True
        for img_path in images:
            if not _is_valid_image(img_path):
                logger.warning(f"Corrupted image: {img_path}")
                valid = False
                break

        if not valid:
            skipped_count += 1
            continue

        record = {
            "note_id": note_dir.name,
            "note_path": str(note_dir),
            "label": label,
        }
        for i, img_path in enumerate(images, start=1):
            record[f"img_{i}"] = str(img_path)

        records.append(record)

    logger.info(
        f"[{root_dir.name}] valid={len(records)}, skipped={skipped_count}"
    )
    return records


# ============================================================================
# MAIN BUILD FUNCTION
# ============================================================================

def build_metadata() -> pd.DataFrame:
    """Scan both class directories and return a combined DataFrame."""
    logger.info("Scanning real notes …")
    real_records = _scan_notes(REAL_NOTES_DIR, label=1)
    logger.info("Scanning fake notes …")
    fake_records = _scan_notes(FAKE_NOTES_DIR, label=0)

    df = pd.DataFrame(real_records + fake_records)
    logger.info(f"Total valid notes: {len(df)} "
                f"(real={len(real_records)}, fake={len(fake_records)})")
    return df


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform stratified note-level split ensuring all 6 views of a note
    stay in the same partition.

    Adds a `split` column with values: train / val / test.
    """
    seed_everything()

    labels = df["label"].values

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        np.arange(len(df)),
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=SEED,
    )

    # Second split: val vs test (equal halves of the remainder)
    temp_labels = labels[temp_idx]
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=SEED,
    )

    df = df.copy()
    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    for s in ["train", "val", "test"]:
        subset = df[df["split"] == s]
        logger.info(
            f"  {s:>5}: {len(subset)} notes | "
            f"real={len(subset[subset.label==1])}, "
            f"fake={len(subset[subset.label==0])}"
        )

    return df


def save_splits(df: pd.DataFrame) -> Tuple[Path, Path, Path, Path]:
    """Save full metadata and per-split CSVs."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    full_path = SPLITS_DIR / "metadata_full.csv"
    df.to_csv(full_path, index=False)
    logger.info(f"Saved -> {full_path}")

    paths = [full_path]
    for split_name in ["train", "val", "test"]:
        p = SPLITS_DIR / f"{split_name}.csv"
        df[df["split"] == split_name].to_csv(p, index=False)
        paths.append(p)
        logger.info(f"Saved -> {p}")

    return tuple(paths)


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_phase1():
    """Execute the full Phase 1 pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1 -- Dataset Engineering")
    logger.info("=" * 60)
    df = build_metadata()
    df = stratified_split(df)
    save_splits(df)

    # Summary statistics
    logger.info("\n--- Dataset Summary ---")
    logger.info(f"  Total notes       : {len(df)}")
    logger.info(f"  Total images      : {len(df) * NUM_VIEWS}")
    logger.info(f"  Real notes        : {len(df[df.label==1])}")
    logger.info(f"  Fake notes        : {len(df[df.label==0])}")
    logger.info(f"  Class ratio (R/F) : "
                f"{len(df[df.label==1]) / len(df[df.label==0]):.2f}")
    return df


if __name__ == "__main__":
    run_phase1()
