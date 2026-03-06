"""
Script: Export and Compress Model (Phase 7)
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
                            / "outputs" / "logs" / "export.log"),
    ]
)

import torch
from src.config import seed_everything, DEVICE, CHECKPOINT_DIR
from src.models.attention import MultiViewAttentionNet
from src.compression.export import run_compression_pipeline


def main():
    seed_everything()

    # Load best model
    model = MultiViewAttentionNet(pretrained=False)
    ckpt_path = CHECKPOINT_DIR / "attention_transformer_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: No trained checkpoint found. Using untrained model.")
        model = MultiViewAttentionNet(pretrained=True)

    model.eval()

    # Run full compression pipeline
    results = run_compression_pipeline(model, "jaaltaka_attention")

    print("\n=== Compression Results ===")
    for fmt, stats in results.items():
        print(f"\n  {fmt}:")
        for k, v in stats.items():
            print(f"    {k}: {v:.2f}" if isinstance(v, float)
                  else f"    {k}: {v}")


if __name__ == "__main__":
    main()
