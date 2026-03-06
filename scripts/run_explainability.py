"""
Script: Run Explainability Analysis (Phase 6)
Generates Grad-CAM, LIME, and SHAP explanations.
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
                            / "outputs" / "logs" / "explainability.log"),
    ]
)

import torch
from src.config import (
    seed_everything, DEVICE, CHECKPOINT_DIR, SPLITS_DIR, USE_AMP,
)
from src.dataset.dataloader import build_dataloaders, get_eval_transforms
from src.dataset.dataloader import MultiViewBanknoteDataset
from src.models.attention import MultiViewAttentionNet
from src.explainability.gradcam import (
    MultiViewGradCAM, visualize_gradcam, evaluate_explanation_stability,
)
from src.explainability.lime_explain import MultiViewLIME, visualize_lime
from src.explainability.shap_explain import (
    MultiViewSHAP, visualize_shap_bar, visualize_dataset_shap,
)

logger = logging.getLogger(__name__)


def load_best_model() -> torch.nn.Module:
    """Load the best attention model checkpoint."""
    model = MultiViewAttentionNet(pretrained=False)
    ckpt_path = CHECKPOINT_DIR / "attention_transformer_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded model from {ckpt_path}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}, using untrained model")
        model = MultiViewAttentionNet(pretrained=True)
    model = model.to(DEVICE)
    model.eval()
    return model


def _collect_samples_by_class(test_loader, num_per_class=5):
    """Collect equal numbers of real (label=1) and fake (label=0) samples."""
    fake_samples = []  # label 0
    real_samples = []  # label 1
    for views, label in test_loader:
        lbl = label.item()
        if lbl == 0 and len(fake_samples) < num_per_class:
            fake_samples.append((views, label))
        elif lbl == 1 and len(real_samples) < num_per_class:
            real_samples.append((views, label))
        if len(fake_samples) >= num_per_class and len(real_samples) >= num_per_class:
            break
    # Interleave: fake_0, real_0, fake_1, real_1, ...
    samples = []
    for f, r in zip(fake_samples, real_samples):
        samples.append(f)
        samples.append(r)
    return samples


def main():
    seed_everything()
    model = load_best_model()

    # Load test set
    _, _, test_loader = build_dataloaders(batch_size=1)

    # Collect balanced real + fake samples
    class_names = ["Fake", "Real"]
    num_per_class = 5
    samples = _collect_samples_by_class(test_loader, num_per_class=num_per_class)
    print(f"\nCollected {len(samples)} samples ({num_per_class} fake + {num_per_class} real)")

    # ================================================================
    # GRAD-CAM
    # ================================================================
    print("\n" + "=" * 60)
    print("GRAD-CAM Explanations")
    print("=" * 60)

    gradcam = MultiViewGradCAM(model)

    for i, (views, label) in enumerate(samples):
        label_name = class_names[label.item()]
        heatmaps, pred_class, conf = gradcam.generate(views)
        pred_name = class_names[pred_class]
        note_id = f"sample_{i:03d}_{label_name}"
        print(f"  [{i}] True={label_name}, Pred={pred_name} (conf={conf:.4f})")
        visualize_gradcam(
            views, heatmaps, pred_class, conf,
            note_id=note_id,
        )

    # Evaluate stability
    print("\nEvaluating Grad-CAM stability ...")
    views, _ = samples[0]
    stability = evaluate_explanation_stability(model, views)
    print(f"  Grad-CAM stability score: {stability:.4f}")

    gradcam.cleanup()

    # ================================================================
    # LIME
    # ================================================================
    print("\n" + "=" * 60)
    print("LIME Explanations")
    print("=" * 60)

    lime_explainer = MultiViewLIME(model)

    for i, (views, label) in enumerate(samples):
        label_name = class_names[label.item()]
        print(f"\n  LIME sample {i+1} (True={label_name}) ...")
        explanations = lime_explainer.explain_all_views(
            views, num_samples=300,
        )

        with torch.no_grad():
            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=USE_AMP):
                logits = model(views.to(DEVICE))
            pred_class = logits.argmax(dim=1).item()

        note_id = f"sample_{i:03d}_{label_name}"
        visualize_lime(
            views, explanations, pred_class,
            note_id=note_id,
        )

    # ================================================================
    # SHAP
    # ================================================================
    print("\n" + "=" * 60)
    print("SHAP View-Level Attribution")
    print("=" * 60)

    shap_explainer = MultiViewSHAP(model)

    # Per-sample SHAP for one fake and one real
    for i, (views, label) in enumerate(samples[:2]):
        label_name = class_names[label.item()]
        sv = shap_explainer.compute_view_shapley_values(views)
        visualize_shap_bar(sv, f"sample_{i:03d}_{label_name}")
        print(f"  SHAP sample {i}: True={label_name}, values={sv}")

    # Global dataset SHAP (small subset for speed)
    print("\nComputing dataset-level SHAP values ...")
    _, _, test_loader_shap = build_dataloaders(batch_size=1)
    avg_sv = shap_explainer.compute_dataset_shapley(
        test_loader_shap, max_samples=30,
    )
    visualize_dataset_shap(avg_sv)

    print("\n" + "=" * 60)
    print("Explainability analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
