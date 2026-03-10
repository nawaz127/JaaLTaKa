
import sys
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    seed_everything, DEVICE, CHECKPOINT_DIR, OUTPUT_DIR, NUM_VIEWS,
)
from src.dataset.dataloader import MultiViewBanknoteDataset, get_eval_transforms
from src.models.baseline import MultiViewBaseline
from src.models.attention import MultiViewAttentionNet
from src.explainability.gradcam import MultiViewGradCAM, denormalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTLIER_DIR = OUTPUT_DIR / "error_analysis"
OUTLIER_DIR.mkdir(parents=True, exist_ok=True)

# Define the outlier notes as per manuscript
# note_402 (Fake in dataset, misclassified as Real) - Wait, manuscript says:
# "note_402 (a counterfeit note misclassified as genuine) and 
#  note_405 (a genuine note misclassified as counterfeit)"
# Let's verify labels in dataset.
# Fake = 0, Real = 1

OUTLIERS = [
    {"id": "note_402", "label": 0, "split": "test", "desc": "False Positive (Fake as Real)"},
    {"id": "note_405", "label": 1, "split": "test", "desc": "False Negative (Real as Fake)"}
]

ARCHS = [
    {"name": "ResNet50", "type": "baseline", "backbone": "resnet50", "ckpt": "baseline_resnet50_best.pth"},
    {"name": "MobileNetV2", "type": "baseline", "backbone": "mobilenet_v2", "ckpt": "baseline_mobilenet_v2_best.pth"},
    {"name": "EfficientNet-B0", "type": "baseline", "backbone": "efficientnet_b0", "ckpt": "baseline_efficientnet_b0_best.pth"},
    {"name": "Proposed Attention", "type": "attention", "ckpt": "attention_transformer_best.pth"},
]

def load_model(arch_cfg):
    if arch_cfg["type"] == "baseline":
        model = MultiViewBaseline(backbone_name=arch_cfg["backbone"], pretrained=False)
    else:
        model = MultiViewAttentionNet(pretrained=False)
    
    ckpt_path = CHECKPOINT_DIR / arch_cfg["ckpt"]
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded {arch_cfg['name']} from {ckpt_path}")
    else:
        logger.warning(f"Ckpt {ckpt_path} not found!")
    
    model = model.to(DEVICE).eval()
    return model

def get_outlier_data(note_id, label, split):
    from src.config import SPLITS_DIR
    transform = get_eval_transforms()
    csv_path = SPLITS_DIR / f"{split}.csv"
    if not csv_path.exists():
        csv_path = SPLITS_DIR / "metadata_full.csv"
        
    dataset = MultiViewBanknoteDataset(csv_path=csv_path, transform=transform)
    
    for i in range(len(dataset.df)):
        row = dataset.df.iloc[i]
        if row['note_id'] == note_id and int(row['label']) == label:
            views, lbl = dataset[i]
            return views.unsqueeze(0), lbl
    return None, None

def generate_grid():
    seed_everything()
    
    # 1. Load Data
    data = {}
    for out in OUTLIERS:
        views, lbl = get_outlier_data(out["id"], out["label"], out["split"])
        if views is not None:
            data[out["id"]] = (views, lbl)
            logger.info(f"Found data for {out['id']}")
        else:
            logger.error(f"Could not find data for {out['id']} with label {out['label']}")

    # 2. Generate Heatmaps
    results = {} # (note_id, arch_name) -> heatmap
    
    for arch in ARCHS:
        model = load_model(arch)
        gradcam = MultiViewGradCAM(model)
        
        for note_id, (views, lbl) in data.items():
            # Generate for View 1 (Portrait) as representative
            heatmaps, pred, conf = gradcam.generate(views)
            results[(note_id, arch["name"])] = (heatmaps[0], pred, conf, views[0, 0])
            
        gradcam.cleanup()
        del model
        torch.cuda.empty_cache()

    # 3. Plot Grid
    # Rows: Notes (402, 405)
    # Cols: Archs (ResNet, Mobile, Eff, Proposed)
    fig, axes = plt.subplots(len(OUTLIERS), len(ARCHS), figsize=(16, 8))
    
    class_names = ["Fake", "Real"]
    
    for r, out in enumerate(OUTLIERS):
        note_id = out["id"]
        true_label = class_names[out["label"]]
        
        for c, arch in enumerate(ARCHS):
            heatmap, pred, conf, img_tensor = results[(note_id, arch["name"])]
            img = denormalize(img_tensor)
            
            ax = axes[r, c]
            ax.imshow(img)
            ax.imshow(heatmap, cmap="jet", alpha=0.5)
            
            pred_label = class_names[pred]
            color = "green" if pred == out["label"] else "red"
            
            title = f"{arch['name']}\nPred: {pred_label} ({conf:.2%})"
            ax.set_title(title, fontsize=10, color=color)
            ax.axis("off")
            
        # Add row label
        axes[r, 0].text(-20, 112, f"{note_id}\n(True: {true_label})", 
                        rotation=90, va='center', ha='right', 
                        fontweight='bold', transform=axes[r, 0].transData)

    plt.tight_layout()
    grid_path = OUTLIER_DIR / "error_analysis_grid.png"
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved grid to {grid_path}")

if __name__ == "__main__":
    generate_grid()
