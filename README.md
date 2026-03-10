# JaalTaka: Multi-View Counterfeit Banknote Detection

### **Developed by Shah Nawaz**

> A deep learning system for detecting counterfeit Bangladeshi banknotes using multi-view analysis, attention-based transformer fusion, and explainable AI (XAI). Featuring a complete ML pipeline (Phases 1–10), a Flutter Android app with on-device ONNX inference, and a Streamlit web dashboard with Grad-CAM, LIME, and SHAP visualizations.

| Metric | Value |
|---|---|
| **Test Accuracy** | 99.04% |
| **ROC-AUC** | 0.9925 |
| **Model** | Attention Transformer (31.2M params) |
| **Views** | 6 (Front, Back, Watermark, Thread, Serial, Hologram) |
| **ONNX Model** | 119 MB (FP32) / 30 MB (INT8) |
| **Mobile Inference** | ~4.5s (ONNX Runtime, Android) |
| **Developer** | **Shah Nawaz** |

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Pipeline Phases](#pipeline-phases)
5. [Model Architectures](#model-architectures)
6. [Training Configuration](#training-configuration)
7. [Results](#results)
   - [Phase 3: Baseline Model](#phase-3-baseline-model-resnet50--mean-pooling)
   - [Phase 4: Attention Model](#phase-4-attention-transformer-model)
   - [Phase 5: Ablation Studies](#phase-5-ablation-studies)
   - [Phase 6: Explainability](#phase-6-explainability-analysis)
   - [Phase 7: Model Compression](#phase-7-model-compression)
8. [Generated Outputs](#generated-outputs)
9. [Flutter App (Phase 9)](#flutter-app-phase-9)
10. [Streamlit Web App](#streamlit-web-app)
11. [Requirements](#requirements)
12. [Author](#author)

---

## Dataset

| Property | Value |
|---|---|
| Total notes | **1,390** |
| Real notes | 802 (57.7%) |
| Fake notes | 588 (42.3%) |
| Class ratio (Real/Fake) | 1.36 |
| Views per note | 6 |
| Total images | **8,340** |
| Image resolution | 224 × 224 px |
| Split strategy | Stratified, note-level (no data leakage) |

### Data Splits

| Split | Notes | Real | Fake |
|---|---|---|---|
| Train (70%) | 973 | 561 | 412 |
| Validation (15%) | 208 | 120 | 88 |
| Test (15%) | 209 | 121 | 88 |

Input tensor shape: `[batch_size, 6, 3, 224, 224]` (batch × views × channels × H × W)

---

## Project Structure

```
JaalTaka/
├── src/
│   ├── config.py                 # Global configuration
│   ├── dataset/
│   │   ├── build_metadata.py     # Phase 1: Dataset engineering
│   │   └── dataloader.py         # Phase 2: Multi-view data pipeline
│   ├── models/
│   │   ├── baseline.py           # Phase 3: MultiViewResNet
│   │   └── attention.py          # Phase 4: MultiViewAttentionNet
│   ├── training/
│   │   ├── trainer.py            # Training engine (AMP, early stopping)
│   │   └── ablation.py           # Phase 5: Ablation studies
│   ├── evaluation/
│   │   └── metrics.py            # Metrics, plots, confusion matrices
│   ├── explainability/
│   │   ├── gradcam.py            # Phase 6: Grad-CAM
│   │   ├── lime_explain.py       # Phase 6: LIME
│   │   └── shap_explain.py       # Phase 6: SHAP
│   ├── compression/
│   │   └── export.py             # Phase 7: ONNX/TFLite export
│   └── mobile/
│       └── inference.py          # Phase 8: Mobile inference pipeline
├── scripts/
│   ├── run_phase1_metadata.py    # Build dataset metadata
│   ├── train_baseline.py         # Train ResNet baseline
│   ├── train_attention.py        # Train attention model
│   ├── run_ablation.py           # Run ablation studies
│   ├── run_explainability.py     # Generate explanations
│   ├── export_model.py           # Export & compress model
│   ├── generate_report.py        # Generate research outputs
│   └── quantize_onnx.py         # INT8 ONNX quantization
├── flutter_app/                  # Phase 9: Mobile application
│   ├── lib/
│   │   ├── main.dart            # App entry with theme & splash
│   │   ├── models/auth_result.dart   # Data models + history
│   │   ├── providers/auth_provider.dart  # State management
│   │   ├── services/onnx_service.dart    # ONNX Runtime inference
│   │   └── screens/
│   │       ├── splash_screen.dart        # Splash + onboarding
│   │       ├── camera_screen.dart        # Multi-view guided capture
│   │       ├── preview_screen.dart       # Preview before inference
│   │       ├── result_screen.dart        # Auth result + share
│   │       ├── explanation_screen.dart   # XAI (SHAP + heatmap)
│   │       └── history_screen.dart       # Scan history
│   ├── assets/models/           # ONNX model files
│   └── pubspec.yaml
├── streamlit_app/               # Phase 11: Web Dashboard
│   ├── app.py                   # Streamlit app (Grad-CAM, LIME, SHAP)
│   └── requirements.txt
├── outputs/
│   ├── checkpoints/              # Model weights (.pth)
│   ├── logs/                     # Pipeline logs
│   ├── figures/                  # Training curves, confusion matrices, ROC
│   ├── splits/                   # CSV split files
│   ├── exports/                  # ONNX model (FP32 + INT8)
│   ├── explanations/             # Grad-CAM / LIME / SHAP outputs
│   ├── ablation/                 # Ablation study results
│   └── report/                   # LaTeX tables, architecture diagram, stats
├── run_all.py                    # Master pipeline runner
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
conda create -n torch_gpu python=3.11 -y
conda activate torch_gpu
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python run_all.py
```

### 3. Run Individual Phases

```bash
python run_all.py --phase 1      # Phase 1: Build dataset metadata & splits
python run_all.py --phase 2      # Phase 2: Verify data pipeline
python run_all.py --phase 3      # Phase 3: Train baseline model
python run_all.py --phase 4      # Phase 4: Train attention model
python run_all.py --phase 5      # Phase 5: Ablation studies
python run_all.py --phase 6      # Phase 6: Explainability (Grad-CAM, LIME, SHAP)
python run_all.py --phase 7      # Phase 7: Model compression (ONNX export)
python run_all.py --phase 10     # Phase 10: Generate research outputs
```

Run multiple phases at once:

```bash
python run_all.py --phase 6 7 10
```

---

## Pipeline Phases

| Phase | Name | Status | Description |
|---|---|---|---|
| 1 | Dataset Engineering | ✅ Complete | Scan notes, validate 6 views per note, create stratified splits |
| 2 | Data Pipeline | ✅ Complete | Multi-view DataLoader with augmentations, CUDA verification |
| 3 | Baseline Training | ✅ Complete | ResNet50 + mean pooling baseline |
| 4 | Attention Training | ✅ Complete | Transformer-based multi-view fusion |
| 5 | Ablation Studies | ✅ Complete | View count, view dropout rate, model comparison |
| 6 | Explainability | ✅ Complete | Grad-CAM, LIME, SHAP on real & fake samples |
| 7 | Model Compression | ✅ Complete | ONNX export (FP32 119MB) + INT8 quantization (30MB) |
| 8 | Mobile Inference | ✅ Complete | ONNX Runtime on-device inference pipeline |
| 9 | Flutter App | ✅ Complete | Full Android app with 6-view capture, XAI heatmaps, history |
| 10 | Research Outputs | ✅ Complete | LaTeX tables, architecture diagram, dataset stats |
| 11 | Streamlit Web App | ✅ Complete | Web dashboard with Grad-CAM, LIME, SHAP, batch analysis, PDF |

---

## Model Architectures
### Baseline (Phase 3): MultiViewBaseline

```
Input [B, 6, 3, 224, 224]
    │
    ▼
Backbone (ResNet50 / MobileNetV2 / EfficientNet-B0)
    │
    ▼ [B, 6, D] feature vectors
    │
Mean Pooling across 6 views
    │
    ▼ [B, D]
    │
Linear(D → 512) → ReLU → Dropout(0.3) → Linear(512 → 2)
    │
    ▼ [B, 2] logits
```

### Advanced (Phase 4): MultiViewAttentionNet

```
Input [B, 6, 3, 224, 224]
    │
    ▼
Backbone (ResNet50)
    │
    ▼ [B, 6, D]
    │
Linear projection (D → 512) + LayerNorm
...
```

    ▼ [B, 6, 512]
    │
+ Learnable view positional embeddings
    │
    ▼
Transformer Encoder (2 layers, 8 heads, dim_ff=2048, GELU, Pre-LN)
    │
    ▼ [B, 6, 512]
    │
Attention Pooling (learned query → weighted view aggregation)
    │
    ▼ [B, 512]
    │
Linear(512 → 2)
    │
    ▼ [B, 2] logits
```

- **Total parameters**: 31,195,459 (all trainable)

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | AdamW |
| Initial learning rate | 3×10⁻⁴ (ablations) / 1×10⁻⁴ (baseline/attention) |
| Weight decay | 1×10⁻⁴ |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | 50 (baseline/attention) / 30 (ablations) |
| Early stopping patience | 7 |
| Mixed precision | AMP (GradScaler + autocast) |
| Gradient clipping | max_norm = 1.0 |
| Device | CUDA (GPU) |

---

## Results

### Phase 3: Baseline Model (ResNet50 + Mean Pooling)

**Training**: 11 epochs, early stopped at epoch 11 (best checkpoint at epoch 4)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|---|---|---|---|---|---|
| 1 | 0.1926 | 0.9219 | 0.0923 | 0.9808 | 3.00e-04 |
| 4 | 0.0850 | 0.9896 | 0.1241 | **0.9856** | 3.00e-04 |
| 11 | 0.0228 | 0.9917 | 0.1136 | 0.9856 | 1.50e-04 |

**Test Set Evaluation**:

| Metric | Score |
|---|---|
| **Accuracy** | **99.04%** |
| Precision | 99.17% |
| Recall | 99.17% |
| F1-score | 0.9917 |
| ROC-AUC | **0.9917** |

**Per-class breakdown**:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Fake | 0.99 | 0.99 | 0.99 | 88 |
| Real | 0.99 | 0.99 | 0.99 | 121 |

---

### Phase 4: Attention Transformer Model

**Training**: 10 epochs, early stopped at epoch 10 (best checkpoint at epoch 3)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|---|---|---|---|---|---|
| 1 | 0.4736 | 0.8958 | 0.2599 | 0.9808 | 3.00e-04 |
| 3 | 0.2081 | 0.9792 | 0.1037 | **0.9856** | 3.00e-04 |
| 10 | 0.0598 | 0.9885 | 0.3340 | 0.9856 | 1.50e-04 |

**Test Set Evaluation**:

| Metric | Score |
|---|---|
| **Accuracy** | **99.04%** |
| Precision | 99.17% |
| Recall | 99.17% |
| F1-score | 0.9917 |
| ROC-AUC | **0.9925** |

The Attention Transformer achieves the same accuracy as the baseline but with a **higher ROC-AUC** (0.9925 vs 0.9917), indicating better probability calibration and ranking quality.

---

### Phase 5: Ablation Studies

#### 5a. View Count Ablation

How does the number of input views affect classification?

| Views | Accuracy | Precision | Recall | F1 | ROC-AUC | Epochs |
|---|---|---|---|---|---|---|
| 1 | 99.04% | 0.9917 | 0.9917 | 0.9917 | 0.9898 | 12 |
| 3 | 99.04% | 0.9917 | 0.9917 | 0.9917 | 0.9894 | 9 |
| **6** | **99.04%** | **0.9917** | **0.9917** | **0.9917** | **0.9925** | **10** |

**Finding**: All view counts achieve identical test accuracy (99.04%), but 6 views yields the highest ROC-AUC (0.9925), demonstrating that multi-view input improves the model's confidence and ranking ability even when top-line accuracy is saturated.

#### 5b. View Dropout Ablation

What view dropout rate during training yields the most robust model?

| Dropout Rate | Accuracy | Precision | Recall | F1 | ROC-AUC | Best Val Acc | Epochs |
|---|---|---|---|---|---|---|---|
| 0.00 | 99.04% | 0.9917 | 0.9917 | 0.9917 | 0.9925 | 0.9856 | 10 |
| **0.10** | **98.56%** | **0.9836** | **0.9917** | **0.9877** | **0.9979** | **0.9952** | **21** |
| 0.15 | 99.04% | 0.9917 | 0.9917 | 0.9917 | 0.9943 | 0.9856 | 9 |
| 0.25 | 99.04% | 0.9917 | 0.9917 | 0.9917 | 0.9899 | 0.9952 | 10 |

**Finding**: View dropout rate of **0.10** achieves the **highest ROC-AUC (0.9979)** and the **highest validation accuracy (99.52%)**, training for the most epochs (21) which suggests the regularization effect encourages deeper convergence. Although test accuracy is slightly lower (98.56%), the model is significantly better calibrated.

#### 5c. Model Comparison

Head-to-head comparison with identical training settings (30-50 epochs, lr=3e-4):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Parameters |
|---|---|---|---|---|---|---|
| MobileNetV2_MeanPool | 99.04% | 0.9917 | 0.9917 | 0.9917 | 0.9906 | **2.9M** |
| EfficientNetB0_MeanPool | 99.04% | 0.9917 | 0.9917 | 0.9917 | 0.9900 | 4.7M |
| ResNet50_MeanPool | 99.04% | 0.9917 | 0.9917 | 0.9917 | 0.9917 | 24.6M |
| **Proposed Attention Net** | **99.04%** | **0.9917** | **0.9917** | **0.9917** | **0.9925** | 31.2M |

**Finding**: While all models hit a "data saturation" ceiling of 99.04% due to two specific outlier notes (note_402, note_405), the **Proposed Attention Transformer** achieves the **highest ROC-AUC (0.9925)**, indicating superior ranking quality and better handling of ambiguous samples. **MobileNetV2** is the most parameter-efficient baseline.

---

### Phase 6: Explainability Analysis

Three XAI methods applied to the best Attention Transformer model, with balanced samples (5 fake + 5 real notes):

#### Grad-CAM
- **10 heatmap visualizations** generated (interleaved: Fake, Real, Fake, Real, ...)
- Highlights which spatial regions in each view the model focuses on
- **Grad-CAM stability score: 0.9913** (high consistency across views)
- Output: `outputs/explanations/gradcam/gradcam_sample_*_{Fake|Real}.png`

#### LIME (Local Interpretable Model-agnostic Explanations)
- **10 superpixel explanations** (5 fake, 5 real), 300 perturbation samples per view
- Identifies which image superpixels contribute most to the prediction
- All 6 views explained per sample
- Output: `outputs/explanations/lime/lime_sample_*_{Fake|Real}.png`

#### SHAP (SHapley Additive exPlanations)
- **Per-sample visualizations** for first 2 samples (Fake + Real)
- **Global dataset SHAP values** computed over 30 samples:

| View | SHAP Value |
|---|---|
| View 1 | 0.2990 |
| View 2 | 0.1211 |
| View 3 | 0.2655 |
| View 4 | 0.2461 |
| View 5 | 0.2104 |
| View 6 | 0.2996 |

**Finding**: Views 1 and 6 contribute the most to predictions (SHAP ≈ 0.30), while View 2 contributes the least (SHAP ≈ 0.12). This suggests the front and back faces of the banknote carry the strongest authentication signals.

---

### Phase 7: Model Compression

| Format | Size | Mean Latency | Status |
|---|---|---|---|
| PyTorch (.pth) | 119.3 MB | — | ✅ |
| ONNX FP32 (.onnx) | 119.0 MB | 97.5 ms | ✅ |
| **ONNX INT8** | **30.1 MB** | — | ✅ |
| TFLite FP32 | — | — | ❌ Blocked |

- **74.7% size reduction** with INT8 dynamic quantization
- ONNX validation: **PASSED** (numerical correctness verified)
- INT8 model: `outputs/exports/jaaltaka_attention_int8.onnx`

---

## Generated Outputs

### Figures (`outputs/figures/`)

| File | Description |
|---|---|
| `baseline_resnet50_confusion.png` | Baseline confusion matrix |
| `baseline_resnet50_curves.png` | Baseline training/validation curves |
| `baseline_resnet50_roc.png` | Baseline ROC curve |
| `attention_transformer_confusion.png` | Attention model confusion matrix |
| `attention_transformer_curves.png` | Attention training/validation curves |
| `attention_transformer_roc.png` | Attention ROC curve |
| `ablation_attention_{1,3,6}views_*.png` | View count ablation plots |
| `ablation_viewdropout_{0.00,0.10,0.15,0.25}_*.png` | View dropout ablation plots |
| `comparison_{ResNet_MeanPool,Attention_Transformer}_*.png` | Model comparison plots |
| `model_comparison_bar.png` | Side-by-side bar chart comparison |
| `dataset_distribution.png` | Dataset class distribution |

### Reports (`outputs/report/`)

| File | Description |
|---|---|
| `dataset_statistics.json` | Dataset statistics in JSON |
| `architecture_diagram.txt` | Text-based architecture diagram |
| `model_comparison.tex` | LaTeX table for model comparison |
| `view_ablation.tex` | LaTeX table for view ablation results |
| `view_dropout_ablation.tex` | LaTeX table for view dropout results |

### Model Checkpoints (`outputs/checkpoints/`)

- `baseline_resnet50_best.pth` — Best baseline model
- `attention_transformer_best.pth` — Best attention model (primary)
- `ablation_attention_{1,3,6}views_best.pth` — View ablation checkpoints
- `ablation_viewdropout_{0.00,0.10,0.15,0.25}_best.pth` — Dropout ablation checkpoints
- `comparison_{ResNet_MeanPool,Attention_Transformer}_best.pth` — Comparison checkpoints

---

## Flutter App (Phase 9)

A full-featured Android application for on-device banknote authentication using ONNX Runtime.

### Features

| Feature | Description |
|---|---|
| **Multi-View Capture** | Guided 6-view capture with custom bracket overlay and per-view icons |
| **ONNX Runtime Inference** | On-device inference using `jaaltaka_attention.onnx` (119 MB) |
| **Occlusion Heatmaps** | Real-time XAI heatmap generation via occlusion sensitivity (7×7 grid) |
| **SHAP Attribution** | Pre-computed view importance bar chart |
| **Splash & Onboarding** | Animated splash screen with 3-step tutorial |
| **Gallery Upload** | Pick images from gallery via `image_picker` |
| **Flash Control** | Off / Auto / On / Torch modes |
| **Haptic Feedback** | Vibration on capture |
| **Dark/Light Theme** | System/Light/Dark toggle persisted via `SharedPreferences` |
| **Share Results** | Screenshot → share via `share_plus` |
| **Scan History** | Up to 50 past scans with thumbnails, delete/clear |
| **EXIF Fix** | Automatic rotation correction via `bakeOrientation` |
| **Pixel Safety** | Format conversion to uint8×3 channels before normalization |

### Build

```bash
cd flutter_app
flutter pub get
flutter build apk --debug
```

### Architecture

```
SplashScreen → CameraScreen → PreviewScreen → ResultScreen → ExplanationScreen
                  ↓                                              ↑
              HistoryScreen                              (3 tabs: SHAP / Heatmap / Views)
```

### Dependencies

camera, onnxruntime, image, path_provider, permission_handler, provider,
flutter_spinkit, image_picker, share_plus, shared_preferences, intl,
vibration, screenshot

---

## Streamlit Web App

A comprehensive web dashboard for banknote authentication with full XAI visualization.

### Run

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

### Features

| Feature | Description |
|---|---|
| **Single Analysis** | Upload 6 images → instant PyTorch inference |
| **Grad-CAM** | Per-view activation heatmap overlays |
| **LIME** | Superpixel-based local explanations |
| **SHAP** | Global view-level attribution bar chart |
| **Batch Analysis** | Analyze up to 20 notes with accuracy summary table |
| **Demo Mode** | Pre-loaded sample notes from the dataset |
| **PDF Reports** | Download authentication report with all XAI results |
| **Confidence Gauge** | Interactive Plotly gauge showing real probability |
| **Attention Weights** | Dynamic bar chart of learned view importance |

### Screenshots

Access the app at `http://localhost:8501` after running `streamlit run app.py`.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended)
- torchvision, timm, scikit-learn, pandas, matplotlib, seaborn
- LIME, SHAP (explainability)
- ONNX, ONNX Runtime (model export + quantization)
- Streamlit, Plotly, fpdf2 (web app)
- Flutter ≥ 3.0 (mobile app)

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Author

**Shah Nawaz** — Developer & Researcher

- Built the complete ML pipeline (Phases 1–10)
- Designed the multi-view attention transformer architecture
- Developed the Flutter Android app with on-device ONNX inference
- Created the Streamlit web dashboard with Grad-CAM, LIME, SHAP
- Achieved **99.04% accuracy** and **0.9925 ROC-AUC** on counterfeit detection

---

## License

Research use only.
