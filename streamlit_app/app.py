"""
JaalTaka — Streamlit Web Application
======================================
Multi-View Counterfeit Banknote Detection with Explainable AI
Developed by Shah Nawaz

Features:
  - 6-image upload for multi-view analysis
  - Real-time ONNX inference (optimized for deployment)
  - Optional PyTorch Grad-CAM heatmap overlays
  - LIME superpixel explanations
  - SHAP view-level attribution
  - Batch analysis mode
  - PDF report generation
  - Sample demo mode

Usage:
    streamlit run streamlit_app/app.py
"""

import io
import os
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
import plotly.graph_objects as go
import onnxruntime as ort

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    NUM_VIEWS, NUM_CLASSES, CLASS_NAMES,
)
from src.models.attention import MultiViewAttentionNet

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="JaalTaka — Banknote Authentication",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CONSTANTS
# ============================================================================

VIEW_NAMES = [
    "Front", "Back", "Watermark",
    "Security Thread", "Serial Number", "Hologram/UV",
]

# Paths for both PyTorch and ONNX models
PTH_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "attention_transformer_best.pth"
ONNX_PATH = PROJECT_ROOT / "outputs" / "exports" / "jaaltaka_attention_int8.onnx"

SAMPLE_DIRS = {
    "fake": PROJECT_ROOT / "fake_notes",
    "real": PROJECT_ROOT / "real_notes",
}

SHAP_VALUES = [0.2990, 0.1211, 0.2655, 0.2461, 0.2104, 0.2996]


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_onnx_model():
    """Load the ONNX model session."""
    if not ONNX_PATH.exists():
        return None
    
    # Use CPU for Streamlit Cloud stability
    providers = ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(str(ONNX_PATH), providers=providers)
        return session
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None

@st.cache_resource
def load_torch_model():
    """Load the trained MultiViewAttentionNet model (required for Grad-CAM/LIME)."""
    if not PTH_PATH.exists():
        return None
        
    try:
        model = MultiViewAttentionNet(pretrained=False)
        checkpoint = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to("cpu") # Force CPU for deployment
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Preprocess a single PIL image to a normalized tensor."""
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)
    return tensor


def preprocess_views(images: list) -> torch.Tensor:
    """Preprocess a list of 6 PIL images into model input tensor."""
    tensors = [preprocess_image(img) for img in images]
    views = torch.stack(tensors).unsqueeze(0)  # (1, 6, 3, H, W)
    return views


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference_onnx(session, views: torch.Tensor):
    """Run ONNX inference."""
    # Ensure views is on CPU and converted to numpy
    ort_inputs = {session.get_inputs()[0].name: views.cpu().numpy()}
    start = time.time()
    outputs = session.run(None, ort_inputs)
    logits = outputs[0]
    attn_weights = outputs[1] if len(outputs) > 1 else None
    elapsed = (time.time() - start) * 1000

    # Softmax on logits
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs = probs[0]

    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])

    return {
        "predicted_class": pred_class,
        "class_name": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "probabilities": probs,
        "attention_weights": attn_weights[0] if attn_weights is not None else None,
        "inference_time_ms": elapsed,
        "engine": "ONNX (INT8)"
    }

def run_inference_torch(model, views: torch.Tensor):
    """Run PyTorch inference."""
    model.to("cpu")
    views = views.to("cpu")
    start = time.time()
    with torch.no_grad():
        logits, attn_weights = model(views, return_attention=True)
    elapsed = (time.time() - start) * 1000

    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])

    return {
        "predicted_class": pred_class,
        "class_name": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "probabilities": probs,
        "attention_weights": attn_weights[0].cpu().numpy(),
        "inference_time_ms": elapsed,
        "engine": "PyTorch (FP32)"
    }


# ============================================================================
# GRAD-CAM
# ============================================================================

def generate_gradcam(model, views: torch.Tensor, target_class: int):
    """Generate Grad-CAM heatmaps for all views."""
    from src.explainability.gradcam import MultiViewGradCAM

    # Force everything to CPU to avoid device mismatch errors
    model.to("cpu")
    views = views.to("cpu")
    
    gradcam = MultiViewGradCAM(model)
    try:
        # We pass device="cpu" explicitly to be safe
        heatmaps, pred_class, confidence = gradcam.generate(views, target_class=target_class, device=torch.device("cpu"))
    finally:
        gradcam.cleanup()
        
    return heatmaps


def overlay_heatmap(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a heatmap on an image using jet colormap."""
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img_np + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


# ============================================================================
# LIME
# ============================================================================

def generate_lime(model, views: torch.Tensor, view_idx: int, num_samples: int = 300):
    """Generate LIME explanation for a specific view."""
    from src.explainability.lime_explain import MultiViewLIME

    lime_explainer = MultiViewLIME(model)
    explanation = lime_explainer.explain(
        views, view_idx=view_idx, num_samples=num_samples
    )
    return explanation


# ============================================================================
# PDF REPORT
# ============================================================================

def generate_pdf_report(result, images, gradcam_heatmaps=None, note_label="Sample"):
    """Generate a PDF report of the authentication result."""
    from fpdf import FPDF, XPos, YPos
    import io

    def extract_serial(images, note_label):
        for img in images:
            if hasattr(img, 'filename') and img.filename:
                parts = os.path.basename(img.filename).split('_')
                for part in parts:
                    if part.isdigit() and len(part) >= 6:
                        return part
        for part in str(note_label).split('_'):
            if part.isdigit() and len(part) >= 6:
                return part
        return str(note_label)

    note_serial = extract_serial(images, note_label)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "JaalTaka Authentication Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(0, 8, "Developed by Shah Nawaz", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 16)
    verdict_color = (220, 53, 69) if result["class_name"] == "Fake" else (40, 167, 69)
    pdf.set_text_color(*verdict_color)
    pdf.cell(0, 12, f"Verdict: {result['class_name']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Confidence: {result['confidence']:.1%}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"Inference Time: {result['inference_time_ms']:.0f} ms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"Note Serial: {note_serial}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Class Probabilities:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    for i, name in enumerate(CLASS_NAMES):
        pdf.cell(0, 7, f"  {name}: {result['probabilities'][i]:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Attention Weights:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    for i, name in enumerate(VIEW_NAMES):
        pdf.cell(0, 7, f"  {name}: {result['attention_weights'][i]:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Captured Views:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for i, (img, name) in enumerate(zip(images, VIEW_NAMES)):
        if i % 3 == 0 and i > 0:
            pdf.add_page()
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"View {i+1}: {name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        buf = io.BytesIO()
        img.resize((IMAGE_SIZE, IMAGE_SIZE)).save(buf, format="PNG")
        buf.seek(0)
        pdf.image(buf, w=60)
        pdf.ln(3)

    if gradcam_heatmaps is not None:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Grad-CAM Heatmaps", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(3)
        views_tensor = preprocess_views(images)
        for i, (heatmap, name) in enumerate(zip(gradcam_heatmaps, VIEW_NAMES)):
            if i % 3 == 0 and i > 0:
                pdf.add_page()
            img_np = denormalize(views_tensor[0, i])
            overlay = overlay_heatmap(img_np, heatmap)
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.imshow(overlay)
            ax.set_title(name, fontsize=10)
            ax.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, dpi=100, bbox_inches="tight", format="png")
            plt.close(fig)
            buf.seek(0)
            pdf.image(buf, w=60)
            pdf.ln(3)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


# ============================================================================
# SAMPLE DATA
# ============================================================================

def load_sample_note(note_type: str, note_id: str):
    """Load a sample note's 6 views from disk."""
    if note_type == "fake":
        note_dir = SAMPLE_DIRS["fake"] / note_id
    else:
        note_dir = SAMPLE_DIRS["real"] / note_id

    if not note_dir.exists():
        return None

    images = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        for f in sorted(note_dir.glob(f"*{ext}")):
            images.append(Image.open(f).convert("RGB"))
            if len(images) >= 6:
                break
        if len(images) >= 6:
            break

    return images if len(images) == 6 else None


def get_sample_notes():
    """Get list of available sample notes."""
    samples = {"fake": [], "real": []}
    for note_type, base_dir in SAMPLE_DIRS.items():
        if base_dir.exists():
            for d in sorted(base_dir.iterdir()):
                if d.is_dir():
                    count = sum(1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
                    if count >= 6:
                        samples[note_type].append(d.name)
    return samples


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown("# 💵")
        st.title("🔍 JaalTaka")
        st.caption("Multi-View Counterfeit Detection")
        st.markdown("---")

        mode = st.radio(
            "Mode",
            ["Single Analysis", "Batch Analysis", "Demo (Samples)"],
            index=0,
        )

        st.markdown("---")
        st.markdown("### Settings")
        enable_gradcam = st.checkbox("Enable Grad-CAM", value=True)
        enable_lime = st.checkbox("Enable LIME", value=False)
        enable_shap = st.checkbox("Show SHAP Attribution", value=True)

        if enable_lime:
            lime_samples = st.slider("LIME perturbations", 100, 1000, 300, 100)
        else:
            lime_samples = 300

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#888; font-size:12px;'>"
            "Developed by <b>Shah Nawaz</b><br>"
            "Multi-View Attention Transformer<br>"
            "99.04% Accuracy | 0.9925 ROC-AUC"
            "</div>",
            unsafe_allow_html=True,
        )

    return mode, enable_gradcam, enable_lime, enable_shap, lime_samples


def render_header():
    """Render the main header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 JaalTaka: Banknote Authentication")
        st.markdown(
            "**Multi-View Counterfeit Banknote Detection** with Explainable AI  \n"
            "Upload 6 views of a banknote for instant authentication using "
            "Transformer-based attention fusion."
        )
    with col2:
        st.markdown(
            "<div style='text-align:right; padding:20px;'>"
            "<span style='font-size:14px; color:#666;'>Developed by</span><br>"
            "<span style='font-size:20px; font-weight:bold; color:#1a73e8;'>Shah Nawaz</span>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_upload_section():
    """Render the 6-image upload section."""
    st.markdown("### 📸 Upload 6 Views of the Banknote")
    st.info(
        "Upload exactly **6 images** in order: "
        "Front, Back, Watermark, Security Thread, Serial Number, Hologram/UV"
    )

    uploaded = st.file_uploader(
        "Choose 6 banknote images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="upload_views",
    )

    if uploaded and len(uploaded) != 6:
        st.warning(f"Please upload exactly 6 images. You uploaded {len(uploaded)}.")
        return None

    if uploaded and len(uploaded) == 6:
        images = [Image.open(f).convert("RGB") for f in uploaded]

        cols = st.columns(6)
        for i, (col, img, name) in enumerate(zip(cols, images, VIEW_NAMES)):
            with col:
                st.image(img, caption=name, use_container_width=True)

        return images

    return None


def render_result(result):
    """Render authentication result with animated metrics."""
    is_fake = result["class_name"] == "Fake"
    color = "#dc3545" if is_fake else "#28a745"
    icon = "⚠️" if is_fake else "✅"

    st.markdown("### 📊 Authentication Result")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div style='text-align:center; padding:20px; "
            f"background:{'#fff5f5' if is_fake else '#f0fff4'}; "
            f"border-radius:12px; border:2px solid {color};'>"
            f"<span style='font-size:40px;'>{icon}</span><br>"
            f"<span style='font-size:24px; font-weight:bold; color:{color};'>"
            f"{result['class_name']}</span></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    with col3:
        st.metric("Inference Time", f"{result['inference_time_ms']:.0f} ms")
    with col4:
        st.metric("Engine", result["engine"])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["probabilities"][1] * 100,
        title={"text": "Real Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#28a745"},
            "steps": [
                {"range": [0, 30], "color": "#fff5f5"},
                {"range": [30, 70], "color": "#fffbea"},
                {"range": [70, 100], "color": "#f0fff4"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)


def render_attention_weights(result):
    """Render attention weight bar chart."""
    st.markdown("### 🎯 Attention Weights")
    st.caption("How much the model focused on each view")

    weights = result["attention_weights"]
    if weights is None:
        st.warning("Attention weights not available for this model.")
        return

    fig = go.Figure(go.Bar(
        x=VIEW_NAMES,
        y=weights,
        marker_color=["#1a73e8" if w == max(weights) else "#a8d5f2" for w in weights],
        text=[f"{w:.3f}" for w in weights],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="Attention Weight",
        height=350,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_gradcam(model, views, result):
    """Render Grad-CAM heatmap overlays."""
    if model is None:
        st.warning("PyTorch model (`.pth`) not found. Grad-CAM requires the PyTorch model.")
        return None

    st.markdown("### 🔥 Grad-CAM Heatmaps")
    st.caption("Regions the model focuses on for its decision")

    with st.spinner("Generating Grad-CAM heatmaps..."):
        try:
            heatmaps = generate_gradcam(model, views, result["predicted_class"])
            cols = st.columns(6)
            for i, (col, heatmap, name) in enumerate(zip(cols, heatmaps, VIEW_NAMES)):
                with col:
                    img_np = denormalize(views[0, i])
                    overlay = overlay_heatmap(img_np, heatmap)
                    st.image(overlay, caption=name, use_column_width=True, clamp=True)
            return heatmaps
        except Exception as e:
            st.error(f"Grad-CAM generation failed: {e}")
            return None


def render_lime(model, views, result, num_samples):
    """Render LIME explanations."""
    if model is None:
        st.warning("PyTorch model (`.pth`) not found. LIME requires the PyTorch model.")
        return

    st.markdown("### 🧩 LIME Explanations")
    st.caption("Superpixel-based local explanations for each view")

    view_idx = st.selectbox("Select view to explain", range(6),
                            format_func=lambda x: VIEW_NAMES[x])

    with st.spinner(f"Running LIME for {VIEW_NAMES[view_idx]} ({num_samples} perturbations)..."):
        try:
            explanation = generate_lime(model, views, view_idx, num_samples)
            img_np = denormalize(views[0, view_idx])
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_np, caption="Original", use_column_width=True, clamp=True)
            with col2:
                temp, mask = explanation.get_image_and_mask(
                    result["predicted_class"],
                    positive_only=False,
                    num_features=5,
                    hide_rest=False,
                )
                display = temp / 255.0 if temp.max() > 1 else temp
                st.image(display, caption="LIME Explanation", use_column_width=True, clamp=True)
        except Exception as e:
            st.error(f"LIME explanation failed: {e}")


def render_shap():
    """Render SHAP view-level attribution."""
    st.markdown("### 📈 SHAP View Attribution")
    st.caption("Global view-level importance from dataset-wide Shapley analysis (30 samples)")

    fig = go.Figure(go.Bar(
        y=VIEW_NAMES,
        x=SHAP_VALUES,
        orientation="h",
        marker_color=["#4ecdc4" if v >= 0 else "#ff6b6b" for v in SHAP_VALUES],
        text=[f"{v:.4f}" for v in SHAP_VALUES],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="SHAP Value (contribution to prediction)",
        height=300,
        margin=dict(t=20, b=40, l=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Key Finding:** Views 1 (Front) and 6 (Hologram/UV) contribute the most "
        "(SHAP ≈ 0.30), while View 2 (Back) contributes the least (SHAP ≈ 0.12). "
        "The front face and holographic features carry the strongest authentication signals."
    )


def render_batch_analysis(onnx_session, torch_model, enable_gradcam):
    """Render batch analysis mode."""
    st.markdown("### 📦 Batch Analysis")
    st.info("Upload a ZIP file of notes (each folder = 6 images) **or** select from sample notes.")

    uploaded_zip = st.file_uploader("Upload ZIP of notes (each folder = 6 images)", type=["zip"])
    results_list = []
    notes_to_analyze = []
    zip_mode = False

    if uploaded_zip is not None:
        import zipfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            for root, dirs, files in os.walk(tmpdir):
                img_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
                if len(img_files) >= 6:
                    img_files = sorted(img_files)[:6]
                    img_paths = [os.path.join(root, f) for f in img_files]
                    notes_to_analyze.append((os.path.basename(root), img_paths))
        zip_mode = True
    else:
        samples = get_sample_notes()
        all_notes = []
        for ntype, names in samples.items():
            for name in names[:10]:
                all_notes.append((ntype, name))
        if not all_notes:
            st.warning("No sample notes found in the dataset directories.")
            return
        st.markdown(f"Found **{len(all_notes)}** sample notes (showing first 10 per class).")
        num_to_analyze = st.slider("Number of notes to analyze", 1, min(20, len(all_notes)), 5)
        notes_to_analyze = all_notes[:num_to_analyze]

    if st.button("🚀 Run Batch Analysis", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        for idx, note in enumerate(notes_to_analyze):
            if zip_mode:
                note_id, img_paths = note
                images = [Image.open(p).convert("RGB") for p in img_paths]
                true_label = "Unknown"
            else:
                ntype, name = note
                images = load_sample_note(ntype, name)
                note_id = name
                true_label = ntype.capitalize()
            if images is None or len(images) != 6:
                continue
            views = preprocess_views(images)
            
            # Use ONNX if available, else Torch
            if onnx_session:
                result = run_inference_onnx(onnx_session, views)
            elif torch_model:
                result = run_inference_torch(torch_model, views)
            else:
                st.error("No model available for inference.")
                return

            result["note_id"] = note_id
            result["true_label"] = true_label
            result["correct"] = (
                (result["class_name"] == "Real" and true_label.lower() == "real") or
                (result["class_name"] == "Fake" and true_label.lower() == "fake")
            ) if true_label in ["Real", "Fake"] else "-"
            results_list.append(result)
            progress.progress((idx + 1) / len(notes_to_analyze))
        status.text("Batch analysis complete!")

    if results_list:
        correct = sum(1 for r in results_list if r["correct"] == True)
        total = len(results_list)
        accuracy = correct / total if total else 0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("Correct", f"{correct}/{total}")
        with col3:
            avg_time = np.mean([r["inference_time_ms"] for r in results_list])
            st.metric("Avg Inference Time", f"{avg_time:.0f} ms")

        import pandas as pd
        df = pd.DataFrame([{
            "Note": r["note_id"],
            "True": r["true_label"],
            "Predicted": r["class_name"],
            "Confidence": f"{r['confidence']:.2%}",
            "Correct": "✅" if r["correct"] == True else ("❌" if r["correct"] == False else "-"),
            "Time (ms)": f"{r['inference_time_ms']:.0f}",
        } for r in results_list])
        st.dataframe(df, use_container_width=True)

        if st.button("📄 Download Batch Report"):
            try:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 20)
                pdf.cell(0, 15, "JaalTaka Batch Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(10)
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 10, f"Summary: {correct}/{total} correct ({accuracy:.1%})", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(40, 8, "Note", border=1)
                pdf.cell(25, 8, "True", border=1)
                pdf.cell(30, 8, "Predicted", border=1)
                pdf.cell(30, 8, "Confidence", border=1)
                pdf.cell(25, 8, "Correct", border=1)
                pdf.cell(25, 8, "Time", border=1)
                pdf.ln()
                pdf.set_font("Helvetica", "", 9)
                for r in results_list:
                    pdf.cell(40, 7, str(r["note_id"])[:15], border=1)
                    pdf.cell(25, 7, str(r["true_label"]), border=1)
                    pdf.cell(30, 7, str(r["class_name"]), border=1)
                    pdf.cell(30, 7, f"{r['confidence']:.2%}", border=1)
                    pdf.cell(25, 7, "Yes" if r["correct"] == True else "No", border=1)
                    pdf.cell(25, 7, f"{r['inference_time_ms']:.0f} ms", border=1)
                    pdf.ln()
                buf = io.BytesIO()
                pdf.output(buf)
                buf.seek(0)
                st.download_button("Download PDF", data=buf, file_name="batch_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Failed to generate report: {e}")


def render_demo_mode(onnx_session, torch_model, enable_gradcam, enable_lime, enable_shap, lime_samples):
    """Render demo mode."""
    st.markdown("### 🎮 Demo Mode")
    samples = get_sample_notes()
    col1, col2 = st.columns(2)
    with col1:
        note_type = st.selectbox("Note Type", ["fake", "real"])
    with col2:
        available = samples.get(note_type, [])
        if not available:
            st.warning(f"No {note_type} notes found.")
            return
        note_id = st.selectbox("Note ID", available[:20])

    if st.button("🔍 Analyze Selected Note", type="primary"):
        images = load_sample_note(note_type, note_id)
        if images is None:
            st.error("Could not load note images.")
            return
        cols = st.columns(6)
        for i, (col, img, name) in enumerate(zip(cols, images, VIEW_NAMES)):
            with col:
                st.image(img, caption=name, use_container_width=True)
        views = preprocess_views(images)
        with st.spinner("Running inference..."):
            if onnx_session:
                result = run_inference_onnx(onnx_session, views)
            elif torch_model:
                result = run_inference_torch(torch_model, views)
            else:
                st.error("No model available.")
                return
        render_result(result)
        render_attention_weights(result)
        gradcam_heatmaps = render_gradcam(torch_model, views, result) if enable_gradcam else None
        if enable_lime: render_lime(torch_model, views, result, lime_samples)
        if enable_shap: render_shap()
        pdf_buf = generate_pdf_report(result, images, gradcam_heatmaps, f"{note_type}/{note_id}")
        st.download_button("📄 Download PDF Report", data=pdf_buf, file_name=f"jaaltaka_{note_id}.pdf", mime="application/pdf")


def main():
    mode, enable_gradcam, enable_lime, enable_shap, lime_samples = render_sidebar()
    render_header()
    st.markdown("---")

    onnx_session = load_onnx_model()
    torch_model = load_torch_model()

    if not onnx_session and not torch_model:
        st.error("No models found! Please ensure `jaaltaka_attention_int8.onnx` or `attention_transformer_best.pth` exists in the `outputs` directory.")
        st.stop()
    
    if onnx_session:
        st.success("✅ ONNX model loaded (Inference optimized)")
    if torch_model:
        st.success("✅ PyTorch model loaded (XAI enabled)")
    else:
        st.info("ℹ️ Grad-CAM and LIME are disabled because the PyTorch model (.pth) was not found. Using ONNX for fast authentication.")

    if mode == "Single Analysis":
        images = render_upload_section()
        if images is not None:
            if st.button("🔍 Authenticate Banknote", type="primary"):
                views = preprocess_views(images)
                with st.spinner("Running inference..."):
                    if onnx_session:
                        result = run_inference_onnx(onnx_session, views)
                    else:
                        result = run_inference_torch(torch_model, views)
                render_result(result)
                render_attention_weights(result)
                gradcam_heatmaps = render_gradcam(torch_model, views, result) if enable_gradcam else None
                if enable_lime: render_lime(torch_model, views, result, lime_samples)
                if enable_shap: render_shap()
                pdf_buf = generate_pdf_report(result, images, gradcam_heatmaps)
                st.download_button("📄 Download PDF Report", data=pdf_buf, file_name="jaaltaka_report.pdf", mime="application/pdf")
    elif mode == "Batch Analysis":
        render_batch_analysis(onnx_session, torch_model, enable_gradcam)
    elif mode == "Demo (Samples)":
        render_demo_mode(onnx_session, torch_model, enable_gradcam, enable_lime, enable_shap, lime_samples)

    st.markdown("---")
    st.markdown("<div style='text-align:center; padding:20px; color:#888;'><b>JaalTaka</b> — Multi-View Counterfeit Banknote Detection System<br>Developed by <b style='color:#1a73e8;'>Shah Nawaz</b></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
