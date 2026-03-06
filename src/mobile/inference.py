"""
Phase 8 — Mobile Inference Pipeline
======================================
Simulates the end-to-end mobile inference workflow:
  1. Camera capture → 2. Note detection → 3. Region segmentation (6 regions)
  → 4. Model inference → 5. Prediction → 6. Explainability overlay

This module provides the pipeline logic that will be replicated in
the Flutter application (Phase 9).
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    NUM_VIEWS, CLASS_NAMES, USE_AMP, EXPORT_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# REGION SEGMENTATION CONFIG
# ============================================================================
# Default region definitions for Bangladeshi banknotes.
# These represent approximate bounding boxes (relative coords) for the
# 6 key security features that the JaalTaka dataset captures:
#   1. Watermark area (left)
#   2. Security thread (center-left)
#   3. Microprinting zone (center)
#   4. Intaglio print area (center-right)
#   5. Hologram / color-shifting ink (right)
#   6. Serial number region (bottom)

DEFAULT_REGIONS = [
    {"name": "watermark",       "x": 0.00, "y": 0.10, "w": 0.25, "h": 0.80},
    {"name": "security_thread", "x": 0.20, "y": 0.05, "w": 0.15, "h": 0.90},
    {"name": "microprint",      "x": 0.35, "y": 0.20, "w": 0.20, "h": 0.60},
    {"name": "intaglio",        "x": 0.50, "y": 0.10, "w": 0.25, "h": 0.80},
    {"name": "hologram",        "x": 0.75, "y": 0.15, "w": 0.25, "h": 0.70},
    {"name": "serial_number",   "x": 0.10, "y": 0.75, "w": 0.80, "h": 0.25},
]


# ============================================================================
# NOTE DETECTION (simplified)
# ============================================================================

def detect_banknote(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect and crop the banknote from a camera-captured image.

    Uses edge detection + contour finding for a robust bounding rectangle.

    Parameters
    ----------
    image : np.ndarray (H, W, 3) BGR

    Returns
    -------
    cropped : np.ndarray or None if no banknote detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to close gaps
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        logger.warning("No banknote detected in image")
        return None

    # Largest contour by area
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.intp)

    # Perspective transform to straighten
    width, height = int(rect[1][0]), int(rect[1][1])
    if width < height:
        width, height = height, width

    # Ensure minimum size
    if width < 100 or height < 50:
        return None

    dst_pts = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype=np.float32)

    # Sort box points: top-left, top-right, bottom-right, bottom-left
    box_sorted = _order_points(box.astype(np.float32))
    M = cv2.getPerspectiveTransform(box_sorted, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ============================================================================
# REGION SEGMENTATION
# ============================================================================

def segment_regions(
    banknote_image: np.ndarray,
    regions: List[Dict] = DEFAULT_REGIONS,
) -> List[np.ndarray]:
    """
    Crop the 6 security regions from a detected banknote.

    Parameters
    ----------
    banknote_image : (H, W, 3)
    regions : list of dicts with x, y, w, h (relative coordinates)

    Returns
    -------
    crops : list of 6 np.ndarray images
    """
    H, W = banknote_image.shape[:2]
    crops = []

    for region in regions:
        x = int(region["x"] * W)
        y = int(region["y"] * H)
        w = int(region["w"] * W)
        h = int(region["h"] * H)

        # Clamp to image bounds
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = min(w, W - x)
        h = min(h, H - y)

        crop = banknote_image[y:y+h, x:x+w]
        crops.append(crop)

    return crops


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_regions(
    crops: List[np.ndarray],
    target_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """
    Preprocess 6 region crops into a model-ready tensor.

    Returns
    -------
    tensor : (1, 6, 3, H, W)
    """
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    views = []
    for crop in crops:
        # BGR → RGB
        if crop.shape[-1] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop

        # Resize
        resized = cv2.resize(crop_rgb, (target_size, target_size))

        # Normalize
        normalized = (resized.astype(np.float32) / 255.0 - mean) / std

        # HWC → CHW
        tensor = torch.tensor(normalized).permute(2, 0, 1).float()
        views.append(tensor)

    # Pad if fewer than 6 regions
    while len(views) < NUM_VIEWS:
        views.append(torch.zeros(3, target_size, target_size))

    return torch.stack(views).unsqueeze(0)  # (1, 6, 3, H, W)


# ============================================================================
# INFERENCE
# ============================================================================

def run_pytorch_inference(
    model: torch.nn.Module,
    views_tensor: torch.Tensor,
) -> Dict:
    """
    Run PyTorch model inference.

    Returns
    -------
    dict with: prediction, confidence, probabilities, latency_ms
    """
    model.eval()
    views_tensor = views_tensor.to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        from torch.amp import autocast
        with autocast(device_type="cuda", enabled=USE_AMP):
            logits = model(views_tensor)
    latency = (time.perf_counter() - t0) * 1000

    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))

    return {
        "prediction": CLASS_NAMES[pred_class],
        "class_id": pred_class,
        "confidence": float(probs[pred_class]),
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))
        },
        "latency_ms": latency,
    }


def run_onnx_inference(
    onnx_path: Path,
    views_tensor: torch.Tensor,
) -> Dict:
    """Run ONNX Runtime inference."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    input_data = views_tensor.cpu().numpy()

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: input_data})
    latency = (time.perf_counter() - t0) * 1000

    logits = outputs[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()  # softmax
    pred_class = int(np.argmax(probs))

    return {
        "prediction": CLASS_NAMES[pred_class],
        "class_id": pred_class,
        "confidence": float(probs[pred_class]),
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))
        },
        "latency_ms": latency,
    }


# ============================================================================
# FULL MOBILE PIPELINE
# ============================================================================

def mobile_pipeline(
    image_path: str,
    model: Optional[torch.nn.Module] = None,
    onnx_path: Optional[Path] = None,
) -> Dict:
    """
    Full end-to-end mobile inference pipeline.

    1. Load image (simulating camera capture)
    2. Detect banknote
    3. Segment 6 security regions
    4. Preprocess
    5. Run inference
    6. Return result

    Parameters
    ----------
    image_path : str
        Path to captured banknote photo.
    model : nn.Module, optional
        PyTorch model (used if onnx_path is None).
    onnx_path : Path, optional
        ONNX model path (preferred for mobile).
    """
    total_t0 = time.perf_counter()

    # 1. Load
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Could not load image: {image_path}"}

    # 2. Detect banknote
    detected = detect_banknote(image)
    if detected is None:
        # Fallback: use full image
        logger.warning("Banknote detection failed, using full image")
        detected = image

    # 3. Segment regions
    crops = segment_regions(detected)

    # 4. Preprocess
    views_tensor = preprocess_regions(crops)

    # 5. Inference
    if onnx_path is not None:
        result = run_onnx_inference(onnx_path, views_tensor)
    elif model is not None:
        result = run_pytorch_inference(model, views_tensor)
    else:
        return {"error": "No model provided"}

    total_latency = (time.perf_counter() - total_t0) * 1000
    result["total_pipeline_ms"] = total_latency

    logger.info(
        f"  Pipeline result: {result['prediction']} "
        f"({result['confidence']:.2%}) in {total_latency:.0f}ms"
    )

    return result
