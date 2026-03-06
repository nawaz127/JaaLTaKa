"""
Phase 7 — Model Compression & Export
=======================================
Export PyTorch → ONNX → TFLite (INT8 quantized).
Benchmark model size, latency, and memory usage.
"""

import time
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICE, IMAGE_SIZE, NUM_VIEWS, EXPORT_DIR, CHECKPOINT_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ONNX EXPORT
# ============================================================================

def export_to_onnx(
    model: nn.Module,
    export_path: Optional[Path] = None,
    opset_version: int = 17,
) -> Path:
    """
    Export PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        Trained multi-view model.
    export_path : Path
        Output .onnx file path.

    Returns
    -------
    Path to saved ONNX model.
    """
    if export_path is None:
        export_path = EXPORT_DIR / "jaaltaka_model.onnx"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()

    # Dummy input: (1, 6, 3, 224, 224)
    dummy_input = torch.randn(1, NUM_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        opset_version=opset_version,
        input_names=["views"],
        output_names=["logits"],
        dynamic_axes={
            "views": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
    logger.info(f"  ONNX export -> {export_path} ({file_size_mb:.1f} MB)")
    return export_path


def validate_onnx(onnx_path: Path) -> bool:
    """Validate ONNX model structure."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info("  ONNX model validation: PASSED")
        return True
    except Exception as e:
        logger.error(f"  ONNX validation failed: {e}")
        return False


# ============================================================================
# ONNX RUNTIME BENCHMARK
# ============================================================================

def benchmark_onnx(
    onnx_path: Path,
    num_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Benchmark ONNX Runtime inference.

    Returns dict with latency stats and model size.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed")
        return {}

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    dummy = np.random.randn(1, NUM_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    stats = {
        "onnx_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
    }

    logger.info(f"  ONNX Benchmark:")
    for k, v in stats.items():
        logger.info(f"    {k}: {v:.2f}")

    return stats


# ============================================================================
# TFLITE CONVERSION
# ============================================================================

def convert_to_tflite(
    onnx_path: Path,
    tflite_path: Optional[Path] = None,
    quantize_int8: bool = True,
    calibration_data: Optional[np.ndarray] = None,
) -> Path:
    """
    Convert ONNX model to TensorFlow Lite with optional INT8 quantization.

    Parameters
    ----------
    onnx_path : Path
    tflite_path : Path
    quantize_int8 : bool
        Apply INT8 post-training quantization.
    calibration_data : np.ndarray, optional
        Representative data for quantization calibration.

    Returns
    -------
    Path to saved .tflite model.
    """
    if tflite_path is None:
        suffix = "_int8" if quantize_int8 else ""
        tflite_path = EXPORT_DIR / f"jaaltaka_model{suffix}.tflite"

    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError as e:
        logger.error(f"Missing dependency for TFLite conversion: {e}")
        logger.info("Install: pip install onnx-tf tensorflow")
        return tflite_path

    # Step 1: ONNX → TF SavedModel
    logger.info("  Converting ONNX -> TensorFlow SavedModel ...")
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    saved_model_dir = str(EXPORT_DIR / "tf_saved_model")
    tf_rep.export_graph(saved_model_dir)

    # Step 2: TF SavedModel → TFLite
    logger.info("  Converting TF SavedModel -> TFLite ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize_int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        # Representative dataset for calibration
        if calibration_data is None:
            calibration_data = np.random.randn(
                100, NUM_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE
            ).astype(np.float32)

        def representative_dataset():
            for i in range(min(100, len(calibration_data))):
                yield [calibration_data[i:i+1]]

        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    logger.info(f"  TFLite export -> {tflite_path} ({file_size_mb:.1f} MB)")
    return tflite_path


# ============================================================================
# TFLITE BENCHMARK
# ============================================================================

def benchmark_tflite(
    tflite_path: Path,
    num_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Benchmark TFLite inference latency and model size."""
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow not installed for TFLite benchmark")
        return {}

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    dummy = np.random.randn(*input_shape).astype(input_dtype)

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    stats = {
        "tflite_size_mb": os.path.getsize(tflite_path) / (1024 * 1024),
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
    }

    logger.info(f"  TFLite Benchmark:")
    for k, v in stats.items():
        logger.info(f"    {k}: {v:.2f}")

    return stats


# ============================================================================
# FULL COMPRESSION PIPELINE
# ============================================================================

def run_compression_pipeline(
    model: nn.Module,
    experiment_name: str = "jaaltaka",
) -> Dict[str, dict]:
    """
    Full Phase 7 pipeline: PyTorch → ONNX → TFLite (INT8).
    Benchmarks all formats.
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 7 -- Model Compression Pipeline")
    logger.info("=" * 60)

    results = {}

    # --- PyTorch model size ---
    pytorch_path = EXPORT_DIR / f"{experiment_name}.pth"
    torch.save(model.state_dict(), pytorch_path)
    pytorch_size = os.path.getsize(pytorch_path) / (1024 * 1024)
    results["pytorch"] = {"size_mb": pytorch_size}
    logger.info(f"  PyTorch model: {pytorch_size:.1f} MB")

    # --- ONNX ---
    onnx_path = export_to_onnx(model, EXPORT_DIR / f"{experiment_name}.onnx")
    valid = validate_onnx(onnx_path)
    onnx_bench = benchmark_onnx(onnx_path)
    results["onnx"] = onnx_bench

    # --- TFLite ---
    try:
        tflite_path = convert_to_tflite(onnx_path, quantize_int8=False)
        tflite_bench = benchmark_tflite(tflite_path)
        results["tflite_fp32"] = tflite_bench
    except Exception as e:
        logger.warning(f"  TFLite FP32 conversion failed: {e}")

    try:
        tflite_int8_path = convert_to_tflite(onnx_path, quantize_int8=True)
        tflite_int8_bench = benchmark_tflite(tflite_int8_path)
        results["tflite_int8"] = tflite_int8_bench
    except Exception as e:
        logger.warning(f"  TFLite INT8 conversion failed: {e}")

    # --- Summary ---
    logger.info("\n--- Compression Summary ---")
    for fmt, stats in results.items():
        logger.info(f"  {fmt}: {stats}")

    return results
