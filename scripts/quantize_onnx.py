"""
ONNX INT8 Quantization Script
================================
Quantizes the JaalTaka ONNX model to INT8 for smaller size and faster inference.
Developed by Shah Nawaz.

Usage:
    python scripts/quantize_onnx.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import os


def quantize_model():
    project_root = Path(__file__).resolve().parent.parent
    input_model = project_root / "outputs" / "exports" / "jaaltaka_attention.onnx"
    output_model = project_root / "outputs" / "exports" / "jaaltaka_attention_int8.onnx"

    if not input_model.exists():
        print(f"ERROR: Input model not found at {input_model}")
        return

    print(f"Input model: {input_model}")
    print(f"Input size: {input_model.stat().st_size / 1024 / 1024:.1f} MB")

    # Dynamic INT8 quantization
    print("\nRunning INT8 dynamic quantization (optimized for compatibility)...")
    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=True,
    )

    print(f"\nOutput model: {output_model}")
    print(f"Output size: {output_model.stat().st_size / 1024 / 1024:.1f} MB")

    reduction = 1 - (output_model.stat().st_size / input_model.stat().st_size)
    print(f"Size reduction: {reduction * 100:.1f}%")

    # Validate quantized model
    print("\nValidating quantized model...")
    model = onnx.load(str(output_model))
    onnx.checker.check_model(model)
    print("Validation PASSED")

    # Also copy to flutter assets if directory exists
    flutter_assets = project_root / "flutter_app" / "assets" / "models"
    if flutter_assets.exists():
        import shutil
        dest = flutter_assets / "jaaltaka_attention_int8.onnx"
        shutil.copy2(output_model, dest)
        print(f"\nCopied to Flutter assets: {dest}")

    print("\nDone! You can update OnnxService to use the INT8 model for ~3x faster inference.")


if __name__ == "__main__":
    quantize_model()
