"""
ONNX FP16 Conversion Script
==============================
Converts the JaalTaka ONNX model to FP16 (Half Precision).
This reduces size by 50% (~60MB) while maintaining high accuracy and 
ensuring better compatibility than INT8 quantization.

Usage:
    python scripts/convert_fp16.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnx
from onnxconverter_common.float16 import convert_float_to_float16
import os

def convert_to_fp16():
    project_root = Path(__file__).resolve().parent.parent
    input_model = project_root / "outputs" / "exports" / "jaaltaka_attention.onnx"
    output_model = project_root / "outputs" / "exports" / "jaaltaka_attention_fp16.onnx"

    if not input_model.exists():
        print(f"ERROR: Input model not found at {input_model}")
        return

    print(f"Input model: {input_model}")
    print(f"Input size: {input_model.stat().st_size / 1024 / 1024:.1f} MB")

    print("\nConverting to FP16...")
    try:
        model = onnx.load(str(input_model))
        # Use onnxmltools for reliable FP16 conversion
        model_fp16 = convert_float_to_float16(model)
        onnx.save(model_fp16, str(output_model))
    except Exception as e:
        print(f"ERROR during conversion: {e}")
        return

    print(f"\nOutput model: {output_model}")
    print(f"Output size: {output_model.stat().st_size / 1024 / 1024:.1f} MB")

    # Validate model
    print("\nValidating FP16 model...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(output_model), providers=['CPUExecutionProvider'])
        print("Validation PASSED (Model loaded successfully)")
    except Exception as e:
        print(f"Validation FAILED: {e}")

    # Copy to flutter assets
    flutter_assets = project_root / "flutter_app" / "assets" / "models"
    if flutter_assets.exists():
        import shutil
        dest = flutter_assets / "jaaltaka_attention_fp16.onnx"
        shutil.copy2(output_model, dest)
        print(f"\nCopied to Flutter assets: {dest}")

if __name__ == "__main__":
    convert_to_fp16()
