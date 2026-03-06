import onnxruntime as ort
import numpy as np
from pathlib import Path

def test_model():
    model_path = Path("outputs/exports/jaaltaka_attention_int8.onnx")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path}")
    try:
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        print("Model loaded successfully!")
        
        # Test inference with dummy input
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        print(f"Input shape from ONNX: {input_shape}")
        
        # Replace dynamic dimensions with 1 for testing
        test_shape = [s if isinstance(s, int) else 1 for s in input_shape]
        print(f"Using test shape: {test_shape}")
        
        dummy_input = np.random.randn(*test_shape).astype(np.float32)
        
        outputs = session.run(None, {input_name: dummy_input})
        print(f"Inference successful! Output shape: {outputs[0].shape}")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_model()
