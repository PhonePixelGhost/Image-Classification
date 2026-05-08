import onnx
import onnx.shape_inference
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# --- Monkey Patch onnx.shape_inference to bypass strict checks ---
original_infer_shapes_path = onnx.shape_inference.infer_shapes_path

def patched_infer_shapes_path(model_path, output_path=None, check_type=False, strict_mode=False, data_prop=False):
    try:
        # Run in non-strict mode
        return original_infer_shapes_path(model_path, output_path, check_type, False, data_prop)
    except Exception:
        if output_path:
            import shutil
            shutil.copy(model_path, output_path)

onnx.shape_inference.infer_shapes_path = patched_infer_shapes_path
# --------------------------------------------------------------------------

model_path = "models/resnet18.onnx"
quantized_path = "models/resnet18_quantized.onnx"

print(f"Quantizing model: {model_path}...")
try:
    quantize_dynamic(
        model_input=model_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8,
        extra_options={
            'EnableShapeInference': False,
            'DefaultTensorType': onnx.TensorProto.FLOAT  # <--- เพิ่มตัวนี้เพื่อแก้ Error ล่าสุด
        }
    )
except Exception as e:
    print(f"Quantization failed: {e}")

if os.path.exists(quantized_path):
    print(f"Success: {quantized_path} created. Size: {os.path.getsize(quantized_path)/1e6:.2f} MB")
else:
    # Try one more time with a very minimal set of options
    print("Trying one last alternative...")
    quantize_dynamic(
        model_input=model_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8,
        # Minimal options
    )

if os.path.exists(quantized_path):
    print(f"Success on second attempt: {quantized_path}")
