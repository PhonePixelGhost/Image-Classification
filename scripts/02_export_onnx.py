import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
import os

model_id = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_id)
model = ResNetForImageClassification.from_pretrained(model_id).eval()

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Use dummy input for tracing
dummy = torch.randn(1, 3, 224, 224)

# Export using the legacy approach (Tracing) which is more stable for quantization tools
print("Exporting model to ONNX using legacy tracing...")
torch.onnx.export(
    model,
    dummy,
    "models/resnet18.onnx",
    export_params=True,
    opset_version=18,  # Use Opset 11 for better compatibility with quantization
    do_constant_folding=True,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
)

print("ONNX exported successfully to models/resnet18.onnx")
