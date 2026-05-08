import onnxruntime as ort
import numpy as np
import time
import os
from PIL import Image
from transformers import AutoImageProcessor

# Load extractor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

# Test both models
models = {
    "ONNX": "models/resnet18.onnx",
    "ONNX Quantized": "models/resnet18_quantized.onnx"
}

# Create test image if not exists
if not os.path.exists("test.jpg"):
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    img.save("test.jpg")

img = Image.open("test.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="np")
pixel_values = inputs["pixel_values"].astype(np.float32)

for name, model_path in models.items():
    if not os.path.exists(model_path):
        print(f"Skipping {name}: {model_path} not found")
        continue
        
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = session.run(["logits"], {"pixel_values": pixel_values})
        times.append(time.perf_counter() - t0)
    
    print(f"\n{name}:")
    print(f"  Avg Latency: {sum(times)/len(times)*1000:.2f} ms")
    print(f"  P95 Latency: {sorted(times)[94]*1000:.2f} ms")
    print(f"  File Size: {os.path.getsize(model_path)/1e6:.2f} MB")
