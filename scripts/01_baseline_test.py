from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import time
import os
from PIL import Image

model_id = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_id)
model = ResNetForImageClassification.from_pretrained(model_id)
model.eval()

# Create test image if not exists
if not os.path.exists("test.jpg"):
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    img.save("test.jpg")

# Measure Baseline Latency (100 runs)
img = Image.open("test.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt")

times = []
with torch.no_grad():
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(**inputs)
        times.append(time.perf_counter() - t0)

print(f"Baseline Latency (avg): {sum(times)/len(times)*1000:.2f} ms")
print(f"P95 Latency: {sorted(times)[94]*1000:.2f} ms")

# Save model for size measurement
model.save_pretrained("./pytorch_model")
model_size = sum(os.path.getsize(os.path.join("./pytorch_model", f)) 
                 for f in os.listdir("./pytorch_model") if f.endswith(".bin"))
print(f"Model Size: {model_size/1e6:.2f} MB")
