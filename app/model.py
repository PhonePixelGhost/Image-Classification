import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import time
from transformers import AutoImageProcessor, ResNetForImageClassification

# Load feature extractor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

# Optimize session for multi-process environment
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1 # One thread per process worker
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Load ONNX session
session = ort.InferenceSession(
    "models/resnet18_quantized.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

# Load label mapping
cfg = ResNetForImageClassification.from_pretrained("microsoft/resnet-18").config

def run_inference(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="np")
    pixel_values = inputs["pixel_values"].astype(np.float32)

    t0 = time.perf_counter()
    outputs = session.run(["logits"], {"pixel_values": pixel_values})
    elapsed = (time.perf_counter() - t0) * 1000

    logits = outputs[0][0]
    predicted_class_id = int(np.argmax(logits))
    
    return {
        "label": cfg.id2label[predicted_class_id],
        "score": float(np.exp(logits[predicted_class_id]) / np.sum(np.exp(logits))),
        "label_id": predicted_class_id,
        "inference_time_ms": elapsed
    }
