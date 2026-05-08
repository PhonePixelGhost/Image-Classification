---
name: image-classification-mlops
description: >
  ทักษะสำหรับพัฒนาระบบ High-Throughput Image Classification Service ครบวงจร ตั้งแต่
  Model Optimization, FastAPI Development, CI/CD Pipeline จนถึง Performance Testing
  โดยใช้โมเดล microsoft/resnet-18 จาก Hugging Face

  ใช้ skill นี้เมื่อ:
  - ต้องการ Optimize โมเดล (ONNX Conversion + Dynamic Quantization)
  - สร้าง FastAPI ที่รองรับ Concurrent Request ด้วย ProcessPoolExecutor
  - เขียน Dockerfile สำหรับ Production
  - ตั้งค่า GitHub Actions CI/CD → Deploy ไป Hugging Face Spaces
  - เขียน pytest Unit Tests สำหรับ /predict endpoint
  - วิเคราะห์ผล JMeter Load Test (Throughput / P95 Latency)
  - เขียน Project Report หรือสร้าง System Architecture Diagram
---

# High-Throughput Image Classification Service — MLOps Skill

## ภาพรวมโปรเจกต์

| Phase | เนื้อหา |
|---|---|
| 1. Model Optimization | ResNet-18 → ONNX → Dynamic Quantization |
| 2. API Development | FastAPI + ProcessPoolExecutor + Pydantic |
| 3. Automation & CI/CD | pytest + GitHub Actions + HF Spaces Deploy |
| 4. Performance Testing | JMeter Load Test + TPS/P95 Analysis |

**โมเดลหลัก:** `microsoft/resnet-18` (Hugging Face)  
**Stack:** Python 3.11, FastAPI, ONNX Runtime, Transformers, Docker, GitHub Actions

---

## Phase 1 — Model Optimization

### 1.1 Baseline Test (Original PyTorch)

```python
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch, time, os
from PIL import Image

model_id = "microsoft/resnet-18"
extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = ResNetForImageClassification.from_pretrained(model_id)
model.eval()

# วัด Baseline Latency (100 runs)
img = Image.open("test.jpg").convert("RGB")
inputs = extractor(images=img, return_tensors="pt")

times = []
with torch.no_grad():
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(**inputs)
        times.append(time.perf_counter() - t0)

print(f"Baseline Latency (avg): {sum(times)/len(times)*1000:.2f} ms")
print(f"Model Size: {os.path.getsize('pytorch_model.bin')/1e6:.2f} MB")
```

### 1.2 Export to ONNX

```python
import torch
from transformers import AutoFeatureExtractor, ResNetForImageClassification

model_id = "microsoft/resnet-18"
extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = ResNetForImageClassification.from_pretrained(model_id).eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "resnet18.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch_size"}},
    opset_version=17,
)
print("ONNX exported successfully")
```

### 1.3 Dynamic Quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="resnet18.onnx",
    model_output="resnet18_quantized.onnx",
    weight_type=QuantType.QUint8,
)
print("Quantization complete")
```

### 1.4 ตารางเปรียบเทียบ (บันทึกผลจริงลงตาราง)

| Format | File Size (MB) | Avg Latency (ms) | P95 Latency (ms) |
|---|---|---|---|
| Original (PyTorch) | ~45 | baseline | baseline |
| ONNX | ~45 | คาดว่าเร็วขึ้น ~20% | - |
| ONNX Quantized | ~12 | คาดว่าเร็วขึ้น ~40% | - |

> **วิธีวัด:** รัน 100 ครั้ง → เก็บค่า avg และ percentile ด้วย `numpy.percentile(times, 95)`

---

## Phase 2 — API Development

### 2.1 โครงสร้างโปรเจกต์

```
image-classification-service/
├── app/
│   ├── main.py          # FastAPI app
│   ├── model.py         # ONNX inference logic
│   └── schemas.py       # Pydantic models
├── models/
│   └── resnet18_quantized.onnx
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

### 2.2 Pydantic Schemas (`app/schemas.py`)

```python
from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    label: str
    score: float
    label_id: int
    inference_time_ms: float

class ErrorResponse(BaseModel):
    detail: str
    error_code: str
```

### 2.3 ONNX Inference (`app/model.py`)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
import io, time

# Labels จาก ImageNet
from transformers import AutoFeatureExtractor
extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")

# โหลด session ครั้งเดียว (module-level)
session = ort.InferenceSession(
    "models/resnet18_quantized.onnx",
    providers=["CPUExecutionProvider"]
)

def run_inference(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = extractor(images=img, return_tensors="np")
    pixel_values = inputs["pixel_values"].astype(np.float32)

    t0 = time.perf_counter()
    outputs = session.run(["logits"], {"pixel_values": pixel_values})
    elapsed = (time.perf_counter() - t0) * 1000

    logits = outputs[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    label_id = int(np.argmax(probs))

    # ดึง label จาก model config
    from transformers import ResNetForImageClassification
    cfg = ResNetForImageClassification.from_pretrained("microsoft/resnet-18").config
    label = cfg.id2label.get(label_id, str(label_id))

    return {
        "label": label,
        "score": float(probs[label_id]),
        "label_id": label_id,
        "inference_time_ms": round(elapsed, 3),
    }
```

### 2.4 FastAPI Main App (`app/main.py`)

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from concurrent.futures import ProcessPoolExecutor
import asyncio
from app.model import run_inference
from app.schemas import PredictionResponse

app = FastAPI(title="ResNet-18 Image Classifier", version="1.0.0")
executor = ProcessPoolExecutor(max_workers=4)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Allowed: {ALLOWED_CONTENT_TYPES}"
        )

    image_bytes = await file.read()

    # Validate file size
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE // 1024 // 1024} MB."
        )

    # Validate not corrupted (try opening with PIL)
    try:
        from PIL import Image
        import io
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")

    # Run CPU-bound inference in ProcessPoolExecutor (ไม่บล็อก event loop)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, run_inference, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return PredictionResponse(**result)
```

### 2.5 Error Handling Summary

| สถานการณ์ | HTTP Status | รายละเอียด |
|---|---|---|
| ไฟล์ไม่ใช่รูปภาพ | 415 Unsupported Media Type | Content-type ไม่ตรง |
| ไฟล์เสีย (Corrupted) | 400 Bad Request | PIL ไม่สามารถเปิดได้ |
| ไฟล์ใหญ่เกินไป | 413 Request Entity Too Large | เกิน 10MB |
| Inference Error | 500 Internal Server Error | โมเดลทำงานผิดพลาด |

---

## Phase 3 — Dockerfile

```dockerfile
# ใช้ slim image เพื่อลด size
FROM python:3.11-slim

WORKDIR /app

# ติดตั้ง dependencies ก่อน (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy โค้ดและโมเดล
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

**requirements.txt:**
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
onnxruntime==1.18.0
numpy==1.26.4
Pillow==10.3.0
transformers==4.41.0
torch==2.3.0
pydantic==2.7.1
pytest==8.2.0
httpx==0.27.0
```

> **เทคนิคลด Docker Image Size:**
> - ใช้ `python:3.11-slim` (ไม่ใช่ full)
> - `--no-cache-dir` ใน pip
> - ลบ torch ออกหลัง export ONNX (ใน production image ไม่จำเป็น)
> - ใช้ `.dockerignore` เพื่อ exclude `tests/`, `.git/`, `*.pt`

---

## Phase 4 — Unit Testing (`tests/test_api.py`)

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app
from pathlib import Path

client = TestClient(app)

# --- Helper ---
def get_test_image() -> bytes:
    """ใช้ภาพ test จริงหรือสร้าง dummy PNG"""
    from PIL import Image
    import io
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# --- Tests ---

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_predict_returns_valid_json():
    img_bytes = get_test_image()
    res = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "label" in data
    assert "score" in data
    assert isinstance(data["score"], float)
    assert 0.0 <= data["score"] <= 1.0


def test_predict_rejects_non_image():
    res = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert res.status_code == 415


def test_predict_rejects_corrupted_file():
    res = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"\xff\xd8corrupted", "image/jpeg")}
    )
    assert res.status_code == 400


def test_predict_rejects_oversized_file():
    huge = b"A" * (11 * 1024 * 1024)  # 11MB
    res = client.post(
        "/predict",
        files={"file": ("big.jpg", huge, "image/jpeg")}
    )
    assert res.status_code == 413
```

---

## Phase 5 — GitHub Actions CI/CD (`.github/workflows/ci-cd.yml`)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run Unit Tests
        run: pytest tests/ -v --tb=short

  deploy:
    needs: test          # รัน deploy เฉพาะเมื่อ test ผ่านทุก case
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4

      - name: Push to Hugging Face Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config --global user.email "ci@github.com"
          git config --global user.name "GitHub Actions"
          git remote add hf https://user:${HF_TOKEN}@huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
          git push hf main --force
```

> **การตั้งค่า Secret:**  
> ไปที่ GitHub Repo → Settings → Secrets → Actions → New secret  
> ชื่อ: `HF_TOKEN` | ค่า: Hugging Face Access Token (write permission)

---

## Phase 6 — Performance Testing (JMeter)

### 6.1 JMeter Test Plan (.jmx) — Key Settings

| Parameter | Local (Docker) | Cloud (HF Spaces) |
|---|---|---|
| Threads (Users) | 10, 50, 100 | 10, 25, 50 |
| Ramp-Up (sec) | 10 | 20 |
| Loop Count | 100 | 50 |
| Endpoint | `http://localhost:7860/predict` | `https://<space>.hf.space/predict` |

### 6.2 Metrics ที่ต้องรายงาน

| Metric | คำอธิบาย | เป้าหมาย |
|---|---|---|
| **Throughput (TPS)** | Request ต่อวินาที | สูงที่สุด |
| **P95 Latency** | 95th percentile response time | < 2000ms |
| **Error Rate** | % ที่ได้รับ error | < 1% |
| **Avg Latency** | ค่าเฉลี่ย response time | ต่ำที่สุด |

### 6.3 การวิเคราะห์ผล

```
จุดที่ต้องวิเคราะห์:
1. หา "Knee Point" — จุดที่ TPS หยุดเพิ่ม แต่ Latency เริ่มพุ่ง
2. CPU Utilization ใน Docker stats ณ จำนวน concurrent users นั้น
3. เปรียบเทียบ Local vs Cloud เพื่อดู overhead ของ Network/HF cold-start
```

---

## Phase 7 — cURL Examples

```bash
# Health Check
curl https://<USERNAME>-<SPACE>.hf.space/health

# Predict (ส่งไฟล์รูปภาพจริง)
curl -X POST "https://<USERNAME>-<SPACE>.hf.space/predict" \
  -H "accept: application/json" \
  -F "file=@/path/to/your/image.jpg"

# Postman Collection — ดูไฟล์ postman_collection.json ใน repo
```

---

## Checklist Deliverables

- [ ] Project Report (PDF) — Model details, Optimization table, Error strategy, JMeter analysis, Architecture diagram
- [ ] GitHub Repo — Source code + `.github/workflows/ci-cd.yml` + `README.md`
- [ ] `resnet18_quantized.onnx` — โมเดลที่ optimize แล้ว
- [ ] `tests/test_api.py` — pytest ครอบคลุม Happy path + Error cases
- [ ] `Dockerfile` — Production-ready
- [ ] JMeter Test Plan (`.jmx`)
- [ ] Postman Collection (`.json`)
- [ ] Hugging Face Space — Live API endpoint
- [ ] Presentation Slides + Live Demo (9 พ.ค. 2569)

---

## Notes & Tips

- **HF Spaces Free Tier** ใช้ CPU เท่านั้น — ONNX Runtime บน CPU เหมาะสมที่สุด
- **Cold Start** ใน HF Spaces อาจทำให้ request แรกช้า — ควรระบุในรายงาน
- **ProcessPoolExecutor** ต้องระวัง: แต่ละ worker โหลด ONNX session แยกกัน (memory x workers)
- **Pydantic v2** syntax เปลี่ยนจาก v1 — ใช้ `model_config` แทน `class Config`
- ใน `pytest` ต้องมี `conftest.py` หรือ set `PYTHONPATH=.` ให้ถูกต้อง
