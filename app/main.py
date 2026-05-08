from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from concurrent.futures import ProcessPoolExecutor
import asyncio
from app.model import run_inference
from app.schemas import PredictionResponse
from PIL import UnidentifiedImageError

app = FastAPI(title="ResNet-18 Image Classifier", version="1.0.0")
executor = ProcessPoolExecutor(max_workers=4)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    # ... (HTML UI code remains the same)
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ResNet-18 Image Classifier</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; color: #333; }
            .card { background: white; padding: 2rem; border-radius: 1.5rem; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); width: 100%; max-width: 450px; text-align: center; }
            h1 { color: #1a202c; margin-bottom: 1.5rem; font-size: 1.5rem; }
            #preview { width: 100%; height: 250px; border: 2px dashed #e2e8f0; border-radius: 1rem; object-fit: cover; margin-bottom: 1.5rem; display: none; }
            .upload-btn-wrapper { position: relative; overflow: hidden; display: inline-block; width: 100%; }
            .btn { border: none; color: white; background: #667eea; padding: 0.75rem 2rem; border-radius: 0.75rem; font-weight: 600; cursor: pointer; transition: 0.3s; width: 100%; font-size: 1rem; }
            .btn:hover { background: #5a67d8; transform: translateY(-2px); }
            input[type=file] { font-size: 100px; position: absolute; left: 0; top: 0; opacity: 0; cursor: pointer; }
            #result { margin-top: 1.5rem; padding: 1rem; border-radius: 1rem; background: #f7fafc; display: none; text-align: left; }
            .label { font-weight: 600; color: #4a5568; }
            .value { color: #2d3748; float: right; }
            .loading { display: none; margin-top: 1rem; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>ResNet-18 Quantized</h1>
            <img id="preview" src="#" alt="Preview">
            <div class="upload-btn-wrapper">
                <button class="btn" id="btn-text">Select Image to Predict</button>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            <div id="loading" class="loading">⌛ Predicting...</div>
            <div id="result">
                <div><span class="label">Label:</span> <span class="value" id="res-label">-</span></div>
                <div style="margin-top:0.5rem"><span class="label">Confidence:</span> <span class="value" id="res-score">-</span></div>
                <div style="margin-top:0.5rem"><span class="label">Inference:</span> <span class="value" id="res-time">-</span></div>
            </div>
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const preview = document.getElementById('preview');
            const resultDiv = document.getElementById('result');
            const loading = document.getElementById('loading');
            const btnText = document.getElementById('btn-text');

            fileInput.onchange = evt => {
                const [file] = fileInput.files;
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    preview.style.display = 'block';
                    predict(file);
                }
            }

            async function predict(file) {
                const formData = new FormData();
                formData.append('file', file);

                loading.style.display = 'block';
                resultDiv.style.display = 'none';
                btnText.disabled = true;

                try {
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    if (response.status !== 200) {
                        alert(data.detail || 'Prediction failed');
                        return;
                    }

                    document.getElementById('res-label').innerText = data.label;
                    document.getElementById('res-score').innerText = (data.score * 100).toFixed(2) + '%';
                    document.getElementById('res-time').innerText = data.inference_time_ms.toFixed(2) + ' ms';
                    
                    resultDiv.style.display = 'block';
                } catch (e) {
                    alert('Error predicting image');
                } finally {
                    loading.style.display = 'none';
                    btnText.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # 1. ตรวจสอบ Content Type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    
    # 2. อ่านข้อมูล
    image_bytes = await file.read()

    # 3. ตรวจสอบขนาดไฟล์ (Fix สำหรับ test_predict_rejects_oversized_file)
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # 4. รัน Inference และดักจับ Error (Fix สำหรับ test_predict_rejects_corrupted_file)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, run_inference, image_bytes)
        return result
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
