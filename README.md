# High-Throughput Image Classification Service

A production-ready image classification API using ResNet-18 with ONNX optimization, FastAPI, and CI/CD pipeline.

## Features

- **Optimized Model**: ResNet-18 converted to ONNX with dynamic quantization (~70% size reduction)
- **High Performance**: ProcessPoolExecutor for concurrent request handling
- **Production Ready**: Docker containerization, comprehensive error handling
- **CI/CD Pipeline**: Automated testing and deployment to Hugging Face Spaces
- **Comprehensive Testing**: pytest unit tests with 100% endpoint coverage

## Project Structure

```
image-classification-service/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── model.py         # ONNX inference logic
│   └── schemas.py       # Pydantic models
├── models/
│   └── resnet18_quantized.onnx  # Optimized model
├── tests/
│   └── test_api.py      # Unit tests
├── scripts/
│   ├── 01_baseline_test.py      # PyTorch baseline benchmark
│   ├── 02_export_onnx.py        # Export to ONNX
│   ├── 03_quantize.py           # Dynamic quantization
│   └── 04_benchmark_onnx.py     # ONNX benchmark
├── .github/
│   └── workflows/
│       └── ci-cd.yml    # GitHub Actions pipeline
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Model

Run the optimization scripts in order:

```bash
cd scripts
python 01_baseline_test.py      # Measure PyTorch baseline
python 02_export_onnx.py        # Export to ONNX
python 03_quantize.py           # Apply quantization
python 04_benchmark_onnx.py     # Compare performance
cd ..
```

### 3. Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### 4. Test the API

```bash
# Health check
curl http://localhost:7860/health

# Predict
curl -X POST "http://localhost:7860/predict" \
  -H "accept: application/json" \
  -F "file=@/path/to/image.jpg"
```

## Docker Deployment

### Build and Run

```bash
docker build -t image-classifier .
docker run -p 7860:7860 image-classifier
```

## Testing

```bash
pytest tests/ -v
```

## API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### POST /predict

Image classification endpoint.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "label": "tabby, tabby cat",
  "score": 0.8234,
  "label_id": 281,
  "inference_time_ms": 45.123
}
```

**Error Codes:**
- `400`: Corrupted or invalid image
- `413`: File too large (max 10MB)
- `415`: Unsupported media type
- `500`: Inference error

## Performance Metrics

| Format | File Size | Avg Latency | P95 Latency |
|--------|-----------|-------------|-------------|
| PyTorch | ~45 MB | baseline | baseline |
| ONNX | ~45 MB | ~20% faster | - |
| ONNX Quantized | ~12 MB | ~40% faster | - |

*Run benchmark scripts to get actual measurements on your hardware*

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Runs unit tests on every push/PR
2. Deploys to Hugging Face Spaces on main branch (requires `HF_TOKEN` secret)

### Setup Hugging Face Deployment

1. Create a Hugging Face Space
2. Generate an access token with write permissions
3. Add `HF_TOKEN` to GitHub repository secrets
4. Update `.github/workflows/ci-cd.yml` with your Space URL

## Model Details

- **Base Model**: microsoft/resnet-18 (Hugging Face)
- **Task**: Image Classification (ImageNet-1k)
- **Input**: RGB images (224x224)
- **Output**: 1000 class probabilities
- **Optimization**: ONNX + Dynamic Quantization (QUint8)

## Development

### Adding New Features

1. Update code in `app/`
2. Add tests in `tests/`
3. Run tests: `pytest tests/ -v`
4. Update documentation

### Performance Testing

Use JMeter or similar tools to test throughput:
- Concurrent users: 10, 50, 100
- Measure: TPS, P95 latency, error rate

## License

MIT

## Acknowledgments

- Model: microsoft/resnet-18 from Hugging Face
- Framework: FastAPI, ONNX Runtime
