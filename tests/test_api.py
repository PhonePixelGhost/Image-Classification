import pytest
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io

client = TestClient(app)

# --- Helper ---
def get_test_image() -> bytes:
    """Create a dummy test image"""
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
