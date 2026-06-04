import io
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fake_image_bytes():
    """Creates a real in-memory image to upload in tests."""
    img = Image.new("RGB", (256, 256), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

# ── Test 1: Model loading ─────────────────────────────────────────────────────

def test_model_loads_on_startup():
    """S3 download is mocked — checks model and class names are set correctly."""
    fake_class_names = {str(i): f"disease_{i}" for i in range(29)}

    with patch("main.s3_client.download_file"), \
         patch("main.torch.load", return_value={}), \
         patch("main.models.alexnet") as mock_alexnet, \
         patch("builtins.open", MagicMock(
             return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(
                     read=MagicMock(return_value=json.dumps(fake_class_names))
                 )),
                 __exit__=MagicMock(return_value=False)
             )
         )), \
         patch("main.json.load", return_value=fake_class_names):

        from main import load_model_from_s3
        import main
        load_model_from_s3()

        assert main.model is not None
        assert main.idx_class is not None
        assert len(main.idx_class) == 29

# ── Test 2: /predict returns 503 when model is not loaded ─────────────────────

def test_predict_returns_503_when_model_not_loaded():
    """If model failed to load, endpoint should return 503."""
    import main
    main.model = None  # force model to be None

    from main import app
    client = TestClient(app, raise_server_exceptions=False)

    img_bytes = make_fake_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]

# ── Test 3: /predict returns 400 for a non-image file ─────────────────────────

def test_predict_rejects_non_image():
    """Sending a text file should return 400."""
    import main
    main.model = MagicMock()  # pretend model is loaded
    main.idx_class = {"0": "healthy"}

    from main import app
    client = TestClient(app, raise_server_exceptions=False)

    garbage = io.BytesIO(b"this is not an image")
    response = client.post(
        "/predict",
        files={"file": ("bad.txt", garbage, "text/plain")}
    )
    assert response.status_code == 400

# ── Test 4: Full pipeline with mocked S3 and Gemini ───────────────────────────

def test_predict_full_pipeline():
    """Mocks S3 model + Gemini API, checks full response shape."""
    import main
    import torch

    # Fake model that returns a tensor pointing to class 0
    fake_model = MagicMock()
    fake_model.return_value = torch.zeros(1, 29)
    main.model = fake_model
    main.idx_class = {str(i): f"disease_{i}" for i in range(29)}

    fake_gemini_response = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "description": "A serious plant disease.",
                        "symptoms": ["s1", "s2", "s3", "s4"],
                        "treatment": ["t1", "t2", "t3", "t4"],
                        "prevention": ["p1", "p2", "p3", "p4"]
                    })
                }]
            }
        }]
    }

    with patch("main.httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = fake_gemini_response
        mock_client.return_value.__aenter__.return_value.post = \
            MagicMock(return_value=mock_response)

        from main import app
        client = TestClient(app, raise_server_exceptions=False)

        img_bytes = make_fake_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("plant.jpg", img_bytes, "image/jpeg")}
        )

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "disease_info" in body
    assert "symptoms" in body["disease_info"]
    assert "treatment" in body["disease_info"]