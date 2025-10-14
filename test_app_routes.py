import io
from PIL import Image
import numpy as np
import pytest
import app as flask_app_module

@pytest.fixture()
def client():
    flask_app_module.app.config.update(TESTING=True)
    with flask_app_module.app.test_client() as c:
        yield c

def make_png_bytes():
    img = Image.new("RGB", (224, 224), (100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def test_home_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Hand Sign Digit Language Detection" in resp.data

def test_prediction_happy_path(client, monkeypatch):
    # stub model to keep it fast
    import model as model_module
    class DummyModel:
        def predict(self, x):  # x shape should be (1, 224, 224, 3)
            assert x.shape[1:] == (224, 224, 3)
            return np.array([[0,0,0,0,0,1,0,0,0,0]])
    monkeypatch.setattr(model_module, "model", DummyModel())

    data = {"file": (make_png_bytes(), "test.png")}
    resp = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Prediction" in resp.data
    assert b"5" in resp.data  # from DummyModel argmax

def test_prediction_missing_file(client):
    resp = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Invalid request method" not in resp.data  # should be handled by our try/except
    assert b"cannot be processed" in resp.data.lower() or b"file" in resp.data.lower()

def test_prediction_bad_type(client):
    data = {"file": (io.BytesIO(b"not-an-image"), "fake.txt")}
    resp = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    # our app shows an error on bad image
    assert b"cannot be processed" in resp.data.lower()
