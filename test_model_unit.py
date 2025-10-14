import io
from PIL import Image
import numpy as np
import model

def make_rgb_image(width=224, height=224, color=(10, 20, 30)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def test_preprocess_img_returns_expected_shape_and_scale(tmp_path):
    # make a temporary image file
    img_buf = make_rgb_image(300, 400)
    fpath = tmp_path / "sample.png"
    fpath.write_bytes(img_buf.read())

    x = model.preprocess_img(str(fpath))
    assert isinstance(x, np.ndarray)
    assert x.shape == (1, 224, 224, 3)
    assert x.min() >= 0.0 and x.max() <= 1.0

def test_predict_result_uses_argmax(monkeypatch):
    # mock model.predict to avoid loading TF and the .h5 file
    class DummyModel:
        def predict(self, x):
            # batch size 1, 10-class logits
            return np.array([[0.1, 0.2, 0.05, 0.03, 0.0, 0.15, 0.1, 0.05, 0.12, 0.08]])
    monkeypatch.setattr(model, "model", DummyModel())

    fake_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    pred = model.predict_result(fake_input)
    assert pred == 1  # index of max value above
