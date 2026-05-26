import importlib
import sys
import types
from io import BytesIO

import pytest
import soundfile as sf
from fastapi.testclient import TestClient


def _wav_bytes() -> bytes:
    buf = BytesIO()
    sf.write(buf, [0.0, 0.1, -0.1, 0.0], 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def app_module(monkeypatch):
    fake = types.ModuleType("app.inference")

    class _InferenceError(Exception):
        pass

    def _enhance_audio_bytes(content: bytes) -> bytes:
        return content

    fake.InferenceError = _InferenceError
    fake.enhance_audio_bytes = _enhance_audio_bytes
    monkeypatch.setitem(sys.modules, "app.inference", fake)
    sys.modules.pop("app.main", None)
    return importlib.import_module("app.main")


def test_root_and_health(app_module) -> None:
    client = TestClient(app_module.app)
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/favicon.ico").status_code == 204


def test_enhance_rejects_empty_file(app_module) -> None:
    client = TestClient(app_module.app)
    res = client.post("/enhance/web", files={"file": ("x.wav", b"", "audio/wav")})
    assert res.status_code == 400
    assert "empty" in res.text.lower()


def test_enhance_returns_audio_response(app_module) -> None:
    client = TestClient(app_module.app)
    content = _wav_bytes()
    res = client.post("/enhance/web", files={"file": ("x.wav", content, "audio/wav")})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("audio/wav")
    assert "processed-x.wav" in res.headers.get("content-disposition", "")
