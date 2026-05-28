import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.config import Settings  # noqa: E402
from app.inference import InferenceError  # noqa: E402
from app.main import app  # noqa: E402
import app.main as main_mod  # noqa: E402
from app.services.rate_limit import InMemoryRateLimiter  # noqa: E402


def _upload(client: TestClient, route: str, content: bytes, filename: str = "clip.wav"):
    return client.post(route, files={"file": (filename, io.BytesIO(content), "audio/wav")})


def _settings(api_limit: int = 100, web_limit: int = 100) -> Settings:
    return Settings(
        enhance_api_rate_limit=api_limit,
        enhance_web_rate_limit=web_limit,
        enhance_api_rate_window_seconds=60,
        enhance_web_rate_window_seconds=60,
    )


def test_enhance_api_returns_400_for_empty_upload() -> None:
    main_mod.rate_limiter = InMemoryRateLimiter()
    def _get_settings() -> Settings:
        return _settings()

    app.dependency_overrides[main_mod.get_settings] = _get_settings
    client = TestClient(app)

    resp = _upload(client, "/enhance", b"")

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Uploaded file is empty."
    app.dependency_overrides.clear()


def test_enhance_api_returns_audio_and_filename_on_success(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "enhance_audio_bytes", lambda _: b"wav-bytes")
    main_mod.rate_limiter = InMemoryRateLimiter()
    def _get_settings() -> Settings:
        return _settings()

    app.dependency_overrides[main_mod.get_settings] = _get_settings
    client = TestClient(app)

    resp = _upload(client, "/enhance", b"in", filename="demo.mp3")

    assert resp.status_code == 200
    assert resp.content == b"wav-bytes"
    assert resp.headers["content-type"].startswith("audio/wav")
    assert 'filename="processed-demo.wav"' in resp.headers["content-disposition"]
    app.dependency_overrides.clear()


def test_enhance_api_returns_400_for_inference_error(monkeypatch) -> None:
    def _raise(_: bytes) -> bytes:
        raise InferenceError("bad audio")

    monkeypatch.setattr(main_mod, "enhance_audio_bytes", _raise)
    main_mod.rate_limiter = InMemoryRateLimiter()
    def _get_settings() -> Settings:
        return _settings()

    app.dependency_overrides[main_mod.get_settings] = _get_settings
    client = TestClient(app)

    resp = _upload(client, "/enhance", b"in")

    assert resp.status_code == 400
    assert resp.json()["detail"] == "bad audio"
    app.dependency_overrides.clear()


def test_enhance_api_returns_500_for_missing_weights(monkeypatch) -> None:
    def _raise(_: bytes) -> bytes:
        raise FileNotFoundError("weights")

    monkeypatch.setattr(main_mod, "enhance_audio_bytes", _raise)
    main_mod.rate_limiter = InMemoryRateLimiter()
    def _get_settings() -> Settings:
        return _settings()

    app.dependency_overrides[main_mod.get_settings] = _get_settings
    client = TestClient(app)

    resp = _upload(client, "/enhance", b"in")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Model weights are not available."
    app.dependency_overrides.clear()


def test_enhance_api_returns_500_for_unexpected_error(monkeypatch) -> None:
    def _raise(_: bytes) -> bytes:
        raise RuntimeError("boom")

    monkeypatch.setattr(main_mod, "enhance_audio_bytes", _raise)
    main_mod.rate_limiter = InMemoryRateLimiter()
    def _get_settings() -> Settings:
        return _settings()

    app.dependency_overrides[main_mod.get_settings] = _get_settings
    client = TestClient(app)

    resp = _upload(client, "/enhance", b"in")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Audio processing failed."
    app.dependency_overrides.clear()


def test_enhance_api_rate_limit_returns_429(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "enhance_audio_bytes", lambda _: b"ok")
    main_mod.rate_limiter = InMemoryRateLimiter()
    def _get_settings() -> Settings:
        return _settings(api_limit=0, web_limit=100)

    app.dependency_overrides[main_mod.get_settings] = _get_settings
    client = TestClient(app)

    resp = _upload(client, "/enhance", b"in")

    assert resp.status_code == 429
    assert resp.json()["detail"] == "Rate limit exceeded."
    assert "retry-after" in resp.headers
    app.dependency_overrides.clear()


def test_enhance_api_and_web_use_separate_rate_limits(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "enhance_audio_bytes", lambda _: b"ok")
    main_mod.rate_limiter = InMemoryRateLimiter()
    def _get_settings() -> Settings:
        return _settings(api_limit=0, web_limit=2)

    app.dependency_overrides[main_mod.get_settings] = _get_settings
    client = TestClient(app)

    api_resp = _upload(client, "/enhance", b"in")
    web_resp = _upload(client, "/enhance/web", b"in")

    assert api_resp.status_code == 429
    assert web_resp.status_code == 200
    app.dependency_overrides.clear()
