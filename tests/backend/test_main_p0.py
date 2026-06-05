import io
import sys
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.config import Settings  # noqa: E402
from app.db.session import get_db  # noqa: E402
from app.deps import get_active_api_key, get_current_user  # noqa: E402
from app.inference import InferenceError  # noqa: E402
from app.main import app  # noqa: E402
import app.main as main_mod  # noqa: E402
from app.services import api_entitlements as ent_mod  # noqa: E402
from app.services.rate_limit import InMemoryRateLimiter  # noqa: E402


class FakeApiKey:
    def __init__(self, key_id=None, user_id=None):
        self.id = key_id or uuid4()
        self.user_id = user_id or uuid4()


class FakeUser:
    def __init__(self, user_id=None):
        self.id = user_id or uuid4()


def _upload(
    client: TestClient,
    route: str,
    content: bytes,
    filename: str = "clip.wav",
    headers: dict[str, str] | None = None,
):
    return client.post(
        route,
        files={"file": (filename, io.BytesIO(content), "audio/wav")},
        headers=headers or {},
    )


def _settings(
    api_pro_limit: int = 100,
    api_free_limit: int = 100,
    web_limit: int = 100,
) -> Settings:
    return Settings(
        enhance_api_pro_rate_limit=api_pro_limit,
        enhance_api_pro_rate_window_seconds=60,
        enhance_api_free_rate_limit=api_free_limit,
        enhance_api_free_rate_window_seconds=900,
        enhance_web_rate_limit=web_limit,
        enhance_web_rate_window_seconds=60,
    )


def _setup_client(
    api_pro_limit: int = 100,
    api_free_limit: int = 100,
    web_limit: int = 100,
    plan_id: str = "pro",
    api_key: FakeApiKey | None = None,
    user: FakeUser | None = None,
) -> tuple[TestClient, FakeApiKey, FakeUser]:
    main_mod.rate_limiter = InMemoryRateLimiter()
    app.dependency_overrides[main_mod.get_settings] = lambda: _settings(
        api_pro_limit=api_pro_limit,
        api_free_limit=api_free_limit,
        web_limit=web_limit,
    )
    main_mod.resolve_user_plan_id = lambda _db, _uid: plan_id

    def _fake_get_db():
        yield None

    app.dependency_overrides[get_db] = _fake_get_db
    key = api_key or FakeApiKey()
    fake_user = user or FakeUser()
    app.dependency_overrides[get_active_api_key] = lambda: key
    app.dependency_overrides[get_current_user] = lambda: fake_user
    return TestClient(app), key, fake_user


def _clear_overrides() -> None:
    app.dependency_overrides.clear()
    main_mod.resolve_user_plan_id = ent_mod.resolve_user_plan_id


def test_enhance_api_returns_401_without_api_key() -> None:
    main_mod.rate_limiter = InMemoryRateLimiter()
    client = TestClient(app)

    resp = _upload(client, "/enhance", b"in")

    assert resp.status_code == 401
    assert resp.json()["detail"] == "API key is required"


def test_enhance_api_returns_403_for_invalid_api_key() -> None:
    def _reject_key() -> FakeApiKey:
        raise HTTPException(status_code=403, detail="Invalid API key")

    app.dependency_overrides[get_active_api_key] = _reject_key
    client = TestClient(app)

    resp = _upload(client, "/enhance", b"in", headers={"X-Api-Key": "sk-live-bad"})

    assert resp.status_code == 403
    assert resp.json()["detail"] == "Invalid API key"
    _clear_overrides()


def test_enhance_web_returns_401_without_bearer_token() -> None:
    main_mod.rate_limiter = InMemoryRateLimiter()
    client = TestClient(app)

    resp = _upload(client, "/enhance/web", b"in")

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Not authenticated"


def test_enhance_api_returns_400_for_empty_upload() -> None:
    client, _, _ = _setup_client()

    resp = _upload(client, "/enhance", b"")

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Uploaded file is empty."
    _clear_overrides()


def test_enhance_api_returns_413_when_upload_exceeds_2mb() -> None:
    client, _, _ = _setup_client()
    oversized = b"x" * (2 * 1024 * 1024 + 1)

    resp = _upload(client, "/enhance", oversized)

    assert resp.status_code == 413
    assert "2 MB" in resp.json()["detail"]
    _clear_overrides()


def test_enhance_api_returns_audio_and_filename_on_success(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "enhance_audio_bytes", lambda _: b"wav-bytes")
    client, _, _ = _setup_client()

    resp = _upload(client, "/enhance", b"in", filename="demo.mp3")

    assert resp.status_code == 200
    assert resp.content == b"wav-bytes"
    assert resp.headers["content-type"].startswith("audio/wav")
    assert 'filename="processed-demo.wav"' in resp.headers["content-disposition"]
    _clear_overrides()


def test_enhance_api_returns_400_for_inference_error(monkeypatch) -> None:
    def _raise(_: bytes) -> bytes:
        raise InferenceError("bad audio")

    monkeypatch.setattr(main_mod, "enhance_audio_bytes", _raise)
    client, _, _ = _setup_client()

    resp = _upload(client, "/enhance", b"in")

    assert resp.status_code == 400
    assert resp.json()["detail"] == "bad audio"
    _clear_overrides()


def test_enhance_api_returns_500_for_missing_weights(monkeypatch) -> None:
    def _raise(_: bytes) -> bytes:
        raise FileNotFoundError("weights")

    monkeypatch.setattr(main_mod, "enhance_audio_bytes", _raise)
    client, _, _ = _setup_client()

    resp = _upload(client, "/enhance", b"in")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Model weights are not available."
    _clear_overrides()


def test_enhance_api_returns_500_for_unexpected_error(monkeypatch) -> None:
    def _raise(_: bytes) -> bytes:
        raise RuntimeError("boom")

    monkeypatch.setattr(main_mod, "enhance_audio_bytes", _raise)
    client, _, _ = _setup_client()

    resp = _upload(client, "/enhance", b"in")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Audio processing failed."
    _clear_overrides()


def test_enhance_api_rate_limit_returns_429(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "enhance_audio_bytes", lambda _: b"ok")
    client, _, _ = _setup_client(api_pro_limit=0, plan_id="pro")

    resp = _upload(client, "/enhance", b"in")

    assert resp.status_code == 429
    assert resp.json()["detail"] == "Rate limit exceeded."
    assert "retry-after" in resp.headers
    _clear_overrides()


def test_enhance_api_and_web_use_separate_rate_limits(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "enhance_audio_bytes", lambda _: b"ok")
    client, _, _ = _setup_client(api_pro_limit=0, web_limit=2, plan_id="pro")

    api_resp = _upload(client, "/enhance", b"in")
    web_resp = _upload(client, "/enhance/web", b"in")

    assert api_resp.status_code == 429
    assert web_resp.status_code == 200
    _clear_overrides()


def test_enhance_api_free_plan_uses_free_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "enhance_audio_bytes", lambda _: b"ok")
    client, _, _ = _setup_client(api_free_limit=2, api_pro_limit=100, plan_id="free")

    first = _upload(client, "/enhance", b"in")
    second = _upload(client, "/enhance", b"in")
    third = _upload(client, "/enhance", b"in")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    _clear_overrides()


def test_enhance_api_pro_plan_uses_pro_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "enhance_audio_bytes", lambda _: b"ok")
    client, _, _ = _setup_client(api_pro_limit=1, api_free_limit=100, plan_id="pro")

    first = _upload(client, "/enhance", b"in")
    second = _upload(client, "/enhance", b"in")

    assert first.status_code == 200
    assert second.status_code == 429
    _clear_overrides()


def test_enhance_web_rate_limit_is_per_user(monkeypatch) -> None:
    monkeypatch.setattr(main_mod, "enhance_audio_bytes", lambda _: b"ok")
    user_a = FakeUser()
    user_b = FakeUser()
    client, _, _ = _setup_client(web_limit=1, user=user_a)

    first = _upload(client, "/enhance/web", b"in")
    second = _upload(client, "/enhance/web", b"in")

    app.dependency_overrides[get_current_user] = lambda: user_b
    third = _upload(client, "/enhance/web", b"in")

    assert first.status_code == 200
    assert second.status_code == 429
    assert third.status_code == 200
    _clear_overrides()
