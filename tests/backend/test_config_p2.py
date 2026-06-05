import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.config import Settings, get_settings  # noqa: E402


def test_settings_defaults_are_loaded() -> None:
    s = Settings()
    assert s.jwt_algorithm == "HS256"
    assert s.enhance_api_free_rate_limit == 30
    assert s.enhance_api_free_rate_window_seconds == 900
    assert s.enhance_api_pro_rate_limit == 15
    assert s.enhance_api_pro_rate_window_seconds == 60
    assert s.enhance_web_rate_limit == 30


def test_settings_accept_env_overrides_and_ignore_extra(monkeypatch) -> None:
    monkeypatch.setenv("ENHANCE_API_PRO_RATE_LIMIT", "77")
    monkeypatch.setenv("SOME_UNUSED_VAR", "ignored")
    s = Settings()
    assert s.enhance_api_pro_rate_limit == 77


def test_get_settings_returns_settings_instance() -> None:
    assert isinstance(get_settings(), Settings)
