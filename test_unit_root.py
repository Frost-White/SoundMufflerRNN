from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings
from app.services.rate_limit import InMemoryRateLimiter


def test_settings_override_from_code() -> None:
    settings = Settings(jwt_secret="root-test-secret", enhance_api_rate_limit=9)
    assert settings.jwt_secret == "root-test-secret"
    assert settings.enhance_api_rate_limit == 9


def test_rate_limiter_blocks_after_limit() -> None:
    limiter = InMemoryRateLimiter()
    ok1, _ = limiter.check("root-key", limit=1, window_seconds=60)
    ok2, retry = limiter.check("root-key", limit=1, window_seconds=60)
    assert ok1 is True
    assert ok2 is False
    assert retry >= 1
