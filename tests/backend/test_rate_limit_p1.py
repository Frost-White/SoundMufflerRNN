import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

import app.services.rate_limit as rl_mod  # noqa: E402


def test_rate_limiter_allows_until_limit_then_blocks() -> None:
    limiter = rl_mod.InMemoryRateLimiter()
    assert limiter.check("k", limit=2, window_seconds=60) == (True, 0)
    assert limiter.check("k", limit=2, window_seconds=60) == (True, 0)
    allowed, retry_after = limiter.check("k", limit=2, window_seconds=60)
    assert allowed is False
    assert retry_after >= 1


def test_rate_limiter_window_reset(monkeypatch) -> None:
    ticks = iter([0.0, 0.5, 2.1])
    monkeypatch.setattr(rl_mod.time, "monotonic", lambda: next(ticks))
    limiter = rl_mod.InMemoryRateLimiter()
    assert limiter.check("k", limit=1, window_seconds=2) == (True, 0)
    assert limiter.check("k", limit=1, window_seconds=2)[0] is False
    assert limiter.check("k", limit=1, window_seconds=2) == (True, 0)


def test_rate_limiter_keys_are_isolated() -> None:
    limiter = rl_mod.InMemoryRateLimiter()
    assert limiter.check("a", limit=1, window_seconds=60) == (True, 0)
    assert limiter.check("b", limit=1, window_seconds=60) == (True, 0)
    assert limiter.check("a", limit=1, window_seconds=60)[0] is False
