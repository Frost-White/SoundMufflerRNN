from app.services.rate_limit import InMemoryRateLimiter


def test_rate_limiter_blocks_after_limit() -> None:
    limiter = InMemoryRateLimiter()

    ok1, retry1 = limiter.check("k", limit=2, window_seconds=30)
    ok2, retry2 = limiter.check("k", limit=2, window_seconds=30)
    ok3, retry3 = limiter.check("k", limit=2, window_seconds=30)

    assert (ok1, retry1) == (True, 0)
    assert (ok2, retry2) == (True, 0)
    assert ok3 is False
    assert retry3 >= 1


def test_rate_limiter_resets_after_window(monkeypatch) -> None:
    limiter = InMemoryRateLimiter()
    times = iter([0.0, 1.0, 12.0])
    monkeypatch.setattr("app.services.rate_limit.time.monotonic", lambda: next(times))

    ok1, _ = limiter.check("k", limit=1, window_seconds=10)
    ok2, _ = limiter.check("k", limit=1, window_seconds=10)
    ok3, _ = limiter.check("k", limit=1, window_seconds=10)

    assert ok1 is True
    assert ok2 is False
    assert ok3 is True
