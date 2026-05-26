from __future__ import annotations

import threading
import time


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, int]] = {}

    def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            started_at, count = self._state.get(key, (now, 0))
            elapsed = now - started_at
            if elapsed >= window_seconds:
                started_at = now
                count = 0

            if count >= limit:
                retry_after = max(1, int(window_seconds - elapsed))
                return False, retry_after

            self._state[key] = (started_at, count + 1)
            return True, 0


rate_limiter = InMemoryRateLimiter()
