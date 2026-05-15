"""Minimal console progress for long scans (preload, etc.)."""

from __future__ import annotations

import sys


def tick(label: str, done: int, total: int, *, max_ticks: int = 80) -> None:
    if total <= 0:
        return
    step = max(1, total // max_ticks)
    if done == total or done % step == 0 or total <= 30:
        pct = 100.0 * done / total
        sys.stdout.write(f"\r[{label}] {done}/{total} ({pct:.1f}%)")
        sys.stdout.flush()


def endline() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()
