"""Shared terminal progress bar line (no side effects)."""


def bar_line(label: str, current: int, total: int, width: int = 40) -> str:
    if total == 0:
        return f"{label} |{'-' * width}| 0/0"
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"{label} |{bar}| {current}/{total}"
