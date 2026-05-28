import sys
from pathlib import Path

# pylint: disable=import-error


ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_PATH = ROOT / "preprocess_data"
if str(PREPROCESS_PATH) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_PATH))

from console_bar import bar_line  # noqa: E402


def test_bar_line_handles_zero_total() -> None:
    assert bar_line("X", 0, 0, width=10) == "X |----------| 0/0"


def test_bar_line_full_progress() -> None:
    assert bar_line("X", 10, 10, width=10) == "X |##########| 10/10"


def test_bar_line_partial_progress() -> None:
    assert bar_line("X", 3, 10, width=10) == "X |###-------| 3/10"
