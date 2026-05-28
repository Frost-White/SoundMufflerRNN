import sys
from pathlib import Path

# pylint: disable=import-error


ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_PATH = ROOT / "preprocess_data"
if str(PREPROCESS_PATH) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_PATH))

import check_sampling_rates as csr_mod  # noqa: E402


class _Future:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _Executor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, func, arg):
        return _Future(func(arg))


def test_get_sampling_rate_returns_none_on_error(monkeypatch) -> None:
    monkeypatch.setattr(csr_mod.wave, "open", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("bad")))
    assert csr_mod.get_sampling_rate("x.wav") is None


def test_check_sampling_rates_parallel_counts_known_and_other(monkeypatch) -> None:
    def _walk(_base):
        return [("root", [], ["a.wav", "b.wav", "c.wav", "d.txt"])]

    monkeypatch.setattr(csr_mod.os, "walk", _walk)
    monkeypatch.setattr(csr_mod, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(csr_mod, "as_completed", lambda futures: list(futures))

    values = {"root/a.wav": 16000, "root/b.wav": 48000, "root/c.wav": 44100}
    monkeypatch.setattr(csr_mod, "get_sampling_rate", lambda path: values[path.replace("\\", "/")])

    counts, others = csr_mod.check_sampling_rates_parallel("base")
    assert counts == {16000: 1, 48000: 1}
    assert others == {44100: 1}


def test_check_sampling_rates_parallel_handles_none_results(monkeypatch) -> None:
    def _walk(_base):
        return [("root", [], ["a.wav"])]

    monkeypatch.setattr(csr_mod.os, "walk", _walk)
    monkeypatch.setattr(csr_mod, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(csr_mod, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(csr_mod, "get_sampling_rate", lambda _path: None)

    counts, others = csr_mod.check_sampling_rates_parallel("base")
    assert counts == {16000: 0, 48000: 0}
    assert others == {}
