import sys
from pathlib import Path

import numpy as np
import pytest

# pylint: disable=import-error


ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_PATH = ROOT / "preprocess_data"
if str(PREPROCESS_PATH) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_PATH))

import chunk_audio as chunk_mod  # noqa: E402


def test_chunk_audio_returns_empty_for_short_waveform() -> None:
    wav = np.zeros(chunk_mod.CHUNK_SIZE - 1, dtype=np.float32)
    assert chunk_mod.chunk_audio(wav) == []


def test_chunk_audio_returns_expected_count() -> None:
    wav = np.arange(chunk_mod.CHUNK_SIZE + chunk_mod.HOP_SIZE, dtype=np.float32)
    chunks = chunk_mod.chunk_audio(wav)
    assert len(chunks) == 2
    assert len(chunks[0]) == chunk_mod.CHUNK_SIZE


def test_get_audio_files_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.FLAC").write_bytes(b"x")
    files = chunk_mod.get_audio_files(str(tmp_path))
    names = sorted(filename for _, filename in files)
    assert names == ["a.wav", "c.FLAC"]


def test_process_file_raises_for_wrong_sampling_rate(monkeypatch) -> None:
    monkeypatch.setattr(chunk_mod.sf, "read", lambda _path: (np.zeros(chunk_mod.CHUNK_SIZE), 16000))
    with pytest.raises(ValueError):
        chunk_mod.process_file(("inroot", "x.wav", "inroot", "outroot"))


def test_process_file_writes_chunk_files(monkeypatch, tmp_path: Path) -> None:
    written = []
    wav = np.arange(chunk_mod.CHUNK_SIZE + chunk_mod.HOP_SIZE, dtype=np.float32)
    monkeypatch.setattr(chunk_mod.sf, "read", lambda _path: (wav, chunk_mod.SR))
    monkeypatch.setattr(chunk_mod, "ensure_dir", lambda _p: None)
    monkeypatch.setattr(chunk_mod.sf, "write", lambda path, _chunk, _sr: written.append(path))

    in_root = str(tmp_path / "in")
    out_root = str(tmp_path / "out")
    root = str(tmp_path / "in" / "sub")
    result = chunk_mod.process_file((root, "demo.wav", in_root, out_root))

    assert result is True
    assert len(written) == 2
    assert written[0].endswith("demo_ch0.wav")
    assert written[1].endswith("demo_ch1.wav")
