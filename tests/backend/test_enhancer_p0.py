import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# pylint: disable=protected-access


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

import app.inference.enhancer as enhancer  # noqa: E402


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32, copy=False), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_resample_linear_passthrough_same_rate() -> None:
    wav = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    out = enhancer._resample_linear(wav, src_sr=16000, dst_sr=16000)
    assert np.allclose(out, wav)


def test_resample_linear_empty_returns_empty() -> None:
    out = enhancer._resample_linear(np.array([], dtype=np.float32), src_sr=16000, dst_sr=48000)
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_get_model_raises_when_weights_missing(monkeypatch) -> None:
    enhancer.get_model.cache_clear()
    monkeypatch.setenv("SOUNDMUFFLER_WEIGHTS_PATH", "D:/SoundMufflerRNN/does-not-exist.pt")
    with pytest.raises(FileNotFoundError):
        enhancer.get_model()
    enhancer.get_model.cache_clear()


def test_enhance_audio_bytes_rejects_invalid_audio() -> None:
    with pytest.raises(enhancer.InferenceError):
        enhancer.enhance_audio_bytes(b"not-audio")


def test_enhance_audio_bytes_stereo_is_mixed_and_written(monkeypatch) -> None:
    def _identity(wav: np.ndarray) -> np.ndarray:
        return wav

    monkeypatch.setattr(enhancer, "_enhance_waveform", _identity)
    stereo = np.stack(
        [
            np.linspace(-0.1, 0.1, 64, dtype=np.float32),
            np.linspace(0.1, -0.1, 64, dtype=np.float32),
        ],
        axis=1,
    )
    in_bytes = _wav_bytes(stereo, sr=enhancer.SR)

    out_bytes = enhancer.enhance_audio_bytes(in_bytes)

    with sf.SoundFile(io.BytesIO(out_bytes)) as f:
        assert int(f.samplerate) == enhancer.SR
        out = f.read(dtype="float32", always_2d=False)
    assert out.ndim == 1
    assert out.shape[0] == 64


def test_enhance_audio_bytes_resamples_back_to_input_rate(monkeypatch) -> None:
    def _identity(wav: np.ndarray) -> np.ndarray:
        return wav

    monkeypatch.setattr(enhancer, "_enhance_waveform", _identity)
    wav = np.sin(np.linspace(0, 2 * np.pi, 80, dtype=np.float32))
    in_bytes = _wav_bytes(wav, sr=16000)

    out_bytes = enhancer.enhance_audio_bytes(in_bytes)

    with sf.SoundFile(io.BytesIO(out_bytes)) as f:
        assert int(f.samplerate) == 16000
