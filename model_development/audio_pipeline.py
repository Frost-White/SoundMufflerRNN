"""Audio I/O + chunking + per-chunk STFT pipeline."""

import os
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa

SR             = 48000
CHUNK_SAMPLES  = 960                                # 20 ms @ 48 kHz
CHUNK_OVERLAP  = 240                                # 5 ms overlap
CHUNK_HOP      = CHUNK_SAMPLES - CHUNK_OVERLAP      # 720
N_FFT          = 960
WIN_LENGTH     = 960
STFT_HOP       = 240
WINDOW         = "hann"

_BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(_BASE_DIR, "..", "data", "train", "clean_trainset_56spk_wav")
)


def find_first_wav(root: str) -> str:
    """First .wav found by alphabetical walk under `root`."""
    for cur_dir, _, names in os.walk(root):
        for name in sorted(names):
            if name.lower().endswith(".wav"):
                return os.path.join(cur_dir, name)
    raise FileNotFoundError(f"No .wav under: {root}")


def load_audio(path: str, expected_sr: int = SR) -> Tuple[np.ndarray, int]:
    """Read mono float waveform; collapse stereo by mean."""
    audio, sr = sf.read(path, always_2d=False)
    if sr != expected_sr:
        raise ValueError(f"Expected {expected_sr} Hz, got {sr} Hz: {path}")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio, sr


def chunk_waveform(
    waveform: np.ndarray,
    chunk_size: int = CHUNK_SAMPLES,
    hop: int = CHUNK_HOP,
) -> np.ndarray:
    """Split waveform into overlapping chunks of shape (num_chunks, chunk_size)."""
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1-D waveform, got shape {waveform.shape}")
    if len(waveform) < chunk_size:
        return np.empty((0, chunk_size), dtype=waveform.dtype)

    windows = np.lib.stride_tricks.sliding_window_view(waveform, chunk_size)
    chunks = windows[::hop]
    return np.ascontiguousarray(chunks)


def stft_chunk(
    chunk: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = STFT_HOP,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
) -> np.ndarray:
    """Single-chunk STFT, returns (n_fft//2+1, num_frames) complex array."""
    return librosa.stft(
        chunk.astype(np.float32, copy=False),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
    )


def stft_chunks(chunks: np.ndarray, **kwargs) -> np.ndarray:
    """Per-chunk STFT stacked as (num_chunks, freq_bins) when each chunk yields 1 frame."""
    if chunks.size == 0:
        return np.empty((0, N_FFT // 2 + 1), dtype=np.complex64)
    spectra = [stft_chunk(c, **kwargs) for c in chunks]
    stacked = np.stack(spectra, axis=0)            # (num_chunks, freq, frames)
    if stacked.shape[-1] == 1:
        stacked = stacked[..., 0]                  # squeeze single-frame axis
    return stacked.astype(np.complex64, copy=False)


def describe_signal(path: str, audio: np.ndarray, sr: int) -> None:
    duration = len(audio) / sr
    print(
        f"[load]  file={os.path.basename(path)}  sr={sr}  "
        f"samples={len(audio)}  duration={duration:.3f}s  dtype={audio.dtype}"
    )


def describe_chunks(chunks: np.ndarray) -> None:
    chunk_ms = CHUNK_SAMPLES / SR * 1000
    overlap_ms = CHUNK_OVERLAP / SR * 1000
    print(
        f"[chunk] chunk_size={CHUNK_SAMPLES} ({chunk_ms:.1f} ms)  "
        f"overlap={CHUNK_OVERLAP} ({overlap_ms:.1f} ms)  hop={CHUNK_HOP}  "
        f"num_chunks={chunks.shape[0]}  shape={tuple(chunks.shape)}"
    )


def describe_stft(spectra: np.ndarray) -> None:
    print(
        f"[stft]  n_fft={N_FFT}  hop={STFT_HOP}  win={WINDOW}  center=False  "
        f"shape={tuple(spectra.shape)}  dtype={spectra.dtype}"
    )
    if spectra.shape[0] > 0:
        mag0 = np.abs(spectra[0])
        print(
            f"        |X[0]| min={mag0.min():.4e}  "
            f"max={mag0.max():.4e}  mean={mag0.mean():.4e}"
        )


def run_demo(data_dir: str = DEFAULT_DATA_DIR) -> None:
    path = find_first_wav(data_dir)
    audio, sr = load_audio(path)
    describe_signal(path, audio, sr)

    chunks = chunk_waveform(audio)
    describe_chunks(chunks)

    spectra = stft_chunks(chunks)
    describe_stft(spectra)


if __name__ == "__main__":
    run_demo()
