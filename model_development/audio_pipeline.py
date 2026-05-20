"""Audio I/O + chunking + per-chunk STFT pipeline."""

import os
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

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
    spec = analysis_stft_chunks(
        np.ascontiguousarray(chunk.astype(np.float32, copy=False))[None, :],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
    )
    return spec[0][:, None]


def _hann_window_torch(win_length: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Use a single window definition in both analysis and synthesis.
    return torch.hann_window(win_length, periodic=True, dtype=dtype, device=device)


def _hann_window_np(win_length: int) -> np.ndarray:
    return _hann_window_torch(win_length, dtype=torch.float32, device=torch.device("cpu")).cpu().numpy()


def analysis_stft_chunks_torch(
    chunks: torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = STFT_HOP,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
    center: bool = False,
) -> torch.Tensor:
    """Analyze chunks into one-frame complex STFT bins: (num_chunks, n_fft//2+1)."""
    if chunks.numel() == 0:
        return torch.empty((0, n_fft // 2 + 1), dtype=torch.complex64, device=chunks.device)
    if chunks.ndim != 2:
        raise ValueError(f"Expected (num_chunks, chunk_samples), got {tuple(chunks.shape)}")

    x = chunks.to(dtype=torch.float32)
    if window == "hann":
        win = _hann_window_torch(win_length, dtype=x.dtype, device=x.device)
    else:
        raise ValueError(f"Unsupported window: {window}")
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=win,
        center=center,
        return_complex=True,
    )
    if spec.shape[-1] != 1:
        raise ValueError(f"Expected single STFT frame per chunk, got {spec.shape}")
    return spec[..., 0]


def analysis_stft_chunks(
    chunks: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = STFT_HOP,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
    center: bool = False,
) -> np.ndarray:
    """Analyze chunks into one-frame complex STFT bins: (num_chunks, n_fft//2+1)."""
    if chunks.size == 0:
        return np.empty((0, n_fft // 2 + 1), dtype=np.complex64)
    if chunks.ndim != 2:
        raise ValueError(f"Expected (num_chunks, chunk_samples), got {chunks.shape}")

    x = torch.from_numpy(np.ascontiguousarray(chunks.astype(np.float32, copy=False)))
    spec = analysis_stft_chunks_torch(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )
    return spec.cpu().numpy().astype(np.complex64, copy=False)


def synthesis_istft_chunks_torch(
    spectra: torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = STFT_HOP,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
    center: bool = False,
    length: int = CHUNK_SAMPLES,
) -> torch.Tensor:
    """
    Synthesize chunk signals from one-frame complex STFT bins: (num_chunks, chunk_samples).

    This returns a stable, windowed reconstruction. Final waveform normalization
    should be handled in overlap-add using the same synthesis window.
    """
    if spectra.numel() == 0:
        return torch.zeros((0, length), dtype=torch.float32, device=spectra.device)
    if spectra.ndim != 2:
        raise ValueError(f"Expected (num_chunks, freq_bins), got {tuple(spectra.shape)}")
    if spectra.shape[1] != n_fft // 2 + 1:
        raise ValueError(f"Unexpected freq bins: {spectra.shape[1]} (expected {n_fft // 2 + 1})")

    half = spectra.to(dtype=torch.complex64)
    mirrored = torch.conj(torch.flip(half[:, 1:-1], dims=[1]))
    full = torch.cat([half, mirrored], dim=1)
    chunks_windowed = torch.fft.ifft(full, n=n_fft, dim=1).real.to(dtype=torch.float32)
    return chunks_windowed[:, :length]


def synthesis_istft_chunks(
    spectra: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = STFT_HOP,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
    center: bool = False,
    length: int = CHUNK_SAMPLES,
) -> np.ndarray:
    """Synthesize chunk waveforms from one-frame complex STFT bins: (num_chunks, chunk_samples)."""
    if spectra.size == 0:
        return np.zeros((0, length), dtype=np.float32)
    if spectra.ndim != 2:
        raise ValueError(f"Expected (num_chunks, freq_bins), got {spectra.shape}")

    if window != "hann":
        raise ValueError(f"Unsupported window: {window}")
    if win_length != n_fft:
        raise ValueError("synthesis_istft_chunks currently expects win_length == n_fft")

    half = torch.from_numpy(np.ascontiguousarray(spectra.astype(np.complex64, copy=False)))
    chunks = synthesis_istft_chunks_torch(
        half,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    )
    return np.ascontiguousarray(chunks.cpu().numpy().astype(np.float32, copy=False))


def overlap_add_average(
    chunks: np.ndarray,
    hop: int = CHUNK_HOP,
    *,
    synthesis_window: str = WINDOW,
    win_length: int = CHUNK_SAMPLES,
) -> np.ndarray:
    """Overlap-add chunks with window-aware normalization."""
    if chunks.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    n_chunks, wlen = chunks.shape
    total = (n_chunks - 1) * hop + wlen
    out = np.zeros(total, dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    if synthesis_window == "hann":
        win = _hann_window_np(win_length)
    else:
        raise ValueError(f"Unsupported window: {synthesis_window}")
    win = win[:wlen]
    for i in range(n_chunks):
        start = i * hop
        out[start : start + wlen] += chunks[i]
        weight[start : start + wlen] += win
    # Guard boundary regions where Hann weight is ~0 to avoid blow-ups
    # when masked spectra produce non-zero edge samples.
    min_weight = 1e-3
    stable = weight >= min_weight
    out[stable] /= weight[stable]
    out[~stable] = 0.0
    return out


def stft_chunks(chunks: np.ndarray, **kwargs) -> np.ndarray:
    """Per-chunk STFT stacked as (num_chunks, freq_bins) when each chunk yields 1 frame."""
    return analysis_stft_chunks(chunks, **kwargs)


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
