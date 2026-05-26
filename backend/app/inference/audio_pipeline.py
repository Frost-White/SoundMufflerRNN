from __future__ import annotations

import numpy as np
import torch

SR = 48000
CHUNK_SAMPLES = 960
CHUNK_OVERLAP = 240
CHUNK_HOP = CHUNK_SAMPLES - CHUNK_OVERLAP
N_FFT = 960
WIN_LENGTH = 960
STFT_HOP = 240
WINDOW = "hann"


def _hann_window_torch(win_length: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.hann_window(win_length, periodic=True, dtype=dtype, device=device)


def _hann_window_np(win_length: int) -> np.ndarray:
    return _hann_window_torch(
        win_length,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).cpu().numpy()


def chunk_waveform(
    waveform: np.ndarray,
    chunk_size: int = CHUNK_SAMPLES,
    hop: int = CHUNK_HOP,
    *,
    pad_end: bool = False,
) -> np.ndarray:
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1-D waveform, got shape {waveform.shape}")
    if pad_end:
        if len(waveform) == 0:
            return np.empty((0, chunk_size), dtype=waveform.dtype)
        if len(waveform) <= chunk_size:
            padded = np.pad(waveform, (0, chunk_size - len(waveform)))
            return np.ascontiguousarray(padded[None, :])
        n_chunks = int(np.ceil((len(waveform) - chunk_size) / hop)) + 1
        target_len = (n_chunks - 1) * hop + chunk_size
        if target_len > len(waveform):
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
    if len(waveform) < chunk_size:
        return np.empty((0, chunk_size), dtype=waveform.dtype)
    windows = np.lib.stride_tricks.sliding_window_view(waveform, chunk_size)
    return np.ascontiguousarray(windows[::hop])


def analysis_stft_chunks_torch(
    chunks: torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = STFT_HOP,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
    center: bool = False,
) -> torch.Tensor:
    if chunks.numel() == 0:
        return torch.empty((0, n_fft // 2 + 1), dtype=torch.complex64, device=chunks.device)
    x = chunks.to(dtype=torch.float32)
    if window != "hann":
        raise ValueError(f"Unsupported window: {window}")
    win = _hann_window_torch(win_length, dtype=x.dtype, device=x.device)
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
    if chunks.size == 0:
        return np.empty((0, n_fft // 2 + 1), dtype=np.complex64)
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
    length: int = CHUNK_SAMPLES,
) -> torch.Tensor:
    if spectra.numel() == 0:
        return torch.zeros((0, length), dtype=torch.float32, device=spectra.device)
    half = spectra.to(dtype=torch.complex64)
    mirrored = torch.conj(torch.flip(half[:, 1:-1], dims=[1]))
    full = torch.cat([half, mirrored], dim=1)
    chunks_windowed = torch.fft.ifft(full, n=n_fft, dim=1).real.to(dtype=torch.float32)  # pylint: disable=not-callable
    return chunks_windowed[:, :length]


def synthesis_istft_chunks(
    spectra: np.ndarray,
    n_fft: int = N_FFT,
    length: int = CHUNK_SAMPLES,
) -> np.ndarray:
    if spectra.size == 0:
        return np.zeros((0, length), dtype=np.float32)
    half = torch.from_numpy(np.ascontiguousarray(spectra.astype(np.complex64, copy=False)))
    chunks = synthesis_istft_chunks_torch(half, n_fft=n_fft, length=length)
    return np.ascontiguousarray(chunks.cpu().numpy().astype(np.float32, copy=False))


def project_to_stft_consistency_torch(spectra: torch.Tensor) -> torch.Tensor:
    if spectra.numel() == 0:
        return spectra
    if spectra.ndim == 2:
        flat = spectra
        out_shape = spectra.shape
    elif spectra.ndim == 3:
        out_shape = spectra.shape
        flat = spectra.reshape(-1, spectra.shape[-1])
    else:
        raise ValueError(f"Expected (N,F) or (B,T,F), got {tuple(spectra.shape)}")
    n_fft = int((flat.shape[-1] - 1) * 2)
    half = flat.to(dtype=torch.complex64)
    mirrored = torch.conj(torch.flip(half[:, 1:-1], dims=[1]))
    full = torch.cat([half, mirrored], dim=1)
    chunks_windowed = torch.fft.ifft(full, n=n_fft, dim=1).real.to(dtype=torch.float32)  # pylint: disable=not-callable
    win = _hann_window_torch(n_fft, dtype=chunks_windowed.dtype, device=chunks_windowed.device)
    chunks = chunks_windowed / torch.clamp(win.unsqueeze(0), min=1e-3)
    projected = analysis_stft_chunks_torch(chunks, n_fft=n_fft, hop_length=STFT_HOP, win_length=n_fft)
    return projected.reshape(out_shape)


def project_to_stft_consistency(spectra: np.ndarray) -> np.ndarray:
    if spectra.size == 0:
        return spectra
    x = torch.from_numpy(np.ascontiguousarray(spectra.astype(np.complex64, copy=False)))
    projected = project_to_stft_consistency_torch(x).cpu().numpy()
    return np.ascontiguousarray(projected.astype(np.complex64, copy=False))


def blend_consistent_spectra(
    raw_spectra: np.ndarray,
    projected_spectra: np.ndarray,
    blend: float = 1.0,
) -> np.ndarray:
    a = float(max(0.0, min(1.0, blend)))
    return np.ascontiguousarray((raw_spectra + a * (projected_spectra - raw_spectra)).astype(np.complex64))


def overlap_add_average(
    chunks: np.ndarray,
    hop: int = CHUNK_HOP,
    *,
    synthesis_window: str = WINDOW,
    win_length: int = CHUNK_SAMPLES,
    min_weight: float = 1e-3,
) -> np.ndarray:
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
    stable = weight > 0.0 if min_weight <= 0.0 else weight >= min_weight
    out[stable] /= weight[stable]
    out[~stable] = 0.0
    return out
