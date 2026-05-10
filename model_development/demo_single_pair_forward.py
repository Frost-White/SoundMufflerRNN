"""Pick one noisy wav, locate clean twin by basename, STFT -> TinyDenoiser forward."""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import librosa
import soundfile as sf
import torch

from audio_pipeline import (
    CHUNK_HOP,
    CHUNK_SAMPLES,
    N_FFT,
    SR,
    STFT_HOP,
    WINDOW,
    WIN_LENGTH,
    chunk_waveform,
    load_audio,
    stft_chunks,
)
from model import FREQ_BINS, TinyDenoiser

_LOG_EPS = 1e-8

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NOISY = os.path.normpath(
    os.path.join(_BASE, "..", "data", "train", "noisy_trainset_56spk_wav")
)
_DEFAULT_CLEAN = os.path.normpath(
    os.path.join(_BASE, "..", "data", "train", "clean_trainset_56spk_wav")
)


def collect_wavs(root: str) -> list[str]:
    out: list[str] = []
    for cur, _, names in os.walk(root):
        for name in sorted(names):
            if name.lower().endswith(".wav"):
                out.append(os.path.join(cur, name))
    return out


def find_wav_by_basename(root: str, basename: str) -> str | None:
    for cur, _, names in os.walk(root):
        if basename in names:
            return os.path.join(cur, basename)
    return None


def istft_one_chunk(spec_1d: np.ndarray) -> np.ndarray:
    """Inverse STFT for one row (freq_bins,) matching audio_pipeline.stft_chunk."""
    S = spec_1d[:, np.newaxis].astype(np.complex64, copy=False)
    y = librosa.istft(
        S,
        hop_length=STFT_HOP,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
        window=WINDOW,
        center=False,
        length=CHUNK_SAMPLES,
    )
    return y.astype(np.float32, copy=False)


def overlap_add_from_chunks(chunk_signals: np.ndarray) -> np.ndarray:
    """Sum overlapping chunk waveforms (same framing as chunk_waveform)."""
    n, wlen = chunk_signals.shape
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    total = (n - 1) * CHUNK_HOP + wlen
    out = np.zeros(total, dtype=np.float32)
    for i in range(n):
        start = i * CHUNK_HOP
        out[start : start + wlen] += chunk_signals[i]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="One pair STFT + random-weight model forward.")
    p.add_argument("--noisy-root", default=_DEFAULT_NOISY)
    p.add_argument("--clean-root", default=_DEFAULT_CLEAN)
    p.add_argument("--noisy-file", default=None, help="Exact noisy wav path (skip random pick).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for file pick + model init.")
    p.add_argument(
        "--out",
        default=None,
        help="Output wav path (default: model_development/demo_reconstructed_<stem>.wav).",
    )
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.noisy_file:
        noisy_path = os.path.abspath(args.noisy_file)
        if not os.path.isfile(noisy_path):
            print(f"Not found: {noisy_path}", file=sys.stderr)
            sys.exit(1)
    else:
        wavs = collect_wavs(args.noisy_root)
        if not wavs:
            print(f"No .wav under: {args.noisy_root}", file=sys.stderr)
            sys.exit(1)
        noisy_path = random.choice(wavs)

    base = os.path.basename(noisy_path)
    clean_path = find_wav_by_basename(args.clean_root, base)
    if clean_path is None:
        print(
            f"No clean twin for {base!r} under {args.clean_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"noisy:  {noisy_path}")
    print(f"clean:  {clean_path}")

    noisy_wav, sr = load_audio(noisy_path)
    chunks = chunk_waveform(noisy_wav)
    spectra = stft_chunks(chunks)
    if spectra.shape[0] == 0:
        print("No STFT chunks (audio shorter than one chunk).", file=sys.stderr)
        sys.exit(1)
    mag = np.abs(spectra)
    log_mag = np.log(mag + _LOG_EPS).astype(np.float32)

    print(f"chunks={chunks.shape[0]}  stft={tuple(spectra.shape)}  freq_bins={FREQ_BINS}")

    x = torch.from_numpy(log_mag)
    model = TinyDenoiser()
    with torch.no_grad():
        mask = model(x)

    mask_np = mask.cpu().numpy()
    enhanced_spec = mask_np * spectra

    chunk_out = np.stack([istft_one_chunk(enhanced_spec[i]) for i in range(enhanced_spec.shape[0])])
    wav_out = overlap_add_from_chunks(chunk_out)
    n_orig = len(noisy_wav)
    if len(wav_out) > n_orig:
        wav_out = wav_out[:n_orig]
    elif len(wav_out) < n_orig:
        z = np.zeros(n_orig, dtype=np.float32)
        z[: len(wav_out)] = wav_out
        wav_out = z

    stem = os.path.splitext(base)[0]
    out_path = (
        os.path.abspath(args.out)
        if args.out
        else os.path.join(_BASE, f"demo_reconstructed_{stem}.wav")
    )
    _dir = os.path.dirname(out_path)
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    sf.write(out_path, wav_out, SR, subtype="PCM_16")

    print(f"model in  {tuple(x.shape)}  ->  out {tuple(mask.shape)}")
    print(
        f"mask min/mean/max: {mask.min().item():.4f} / "
        f"{mask.mean().item():.4f} / {mask.max().item():.4f}"
    )
    print(f"saved: {out_path}  samples={len(wav_out)}  sr={SR}")


if __name__ == "__main__":
    main()
