"""Random single-file evaluation: noisy wav -> denoise -> reconstruct -> save."""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import soundfile as sf
import torch

from audio_pipeline import (
    CHUNK_HOP,
    CHUNK_SAMPLES,
    N_FFT,
    SR,
    chunk_waveform,
    load_audio,
    stft_chunks,
)
from model import GRUChunkDenoiser
from training_data import scan_wavs_by_basename

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NOISY_ROOT = os.path.normpath(
    os.path.join(_BASE, "..", "data", "test", "noisy_testset_wav")
)
_DEFAULT_WEIGHTS = os.path.normpath(
    os.path.join(
        _BASE,
        "runs",
        "20260514_231036_gru_h128_L3_bs16_lr0.001",
        "best_weights.pt",
    )
)
_DEFAULT_OUT_DIR = os.path.join(_BASE, "eval_outputs")
_LOG_EPS = 1e-8


def pick_random_wav_path(noisy_root: str) -> str:
    wav_map = scan_wavs_by_basename(noisy_root)
    if not wav_map:
        raise FileNotFoundError(f"No .wav under: {noisy_root}")
    return random.choice(list(wav_map.values()))


def overlap_add_average_from_chunks(chunk_signals: np.ndarray) -> np.ndarray:
    if chunk_signals.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    n_chunks, wlen = chunk_signals.shape
    total = (n_chunks - 1) * CHUNK_HOP + wlen
    out = np.zeros(total, dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    ones = np.ones(wlen, dtype=np.float32)
    for i in range(n_chunks):
        start = i * CHUNK_HOP
        out[start : start + wlen] += chunk_signals[i]
        weight[start : start + wlen] += ones
    out /= np.clip(weight, 1.0, None)
    return out


def load_weights(model: GRUChunkDenoiser, weights_path: str, device: torch.device) -> None:
    obj = torch.load(weights_path, map_location=device)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)


def print_compare_metrics(ref: np.ndarray, est: np.ndarray, label: str) -> None:
    n = min(len(ref), len(est))
    if n == 0:
        print(f"{label} metrics: empty")
        return
    a = ref[:n].astype(np.float64, copy=False)
    b = est[:n].astype(np.float64, copy=False)
    err = a - b
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    snr = float(10.0 * np.log10((np.mean(a * a) + 1e-20) / (mse + 1e-20)))
    print(
        f"{label} metrics: aligned={n}  rmse={rmse:.8f}  "
        f"snr_vs_noisy={snr:.2f}dB  max_abs_diff={float(np.max(np.abs(err))):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise one random test wav and save output.")
    parser.add_argument("--noisy-root", default=_DEFAULT_NOISY_ROOT)
    parser.add_argument("--noisy-file", default=None, help="Exact noisy wav path (skip random pick).")
    parser.add_argument("--weights", default=_DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--log-eps", type=float, default=_LOG_EPS)
    parser.add_argument(
        "--identity-mask",
        action="store_true",
        help="Bypass model and use all-ones mask for reconstruction debug.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    noisy_root = os.path.abspath(args.noisy_root)
    weights = os.path.abspath(args.weights)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(noisy_root):
        print(f"Noisy root not found: {noisy_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(weights):
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)

    if args.noisy_file:
        noisy_path = os.path.abspath(args.noisy_file)
        if not os.path.isfile(noisy_path):
            print(f"Noisy file not found: {noisy_path}", file=sys.stderr)
            sys.exit(1)
    else:
        noisy_path = pick_random_wav_path(noisy_root)
    noisy_wav, _ = load_audio(noisy_path)
    chunks = chunk_waveform(noisy_wav)
    if chunks.shape[0] == 0:
        print("Audio shorter than one chunk.", file=sys.stderr)
        sys.exit(1)
    noisy_spec = stft_chunks(chunks)
    mag = np.abs(noisy_spec)
    x_np = np.log(mag + args.log_eps).astype(np.float32)

    if args.identity_mask:
        mask = np.ones_like(mag, dtype=np.float32)
        print("mask mode: identity (all ones)")
        print(
            "mask min/mean/max: "
            f"{float(mask.min()):.4f} / {float(mask.mean()):.4f} / {float(mask.max()):.4f}"
        )
    else:
        device = torch.device(args.device)
        model = GRUChunkDenoiser(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
        load_weights(model, weights, device)
        model.eval()

        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
        lengths = torch.tensor([x_np.shape[0]], dtype=torch.long)
        with torch.no_grad():
            mask = model(x, lengths).squeeze(0).cpu().numpy()
        print(
            "mask mode: model"
            f" ({os.path.basename(weights)})"
        )
        print(
            "mask min/mean/max: "
            f"{float(mask.min()):.4f} / {float(mask.mean()):.4f} / {float(mask.max()):.4f}"
        )

    synth_spec = np.fft.rfft(chunks, n=N_FFT, axis=1).astype(np.complex64, copy=False)
    enhanced_spec = mask * synth_spec
    chunk_out = np.fft.irfft(enhanced_spec, n=N_FFT, axis=1).astype(np.float32, copy=False)
    wav_out = overlap_add_average_from_chunks(chunk_out)
    if len(wav_out) > len(noisy_wav):
        wav_out = wav_out[: len(noisy_wav)]
    elif len(wav_out) < len(noisy_wav):
        # Preserve non-covered tail from original to avoid zero-padding artifacts.
        wav_out = np.concatenate([wav_out, noisy_wav[len(wav_out) :].astype(np.float32, copy=False)])

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(noisy_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"enhanced_{stem}_{ts}.wav")
    sf.write(out_path, wav_out, SR, subtype="PCM_16")

    print(f"picked: {noisy_path}")
    print(f"saved:  {out_path}")
    print(f"samples={len(wav_out)} sr={SR} chunks={chunks.shape[0]}")
    print_compare_metrics(noisy_wav, wav_out, "noisy->out")


if __name__ == "__main__":
    main()
