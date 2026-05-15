"""Random single-file evaluation: noisy wav -> denoise -> reconstruct -> save."""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime

import librosa
import numpy as np
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


def istft_one_chunk(spec_1d: np.ndarray) -> np.ndarray:
    spec = spec_1d[:, np.newaxis].astype(np.complex64, copy=False)
    wav = librosa.istft(
        spec,
        hop_length=STFT_HOP,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
        window=WINDOW,
        center=False,
        length=CHUNK_SAMPLES,
    )
    return wav.astype(np.float32, copy=False)


def overlap_add_from_chunks(chunk_signals: np.ndarray) -> np.ndarray:
    if chunk_signals.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    total = (chunk_signals.shape[0] - 1) * CHUNK_HOP + chunk_signals.shape[1]
    out = np.zeros(total, dtype=np.float32)
    for i in range(chunk_signals.shape[0]):
        start = i * CHUNK_HOP
        out[start : start + chunk_signals.shape[1]] += chunk_signals[i]
    return out


def load_weights(model: GRUChunkDenoiser, weights_path: str, device: torch.device) -> None:
    obj = torch.load(weights_path, map_location=device)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise one random test wav and save output.")
    parser.add_argument("--noisy-root", default=_DEFAULT_NOISY_ROOT)
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

    enhanced_spec = mask * noisy_spec
    chunk_out = np.stack([istft_one_chunk(enhanced_spec[i]) for i in range(enhanced_spec.shape[0])])
    wav_out = overlap_add_from_chunks(chunk_out)
    if len(wav_out) > len(noisy_wav):
        wav_out = wav_out[: len(noisy_wav)]
    elif len(wav_out) < len(noisy_wav):
        z = np.zeros(len(noisy_wav), dtype=np.float32)
        z[: len(wav_out)] = wav_out
        wav_out = z

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(noisy_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"enhanced_{stem}_{ts}.wav")
    sf.write(out_path, wav_out, SR, subtype="PCM_16")

    print(f"picked: {noisy_path}")
    print(f"saved:  {out_path}")
    print(f"samples={len(wav_out)} sr={SR} chunks={chunks.shape[0]}")


if __name__ == "__main__":
    main()
