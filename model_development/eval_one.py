"""Single-file evaluation: noisy wav -> denoise -> save."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

from core.audio import (
    SR,
    analysis_stft_chunks,
    chunk_waveform,
    load_audio,
    overlap_add_average,
    synthesis_istft_chunks,
)
from core.model import GRUChunkDenoiser

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NOISY_FILE = os.path.normpath(
    os.path.join(_BASE, "..", "data", "test", "noisy_testset_wav", "p257_404.wav")
)
_DEFAULT_WEIGHTS = os.path.normpath(
    os.path.join(
        _BASE,
        "runs",
        "20260522_001000_gru_h128_L3_bs16_lr1e-05_resume_resume",
        "best_weights.pt",
    )
)
_DEFAULT_OUT_DIR = os.path.join(_BASE, "eval_outputs")
_LOG_EPS = 1e-8

def load_weights(model: GRUChunkDenoiser, weights_path: str, device: torch.device) -> None:
    obj = torch.load(weights_path, map_location=device)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)


def enhance_waveform(
    noisy_wav: np.ndarray,
    model: GRUChunkDenoiser | None,
    device: torch.device,
    log_eps: float,
    *,
    identity_mask: bool = False,
    preserve_input_tail: bool = True,
    pad_end_for_chunking: bool = False,
    ola_min_weight: float = 1e-3,
    boundary_pad_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    work_wav = noisy_wav.astype(np.float32, copy=False)
    if boundary_pad_samples > 0:
        work_wav = np.pad(work_wav, (boundary_pad_samples, boundary_pad_samples))

    chunks = chunk_waveform(work_wav, pad_end=pad_end_for_chunking)
    if chunks.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    noisy_spec = analysis_stft_chunks(chunks)
    mag = np.abs(noisy_spec)
    if identity_mask:
        mask = np.ones_like(mag, dtype=np.float32)
    else:
        if model is None:
            raise ValueError("model is required when identity_mask=False")
        x_np = np.log(mag + log_eps).astype(np.float32, copy=False)
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
        lengths = torch.tensor([x_np.shape[0]], dtype=torch.long)
        with torch.no_grad():
            mask = model(x, lengths).squeeze(0).cpu().numpy()

    enhanced_spec = mask * noisy_spec
    chunk_out = synthesis_istft_chunks(enhanced_spec)
    wav_out = overlap_add_average(chunk_out, min_weight=ola_min_weight)
    if len(wav_out) > len(work_wav):
        wav_out = wav_out[: len(work_wav)]
    elif len(wav_out) < len(work_wav) and preserve_input_tail:
        wav_out = np.concatenate([wav_out, work_wav[len(wav_out) :].astype(np.float32, copy=False)])

    if boundary_pad_samples > 0:
        start = boundary_pad_samples
        stop = start + len(noisy_wav)
        wav_out = wav_out[start:stop]
    return wav_out, mask


def print_compare_metrics(ref: np.ndarray, est: np.ndarray, label: str) -> None:
    n = min(len(ref), len(est))
    if n == 0:
        print(f"{label} metrics: empty")
        return {"aligned": 0, "rmse": float("nan"), "snr_db": float("nan"), "max_abs_diff": float("nan")}
    a = ref[:n].astype(np.float64, copy=False)
    b = est[:n].astype(np.float64, copy=False)
    err = a - b
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    snr = float(10.0 * np.log10((np.mean(a * a) + 1e-20) / (mse + 1e-20)))
    max_abs = float(np.max(np.abs(err)))
    print(
        f"{label} metrics: aligned={n}  rmse={rmse:.8f}  "
        f"snr_vs_noisy={snr:.2f}dB  max_abs_diff={max_abs:.6f}"
    )
    return {"aligned": n, "rmse": rmse, "snr_db": snr, "max_abs_diff": max_abs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise one selected wav and save output.")
    parser.add_argument("--noisy-file", default=_DEFAULT_NOISY_FILE)
    parser.add_argument("--weights", default=_DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--log-eps", type=float, default=_LOG_EPS)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--save-noisy", action="store_true")
    args = parser.parse_args()

    noisy_file = os.path.abspath(args.noisy_file)
    weights = os.path.abspath(args.weights)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(noisy_file):
        print(f"Noisy file not found: {noisy_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(weights):
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)

    noisy_wav, _ = load_audio(noisy_file)
    if chunk_waveform(noisy_wav).shape[0] == 0:
        print("Audio shorter than one chunk.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    model = GRUChunkDenoiser(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    load_weights(model, weights, device)
    model.eval()
    wav_out, mask = enhance_waveform(
        noisy_wav,
        model,
        device,
        args.log_eps,
        preserve_input_tail=False,
        pad_end_for_chunking=True,
        ola_min_weight=0.0,
        boundary_pad_samples=240,
    )

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(noisy_file))[0]
    out_path = os.path.join(out_dir, f"enhanced_one_{stem}.wav")
    sf.write(out_path, wav_out, SR, subtype="PCM_16")
    if args.save_noisy:
        noisy_out = os.path.join(out_dir, f"noisy_one_{stem}.wav")
        sf.write(noisy_out, noisy_wav, SR, subtype="PCM_16")
        print(f"saved:  {noisy_out}")

    print(f"file:   {noisy_file}")
    print(f"saved:  {out_path}")
    print(f"samples={len(wav_out)} sr={SR} chunks={mask.shape[0]}")
    print_compare_metrics(noisy_wav, wav_out, "noisy->out")


if __name__ == "__main__":
    main()
