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

from core.audio import (
    SR,
    analysis_stft_chunks,
    chunk_waveform,
    load_audio,
    overlap_add_average,
    synthesis_istft_chunks,
)
from core.model import GRUChunkDenoiser
from training.data import scan_wavs_by_basename

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
    parser.add_argument(
        "--gate-rmse-max",
        type=float,
        default=None,
        help="Optional fail threshold for noisy->out RMSE (lower is better).",
    )
    parser.add_argument(
        "--gate-peak-ratio-max",
        type=float,
        default=None,
        help="Optional fail threshold for max(|out|)/max(|noisy|).",
    )
    parser.add_argument(
        "--strict-length",
        action="store_true",
        help="Do not paste original tail back when output is shorter than input.",
    )
    parser.add_argument(
        "--pad-end-for-chunking",
        action="store_true",
        help="Zero-pad input so last chunk fully covers input tail before split.",
    )
    parser.add_argument(
        "--ola-min-weight",
        type=float,
        default=1e-3,
        help="Boundary stability threshold used during overlap-add normalization.",
    )
    parser.add_argument(
        "--boundary-pad-samples",
        type=int,
        default=0,
        help="Pad both sides before chunking, then crop back after reconstruction.",
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
    if not args.identity_mask and not os.path.isfile(weights):
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
    if chunk_waveform(noisy_wav).shape[0] == 0:
        print("Audio shorter than one chunk.", file=sys.stderr)
        sys.exit(1)

    if args.identity_mask:
        wav_out, mask = enhance_waveform(
            noisy_wav,
            model=None,
            device=torch.device("cpu"),
            log_eps=args.log_eps,
            identity_mask=True,
            preserve_input_tail=not args.strict_length,
            pad_end_for_chunking=args.pad_end_for_chunking,
            ola_min_weight=args.ola_min_weight,
            boundary_pad_samples=args.boundary_pad_samples,
        )
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
        wav_out, mask = enhance_waveform(
            noisy_wav,
            model,
            device,
            args.log_eps,
            preserve_input_tail=not args.strict_length,
            pad_end_for_chunking=args.pad_end_for_chunking,
            ola_min_weight=args.ola_min_weight,
            boundary_pad_samples=args.boundary_pad_samples,
        )
        print(
            "mask mode: model"
            f" ({os.path.basename(weights)})"
        )
        print(
            "mask min/mean/max: "
            f"{float(mask.min()):.4f} / {float(mask.mean()):.4f} / {float(mask.max()):.4f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(noisy_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"enhanced_{stem}_{ts}.wav")
    sf.write(out_path, wav_out, SR, subtype="PCM_16")

    print(f"picked: {noisy_path}")
    print(f"saved:  {out_path}")
    print(f"samples={len(wav_out)} sr={SR} chunks={mask.shape[0]}")
    stats = print_compare_metrics(noisy_wav, wav_out, "noisy->out")

    gate_failed = False
    peak_ratio = float(np.max(np.abs(wav_out)) / (np.max(np.abs(noisy_wav)) + 1e-12))
    print(f"peak_ratio={peak_ratio:.6f}")
    if args.gate_rmse_max is not None and stats["rmse"] > args.gate_rmse_max:
        gate_failed = True
        print(
            f"[gate-fail] rmse={stats['rmse']:.8f} exceeds gate_rmse_max={args.gate_rmse_max:.8f}",
            file=sys.stderr,
        )
    if args.gate_peak_ratio_max is not None and peak_ratio > args.gate_peak_ratio_max:
        gate_failed = True
        print(
            f"[gate-fail] peak_ratio={peak_ratio:.6f} exceeds gate_peak_ratio_max={args.gate_peak_ratio_max:.6f}",
            file=sys.stderr,
        )
    if gate_failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
