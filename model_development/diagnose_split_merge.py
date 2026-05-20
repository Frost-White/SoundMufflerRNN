"""Stage-wise split/merge reconstruction diagnostics."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from core.audio import (
    CHUNK_OVERLAP,
    SR,
    analysis_stft_chunks,
    chunk_waveform,
    load_audio,
    overlap_add_average,
    synthesis_istft_chunks,
)
from eval_one import enhance_waveform


def _build_probe_signal(duration_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float32) / float(SR)
    probe = (
        0.18 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.07 * np.sin(2.0 * np.pi * 1450.0 * t)
        + 0.02 * rng.standard_normal(n)
    )
    return probe.astype(np.float32, copy=False)


def _calc_metrics(ref: np.ndarray, out: np.ndarray) -> dict[str, float]:
    aligned = min(len(ref), len(out))
    if aligned == 0:
        return {
            "aligned": 0.0,
            "rmse": float("nan"),
            "max_abs_diff": float("nan"),
            "snr_db": float("nan"),
            "len_diff": float(len(out) - len(ref)),
            "tail_uncovered": float(max(0, len(ref) - len(out))),
        }

    a = ref[:aligned].astype(np.float64, copy=False)
    b = out[:aligned].astype(np.float64, copy=False)
    err = a - b
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    snr_db = float(10.0 * np.log10((np.mean(a * a) + 1e-20) / (mse + 1e-20)))
    return {
        "aligned": float(aligned),
        "rmse": rmse,
        "max_abs_diff": float(np.max(np.abs(err))),
        "snr_db": snr_db,
        "len_diff": float(len(out) - len(ref)),
        "tail_uncovered": float(max(0, len(ref) - len(out))),
    }


def _stage_ola_only(signal: np.ndarray, pad_end: bool, boundary_pad: int) -> tuple[np.ndarray, dict[str, float]]:
    work = np.pad(signal, (boundary_pad, boundary_pad)) if boundary_pad > 0 else signal
    chunks = chunk_waveform(work, pad_end=pad_end)
    recon = overlap_add_average(chunks, synthesis_window="rect", min_weight=0.0)
    if pad_end and len(recon) > len(work):
        recon = recon[: len(work)]
    if boundary_pad > 0:
        recon = recon[boundary_pad : boundary_pad + len(signal)]
    return recon, {"num_chunks": float(chunks.shape[0])}


def _stage_stft_roundtrip(signal: np.ndarray, pad_end: bool, boundary_pad: int) -> tuple[np.ndarray, dict[str, float]]:
    work = np.pad(signal, (boundary_pad, boundary_pad)) if boundary_pad > 0 else signal
    chunks = chunk_waveform(work, pad_end=pad_end)
    specs = analysis_stft_chunks(chunks)
    chunk_out = synthesis_istft_chunks(specs)
    recon = overlap_add_average(chunk_out, min_weight=0.0)
    if pad_end and len(recon) > len(work):
        recon = recon[: len(work)]
    if boundary_pad > 0:
        recon = recon[boundary_pad : boundary_pad + len(signal)]
    return recon, {"num_chunks": float(chunks.shape[0])}


def _stage_full_identity(signal: np.ndarray, pad_end: bool, boundary_pad: int) -> tuple[np.ndarray, dict[str, float]]:
    recon, mask = enhance_waveform(
        signal,
        model=None,
        device=torch.device("cpu"),
        log_eps=1e-8,
        identity_mask=True,
        preserve_input_tail=False,
        pad_end_for_chunking=pad_end,
        ola_min_weight=0.0,
        boundary_pad_samples=boundary_pad,
    )
    return recon, {"num_chunks": float(mask.shape[0])}


def _print_stage(label: str, metrics: dict[str, float], extra: dict[str, float], passed: bool) -> None:
    print(
        f"[{label}] pass={passed} chunks={int(extra['num_chunks'])} "
        f"len_diff={int(metrics['len_diff'])} tail_uncovered={int(metrics['tail_uncovered'])} "
        f"rmse={metrics['rmse']:.8f} max_abs_diff={metrics['max_abs_diff']:.8f} snr_db={metrics['snr_db']:.2f}"
    )


def _passes(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    if abs(metrics["len_diff"]) > float(args.gate_length_diff_max):
        return False
    if metrics["rmse"] > float(args.gate_rmse_max):
        return False
    if metrics["max_abs_diff"] > float(args.gate_max_abs_diff):
        return False
    if args.gate_snr_min_db is not None and metrics["snr_db"] < float(args.gate_snr_min_db):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose split/merge reconstruction stages.")
    parser.add_argument("--input-wav", default=None, help="Optional probe wav. If omitted, synthetic probe is used.")
    parser.add_argument("--probe-duration-s", type=float, default=1.37)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pad-end",
        dest="pad_end",
        action="store_true",
        default=True,
        help="Pad input tail before chunking to fully cover samples.",
    )
    parser.add_argument(
        "--no-pad-end",
        dest="pad_end",
        action="store_false",
        help="Disable tail padding (useful to expose tail-coverage issues).",
    )
    parser.add_argument(
        "--boundary-pad-samples",
        type=int,
        default=CHUNK_OVERLAP,
        help="Pad both sides before split; helps boundary reconstruction with Hann windows.",
    )
    parser.add_argument("--gate-length-diff-max", type=int, default=0)
    parser.add_argument("--gate-rmse-max", type=float, default=1e-5)
    parser.add_argument("--gate-max-abs-diff", type=float, default=1e-4)
    parser.add_argument("--gate-snr-min-db", type=float, default=80.0)
    args = parser.parse_args()

    if args.input_wav:
        probe_path = os.path.abspath(args.input_wav)
        if not os.path.isfile(probe_path):
            print(f"Input wav not found: {probe_path}", file=sys.stderr)
            sys.exit(1)
        signal, _ = load_audio(probe_path)
        source = probe_path
    else:
        signal = _build_probe_signal(args.probe_duration_s, args.seed)
        source = f"synthetic(duration={args.probe_duration_s:.2f}s,seed={args.seed})"
    signal = signal.astype(np.float32, copy=False)

    print(f"source={source}")
    print(
        f"samples={len(signal)} sr={SR} pad_end={args.pad_end} "
        f"boundary_pad_samples={args.boundary_pad_samples}"
    )

    ola_out, ola_extra = _stage_ola_only(signal, args.pad_end, args.boundary_pad_samples)
    stft_out, stft_extra = _stage_stft_roundtrip(signal, args.pad_end, args.boundary_pad_samples)
    full_out, full_extra = _stage_full_identity(signal, args.pad_end, args.boundary_pad_samples)

    m_ola = _calc_metrics(signal, ola_out)
    m_stft = _calc_metrics(signal, stft_out)
    m_full = _calc_metrics(signal, full_out)

    p_ola = _passes(m_ola, args)
    p_stft = _passes(m_stft, args)
    p_full = _passes(m_full, args)

    _print_stage("OLA-only", m_ola, ola_extra, p_ola)
    _print_stage("STFT-roundtrip", m_stft, stft_extra, p_stft)
    _print_stage("Full-identity", m_full, full_extra, p_full)

    if not p_ola:
        if m_ola["tail_uncovered"] > 0:
            print("root_cause_hint=tail coverage gap (enable --pad-end)")
        else:
            print("root_cause_hint=OLA window/hop normalization mismatch")
        sys.exit(2)
    if not p_stft:
        if args.boundary_pad_samples == 0:
            print("root_cause_hint=Hann boundary loss with center=False (add boundary padding)")
        else:
            print("root_cause_hint=STFT/iSTFT consistency mismatch (window/scale/symmetry)")
        sys.exit(2)
    if not p_full:
        print("root_cause_hint=full identity path differs from staged reconstruction behavior")
        sys.exit(2)

    print("result=PASS all stages")


if __name__ == "__main__":
    main()
