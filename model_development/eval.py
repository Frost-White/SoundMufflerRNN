"""Batch evaluation on test noisy/clean pairs with objective metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Callable

import librosa
import numpy as np
import torch

from audio_pipeline import N_FFT, SR, chunk_waveform, load_audio, stft_chunks
from eval_one import load_weights, overlap_add_average_from_chunks
from model import GRUChunkDenoiser
from training_data import scan_wavs_by_basename

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NOISY_ROOT = os.path.normpath(
    os.path.join(_BASE, "..", "data", "test", "noisy_testset_wav")
)
_DEFAULT_CLEAN_ROOT = os.path.normpath(
    os.path.join(_BASE, "..", "data", "test", "clean_testset_wav")
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
_METRIC_SR = 16000
_EPS = 1e-20

try:
    from pystoi import stoi as stoi_fn

    _HAS_STOI = True
except Exception:
    _HAS_STOI = False

try:
    from pesq import pesq as pesq_fn  # type: ignore[reportMissingImports]

    _HAS_PESQ = True
except Exception:
    _HAS_PESQ = False

try:
    from torchmetrics.functional.audio import (
        scale_invariant_signal_distortion_ratio as tm_si_sdr,
        scale_invariant_signal_noise_ratio as tm_si_snr,
    )

    _HAS_TM = True
except Exception:
    _HAS_TM = False


@dataclass
class EvalRow:
    filename: str
    noisy_path: str
    clean_path: str
    num_samples: int
    duration_s: float
    snr_db: float
    si_sdr_db: float
    si_snr_db: float
    stoi: float
    pesq: float
    status: str
    error: str


def _to_metric_sr(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == _METRIC_SR:
        return x.astype(np.float32, copy=False)
    return librosa.resample(x.astype(np.float32, copy=False), orig_sr=src_sr, target_sr=_METRIC_SR)


def _snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    err = clean - est
    return float(10.0 * np.log10((np.mean(clean * clean) + _EPS) / (np.mean(err * err) + _EPS)))


def _si_sdr_numpy(clean: np.ndarray, est: np.ndarray) -> float:
    s = clean.astype(np.float64, copy=False)
    sh = est.astype(np.float64, copy=False)
    alpha = float(np.dot(sh, s) / (np.dot(s, s) + _EPS))
    s_target = alpha * s
    e_noise = sh - s_target
    return float(10.0 * np.log10((np.dot(s_target, s_target) + _EPS) / (np.dot(e_noise, e_noise) + _EPS)))


def _si_snr_numpy(clean: np.ndarray, est: np.ndarray) -> float:
    s = clean.astype(np.float64, copy=False)
    sh = est.astype(np.float64, copy=False)
    s = s - np.mean(s)
    sh = sh - np.mean(sh)
    alpha = float(np.dot(sh, s) / (np.dot(s, s) + _EPS))
    s_target = alpha * s
    e_noise = sh - s_target
    return float(10.0 * np.log10((np.dot(s_target, s_target) + _EPS) / (np.dot(e_noise, e_noise) + _EPS)))


def _si_metrics(clean: np.ndarray, est: np.ndarray) -> tuple[float, float]:
    if not _HAS_TM:
        return _si_sdr_numpy(clean, est), _si_snr_numpy(clean, est)
    s = torch.from_numpy(clean.astype(np.float32, copy=False)).unsqueeze(0)
    sh = torch.from_numpy(est.astype(np.float32, copy=False)).unsqueeze(0)
    si_sdr = float(tm_si_sdr(sh, s).item())
    si_snr = float(tm_si_snr(sh, s).item())
    return si_sdr, si_snr


def _pair_paths(noisy_root: str, clean_root: str) -> tuple[list[tuple[str, str, str]], int, int]:
    noisy_map = scan_wavs_by_basename(noisy_root)
    clean_map = scan_wavs_by_basename(clean_root)
    names = sorted(set(noisy_map) & set(clean_map))
    pairs = [(name, noisy_map[name], clean_map[name]) for name in names]
    return pairs, len(noisy_map), len(clean_map)


def _enhance_waveform(
    noisy: np.ndarray,
    model: GRUChunkDenoiser,
    device: torch.device,
    log_eps: float,
) -> np.ndarray:
    chunks = chunk_waveform(noisy)
    if chunks.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    noisy_spec = stft_chunks(chunks)
    mag = np.abs(noisy_spec)
    x_np = np.log(mag + log_eps).astype(np.float32, copy=False)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    lengths = torch.tensor([x_np.shape[0]], dtype=torch.long)

    with torch.no_grad():
        mask = model(x, lengths).squeeze(0).cpu().numpy()

    synth_spec = np.fft.rfft(chunks, n=N_FFT, axis=1).astype(np.complex64, copy=False)
    enhanced_spec = mask * synth_spec
    chunk_out = np.fft.irfft(enhanced_spec, n=N_FFT, axis=1).astype(np.float32, copy=False)
    wav_out = overlap_add_average_from_chunks(chunk_out)
    if len(wav_out) > len(noisy):
        wav_out = wav_out[: len(noisy)]
    elif len(wav_out) < len(noisy):
        wav_out = np.concatenate([wav_out, noisy[len(wav_out) :].astype(np.float32, copy=False)])
    return wav_out


def _safe_metric(fn: Callable[[], float]) -> tuple[float, str]:
    try:
        return float(fn()), ""
    except Exception as e:
        return float("nan"), str(e)


def _summary(rows: list[EvalRow]) -> dict:
    ok_rows = [r for r in rows if r.status == "ok"]

    def stats(key: str) -> dict:
        vals = [float(getattr(r, key)) for r in ok_rows if math.isfinite(float(getattr(r, key)))]
        if not vals:
            return {"count": 0, "mean": None, "median": None, "std": None}
        return {
            "count": len(vals),
            "mean": float(statistics.fmean(vals)),
            "median": float(statistics.median(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        }

    return {
        "total_files": len(rows),
        "ok_files": len(ok_rows),
        "error_files": len(rows) - len(ok_rows),
        "metrics": {
            "snr_db": stats("snr_db"),
            "si_sdr_db": stats("si_sdr_db"),
            "si_snr_db": stats("si_snr_db"),
            "stoi": stats("stoi"),
            "pesq": stats("pesq"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all noisy/clean test pairs.")
    parser.add_argument("--noisy-root", default=_DEFAULT_NOISY_ROOT)
    parser.add_argument("--clean-root", default=_DEFAULT_CLEAN_ROOT)
    parser.add_argument("--weights", default=_DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--log-eps", type=float, default=_LOG_EPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None, help="Limit pair count for quick checks.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save-wavs",
        action="store_true",
        help="Save enhanced wavs under eval_outputs/enhanced_wavs.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    noisy_root = os.path.abspath(args.noisy_root)
    clean_root = os.path.abspath(args.clean_root)
    weights = os.path.abspath(args.weights)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(noisy_root):
        raise FileNotFoundError(f"Noisy root not found: {noisy_root}")
    if not os.path.isdir(clean_root):
        raise FileNotFoundError(f"Clean root not found: {clean_root}")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics_eval.csv")
    summary_path = os.path.join(out_dir, "eval_summary.json")
    wav_dir = os.path.join(out_dir, "enhanced_wavs")
    if args.save_wavs:
        os.makedirs(wav_dir, exist_ok=True)

    pairs, noisy_n, clean_n = _pair_paths(noisy_root, clean_root)
    if args.max_files is not None:
        pairs = pairs[: max(args.max_files, 0)]
    if not pairs:
        raise RuntimeError("No paired noisy-clean files found.")

    device = torch.device(args.device)
    model = GRUChunkDenoiser(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    load_weights(model, weights, device)
    model.eval()

    print(f"[eval] noisy={noisy_n} clean={clean_n} paired={len(pairs)}")
    print(
        "[eval] "
        f"device={device} "
        f"torchmetrics={'on' if _HAS_TM else 'fallback_numpy'} "
        f"stoi={'on' if _HAS_STOI else 'off'} "
        f"pesq={'on' if _HAS_PESQ else 'off'}"
    )

    rows: list[EvalRow] = []
    for i, (name, noisy_path, clean_path) in enumerate(pairs, start=1):
        status = "ok"
        err_msgs: list[str] = []
        snr_db = si_sdr_db = si_snr_db = stoi_v = pesq_v = float("nan")
        try:
            noisy_wav, _ = load_audio(noisy_path)
            clean_wav, _ = load_audio(clean_path)
            L = min(len(noisy_wav), len(clean_wav))
            if L < N_FFT:
                raise RuntimeError("too_short_for_chunking")

            noisy = noisy_wav[:L].astype(np.float32, copy=False)
            clean = clean_wav[:L].astype(np.float32, copy=False)
            enhanced = _enhance_waveform(noisy, model, device, args.log_eps)
            n = min(len(clean), len(enhanced))
            clean = clean[:n]
            enhanced = enhanced[:n]

            snr_db = _snr_db(clean, enhanced)
            si_sdr_db, si_snr_db = _si_metrics(clean, enhanced)

            clean16 = _to_metric_sr(clean, SR)
            enh16 = _to_metric_sr(enhanced, SR)

            if _HAS_STOI:
                stoi_v, stoi_err = _safe_metric(
                    lambda: stoi_fn(clean16, enh16, _METRIC_SR, extended=False)
                )
                if stoi_err:
                    err_msgs.append(f"stoi:{stoi_err}")
            else:
                err_msgs.append("stoi:package_not_available")

            if _HAS_PESQ:
                pesq_v, pesq_err = _safe_metric(lambda: pesq_fn(_METRIC_SR, clean16, enh16, "wb"))
                if pesq_err:
                    err_msgs.append(f"pesq:{pesq_err}")
            else:
                err_msgs.append("pesq:package_not_available")

            if args.save_wavs:
                out_wav = os.path.join(wav_dir, f"enhanced_{name}")
                import soundfile as sf

                sf.write(out_wav, enhanced, SR, subtype="PCM_16")
        except Exception as e:
            status = "error"
            err_msgs.append(str(e))
            n = 0

        row = EvalRow(
            filename=name,
            noisy_path=noisy_path,
            clean_path=clean_path,
            num_samples=int(n),
            duration_s=float(n / SR) if n else 0.0,
            snr_db=float(snr_db),
            si_sdr_db=float(si_sdr_db),
            si_snr_db=float(si_snr_db),
            stoi=float(stoi_v),
            pesq=float(pesq_v),
            status=status,
            error="; ".join(err_msgs),
        )
        rows.append(row)

        print(
            f"[{i}/{len(pairs)}] {name}  status={status}  "
            f"snr={row.snr_db:.3f} si_sdr={row.si_sdr_db:.3f} "
            f"si_snr={row.si_snr_db:.3f} stoi={row.stoi:.3f} pesq={row.pesq:.3f}"
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    payload = {
        "config": {
            "noisy_root": noisy_root,
            "clean_root": clean_root,
            "weights": weights,
            "out_dir": out_dir,
            "device": str(device),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "log_eps": args.log_eps,
            "max_files": args.max_files,
            "metric_sr": _METRIC_SR,
            "torchmetrics_enabled": _HAS_TM,
            "stoi_enabled": _HAS_STOI,
            "pesq_enabled": _HAS_PESQ,
        },
        "summary": _summary(rows),
        "artifacts": {"metrics_csv": "metrics_eval.csv", "summary_json": "eval_summary.json"},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[done] csv={csv_path}")
    print(f"[done] summary={summary_path}")
    print(f"[done] ok={payload['summary']['ok_files']} / total={payload['summary']['total_files']}")


if __name__ == "__main__":
    main()
