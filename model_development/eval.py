"""Batch evaluation on test noisy/clean pairs with objective metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Callable

import librosa
import numpy as np
import torch

from core.audio import N_FFT, SR, load_audio
from eval_one import enhance_waveform, load_weights
from core.model import GRUChunkDenoiser
from training.data import scan_wavs_by_basename

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
        "20260519_231544_gru_h128_L3_bs16_lr1e-05",
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
    noisy_snr_db: float
    noisy_si_sdr_db: float
    noisy_si_snr_db: float
    snr_db: float
    si_sdr_db: float
    si_snr_db: float
    snr_gain_db: float
    si_sdr_gain_db: float
    si_snr_gain_db: float
    stoi: float
    pesq: float
    best_lag_samples: int
    peak_ratio: float
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


def _slice_for_lag(ref: np.ndarray, sig: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        ref_s = ref[lag:]
        sig_s = sig[: ref_s.shape[0]]
    else:
        ref_s = ref[: ref.shape[0] + lag]
        sig_s = sig[-lag:]
    n = min(ref_s.shape[0], sig_s.shape[0])
    if n <= 0:
        return ref[:0], sig[:0]
    return ref_s[:n], sig_s[:n]


def _best_lag_by_snr(clean: np.ndarray, est: np.ndarray, max_lag: int) -> int:
    if max_lag <= 0:
        return 0
    best_lag = 0
    best_score = -float("inf")
    for lag in range(-max_lag, max_lag + 1):
        c, e = _slice_for_lag(clean, est, lag)
        if c.size == 0:
            continue
        score = _snr_db(c, e)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


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
    wav_out, _ = enhance_waveform(
        noisy,
        model,
        device,
        log_eps,
        preserve_input_tail=False,
        pad_end_for_chunking=True,
        ola_min_weight=0.0,
        boundary_pad_samples=240,
    )
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
            "noisy_snr_db": stats("noisy_snr_db"),
            "noisy_si_sdr_db": stats("noisy_si_sdr_db"),
            "noisy_si_snr_db": stats("noisy_si_snr_db"),
            "snr_db": stats("snr_db"),
            "si_sdr_db": stats("si_sdr_db"),
            "si_snr_db": stats("si_snr_db"),
            "snr_gain_db": stats("snr_gain_db"),
            "si_sdr_gain_db": stats("si_sdr_gain_db"),
            "si_snr_gain_db": stats("si_snr_gain_db"),
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
        "--gate-min-snr-db",
        type=float,
        default=None,
        help="Optional fail threshold on summary mean SNR dB.",
    )
    parser.add_argument(
        "--gate-min-si-sdr-db",
        type=float,
        default=None,
        help="Optional fail threshold on summary mean SI-SDR dB.",
    )
    parser.add_argument(
        "--gate-min-stoi",
        type=float,
        default=None,
        help="Optional fail threshold on summary mean STOI (requires pystoi).",
    )
    parser.add_argument(
        "--gate-min-pesq",
        type=float,
        default=None,
        help="Optional fail threshold on summary mean PESQ (requires pesq).",
    )
    parser.add_argument(
        "--save-wavs",
        action="store_true",
        help="Save enhanced wavs under eval_outputs/enhanced_wavs.",
    )
    parser.add_argument(
        "--lag-search-max-samples",
        type=int,
        default=0,
        help="Optional sample lag search (+/-N) for SNR/SI metrics alignment.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    noisy_root = os.path.abspath(args.noisy_root)
    clean_root = os.path.abspath(args.clean_root)
    weights = os.path.abspath(args.weights)
    base_out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(noisy_root):
        raise FileNotFoundError(f"Noisy root not found: {noisy_root}")
    if not os.path.isdir(clean_root):
        raise FileNotFoundError(f"Clean root not found: {clean_root}")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")

    model_tag = os.path.basename(os.path.dirname(weights)) or os.path.splitext(os.path.basename(weights))[0]
    out_dir = os.path.join(base_out_dir, model_tag)
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
        (
            noisy_snr_db,
            noisy_si_sdr_db,
            noisy_si_snr_db,
            snr_db,
            si_sdr_db,
            si_snr_db,
            snr_gain_db,
            si_sdr_gain_db,
            si_snr_gain_db,
            stoi_v,
            pesq_v,
            peak_ratio,
        ) = (float("nan"),) * 12
        best_lag_samples = 0
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
            noisy = noisy[:n]
            enhanced = enhanced[:n]

            best_lag_samples = _best_lag_by_snr(clean, enhanced, max(0, int(args.lag_search_max_samples)))
            clean_eval, noisy_eval = _slice_for_lag(clean, noisy, best_lag_samples)
            _, enhanced_eval = _slice_for_lag(clean, enhanced, best_lag_samples)
            if clean_eval.size == 0:
                raise RuntimeError("empty_after_alignment")

            noisy_snr_db = _snr_db(clean_eval, noisy_eval)
            noisy_si_sdr_db, noisy_si_snr_db = _si_metrics(clean_eval, noisy_eval)
            snr_db = _snr_db(clean_eval, enhanced_eval)
            si_sdr_db, si_snr_db = _si_metrics(clean_eval, enhanced_eval)
            snr_gain_db = snr_db - noisy_snr_db
            si_sdr_gain_db = si_sdr_db - noisy_si_sdr_db
            si_snr_gain_db = si_snr_db - noisy_si_snr_db
            peak_ratio = float(
                np.max(np.abs(enhanced_eval)) / (np.max(np.abs(noisy_eval)) + 1e-12)
            )

            clean16 = _to_metric_sr(clean_eval, SR)
            enh16 = _to_metric_sr(enhanced_eval, SR)

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
            noisy_snr_db=float(noisy_snr_db),
            noisy_si_sdr_db=float(noisy_si_sdr_db),
            noisy_si_snr_db=float(noisy_si_snr_db),
            snr_db=float(snr_db),
            si_sdr_db=float(si_sdr_db),
            si_snr_db=float(si_snr_db),
            snr_gain_db=float(snr_gain_db),
            si_sdr_gain_db=float(si_sdr_gain_db),
            si_snr_gain_db=float(si_snr_gain_db),
            stoi=float(stoi_v),
            pesq=float(pesq_v),
            best_lag_samples=int(best_lag_samples),
            peak_ratio=float(peak_ratio),
            status=status,
            error="; ".join(err_msgs),
        )
        rows.append(row)

        print(
            f"[{i}/{len(pairs)}] {name}  status={status}  "
            f"snr={row.snr_db:.3f} (noisy={row.noisy_snr_db:.3f}, gain={row.snr_gain_db:.3f}) "
            f"si_sdr={row.si_sdr_db:.3f} (gain={row.si_sdr_gain_db:.3f}) "
            f"si_snr={row.si_snr_db:.3f} (gain={row.si_snr_gain_db:.3f}) "
            f"lag={row.best_lag_samples} peak_ratio={row.peak_ratio:.3f} "
            f"stoi={row.stoi:.3f} pesq={row.pesq:.3f}"
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
            "lag_search_max_samples": int(args.lag_search_max_samples),
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

    gate_failed = False
    metrics = payload["summary"]["metrics"]

    def _mean_of(key: str) -> float | None:
        return metrics[key]["mean"]

    checks: list[tuple[str, float | None, float | None]] = [
        ("snr_db", _mean_of("snr_db"), args.gate_min_snr_db),
        ("si_sdr_db", _mean_of("si_sdr_db"), args.gate_min_si_sdr_db),
        ("stoi", _mean_of("stoi"), args.gate_min_stoi),
        ("pesq", _mean_of("pesq"), args.gate_min_pesq),
    ]
    for name, mean_v, min_v in checks:
        if min_v is None:
            continue
        if mean_v is None or not math.isfinite(float(mean_v)):
            gate_failed = True
            print(f"[gate-fail] {name} mean is unavailable (threshold={min_v})", file=sys.stderr)
            continue
        if float(mean_v) < float(min_v):
            gate_failed = True
            print(f"[gate-fail] {name} mean={float(mean_v):.6f} < threshold={float(min_v):.6f}", file=sys.stderr)
    if gate_failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
