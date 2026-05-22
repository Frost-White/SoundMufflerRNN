"""Run model on the lowest-PESQ file from metrics_eval.csv and save outputs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import soundfile as sf
import torch

from core.audio import SR, load_audio
from eval import _enhance_waveform
from eval_one import load_weights
from core.model import GRUChunkDenoiser

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_METRICS_CSV = os.path.join(_BASE, "eval_outputs", "20260521_152024_gru_h128_L3_bs16_lr1e-05_resume", "metrics_eval.csv")
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


def _find_worst_pesq_row(csv_path: str) -> dict[str, str]:
    best: tuple[float, dict[str, str]] | None = None
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            try:
                pesq = float(row["pesq"])
            except (KeyError, ValueError):
                continue
            if not math.isfinite(pesq):
                continue
            if best is None or pesq < best[0]:
                best = (pesq, row)
    if best is None:
        raise RuntimeError(f"No valid PESQ rows in {csv_path}")
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance the lowest-PESQ utterance from metrics_eval.csv."
    )
    parser.add_argument("--metrics-csv", default=_DEFAULT_METRICS_CSV)
    parser.add_argument("--weights", default=_DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--log-eps", type=float, default=_LOG_EPS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--also-save-noisy-clean",
        action="store_true",
        help="Also write noisy/clean copies next to enhanced for A/B listening.",
    )
    args = parser.parse_args()

    metrics_csv = os.path.abspath(args.metrics_csv)
    weights = os.path.abspath(args.weights)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isfile(metrics_csv):
        print(f"Metrics CSV not found: {metrics_csv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(weights):
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)

    row = _find_worst_pesq_row(metrics_csv)
    pesq = float(row["pesq"])
    filename = row["filename"]
    noisy_path = row["noisy_path"]
    clean_path = row.get("clean_path", "")

    if not os.path.isfile(noisy_path):
        print(f"Noisy file not found: {noisy_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    model = GRUChunkDenoiser(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    load_weights(model, weights, device)
    model.eval()

    noisy_wav, _ = load_audio(noisy_path)
    enhanced = _enhance_waveform(noisy_wav, model, device, args.log_eps)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    enhanced_path = os.path.join(out_dir, f"worst_pesq_enhanced_{stem}.wav")
    sf.write(enhanced_path, enhanced, SR, subtype="PCM_16")

    print(f"worst_pesq: {pesq:.4f}")
    print(f"filename:   {filename}")
    print(f"noisy:      {noisy_path}")
    if clean_path:
        print(f"clean:      {clean_path}")
    print(f"saved:      {enhanced_path}")
    print(f"samples={len(enhanced)} sr={SR}")

    if args.also_save_noisy_clean:
        noisy_out = os.path.join(out_dir, f"worst_pesq_noisy_{stem}.wav")
        sf.write(noisy_out, noisy_wav, SR, subtype="PCM_16")
        print(f"saved:      {noisy_out}")
        if clean_path and os.path.isfile(clean_path):
            clean_wav, _ = load_audio(clean_path)
            n = min(len(clean_wav), len(noisy_wav))
            clean_out = os.path.join(out_dir, f"worst_pesq_clean_{stem}.wav")
            sf.write(clean_out, clean_wav[:n], SR, subtype="PCM_16")
            print(f"saved:      {clean_out}")


if __name__ == "__main__":
    main()
