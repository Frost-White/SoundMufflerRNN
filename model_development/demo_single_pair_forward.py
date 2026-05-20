"""One noisy wav: STFT chunks -> GRUChunkDenoiser -> overlap-add wav."""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch

from core.audio import (
    SR,
    load_audio,
)
from core.model import FREQ_BINS, GRUChunkDenoiser
from eval_one import enhance_waveform
from training.data import scan_wavs_by_basename

_LOG_EPS = 1e-8

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NOISY = os.path.normpath(
    os.path.join(_BASE, "..", "data", "train", "noisy_trainset_56spk_wav")
)
_DEFAULT_CLEAN = os.path.normpath(
    os.path.join(_BASE, "..", "data", "train", "clean_trainset_56spk_wav")
)


def pick_random_noisy_path(noisy_root: str) -> str:
    m = scan_wavs_by_basename(noisy_root)
    if not m:
        raise FileNotFoundError(f"No .wav under: {noisy_root}")
    return random.choice(list(m.values()))


def find_clean_twin(clean_root: str, basename: str) -> str | None:
    m = scan_wavs_by_basename(clean_root)
    return m.get(basename)


def main() -> None:
    p = argparse.ArgumentParser(description="One wav STFT + GRUChunkDenoiser forward.")
    p.add_argument("--noisy-root", default=_DEFAULT_NOISY)
    p.add_argument("--clean-root", default=_DEFAULT_CLEAN)
    p.add_argument("--noisy-file", default=None, help="Exact noisy wav path (skip random pick).")
    p.add_argument("--weights", default=None, help="Optional .pt state_dict from training.")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--gru-layers", type=int, default=1)
    p.add_argument(
        "--identity-mask",
        action="store_true",
        help="Bypass model and use an all-ones mask for reconstruction sanity checks.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--out",
        default=None,
        help="Output wav (default: model_development/demo_reconstructed_<stem>.wav).",
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
        try:
            noisy_path = pick_random_noisy_path(args.noisy_root)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    base = os.path.basename(noisy_path)
    clean_path = find_clean_twin(args.clean_root, base)
    if clean_path is None:
        print(f"No clean twin for {base!r} under {args.clean_root}", file=sys.stderr)
        sys.exit(1)

    print(f"noisy:  {noisy_path}")
    print(f"clean:  {clean_path}")

    noisy_wav, _ = load_audio(noisy_path)
    if len(noisy_wav) == 0:
        print("No STFT chunks (audio shorter than one chunk).", file=sys.stderr)
        sys.exit(1)

    model: GRUChunkDenoiser | None = None
    if not args.identity_mask:
        model = GRUChunkDenoiser(hidden_dim=args.hidden_dim, num_layers=args.gru_layers)
    if args.weights and model is not None:
        state = torch.load(args.weights, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
    if model is not None:
        model.eval()
    wav_out, mask_np = enhance_waveform(
        noisy_wav,
        model,
        torch.device("cpu"),
        _LOG_EPS,
        identity_mask=args.identity_mask,
    )
    if mask_np.size == 0:
        print("No STFT chunks (audio shorter than one chunk).", file=sys.stderr)
        sys.exit(1)
    print(f"chunks={mask_np.shape[0]}  stft=({mask_np.shape[0]}, {FREQ_BINS})  freq_bins={FREQ_BINS}")

    stem = os.path.splitext(base)[0]
    out_path = (
        os.path.abspath(args.out)
        if args.out
        else os.path.join(_BASE, f"demo_reconstructed_{stem}.wav")
    )
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(out_path, wav_out, SR, subtype="PCM_16")

    print(f"model in  (1, {mask_np.shape[0]}, {FREQ_BINS})  ->  out (1, {mask_np.shape[0]}, {FREQ_BINS})")
    print(
        f"mask min/mean/max: {float(mask_np.min()):.4f} / "
        f"{float(mask_np.mean()):.4f} / {float(mask_np.max()):.4f}"
    )
    print(f"saved: {out_path}  samples={len(wav_out)}  sr={SR}")


if __name__ == "__main__":
    main()
