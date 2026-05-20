"""Resume training from a saved checkpoint (last.pt / custom .pt)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from core.model import FREQ_BINS, GRUChunkDenoiser, model_info
from train import HYPERPARAMS
from training.artifacts import resolve_run_dir, write_model_info_txt
from training.data import collate_padded_utterances, prepare_train_val_datasets
from training.loop import run_training_loop

_BASE = os.path.dirname(os.path.abspath(__file__))
_RUNS_ROOT = os.path.join(_BASE, "runs")
_BASELINE_MODEL_DIR = os.path.join(_BASE, "baseline_model")
_DEFAULT_BASELINE_CKPT = os.path.join(_BASELINE_MODEL_DIR, "last.pt")


def _load_run_config(checkpoint_path: str) -> dict[str, Any]:
    run_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume training from checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint (.pt). Defaults to model_development/baseline_model/last.pt.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Additional epochs to run.")
    parser.add_argument("--run-tag", default=None, help="New run tag (default: <old_tag>_resume).")
    parser.add_argument("--out-dir", default=None, help="Optional explicit output dir.")
    parser.add_argument("--device", default=None, help="cuda/cpu override.")
    parser.add_argument("--w-log-mag", type=float, default=None)
    parser.add_argument("--w-linear-mag", type=float, default=None)
    parser.add_argument("--w-mrstft", type=float, default=None)
    parser.add_argument(
        "--mrstft-resolutions",
        type=int,
        nargs="+",
        default=None,
        help="Flattened triples: n_fft win hop ... (example: 240 240 60 480 480 120).",
    )
    # Backward-compatible aliases
    parser.add_argument("--mse-weight", type=float, default=None)
    parser.add_argument("--mss-weight", type=float, default=None)
    parser.add_argument("--mss-fft-sizes", type=int, nargs="+", default=None, help="Deprecated alias.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Ignore optimizer state from checkpoint.")
    args = parser.parse_args()

    checkpoint = args.checkpoint or _DEFAULT_BASELINE_CKPT
    if not checkpoint:
        print("Checkpoint not found. Pass --checkpoint or create baseline_model/last.pt first.", file=sys.stderr)
        sys.exit(1)
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(checkpoint):
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    hp = dict(HYPERPARAMS)
    old_cfg = _load_run_config(checkpoint)
    for k in hp:
        if k in old_cfg:
            hp[k] = old_cfg[k]

    if args.device is not None:
        hp["device"] = args.device
    if args.w_log_mag is not None:
        hp["w_log_mag"] = args.w_log_mag
    if args.w_linear_mag is not None:
        hp["w_linear_mag"] = args.w_linear_mag
    if args.w_mrstft is not None:
        hp["w_mrstft"] = args.w_mrstft
    if args.mrstft_resolutions is not None:
        if len(args.mrstft_resolutions) % 3 != 0:
            print("--mrstft-resolutions must contain triples: n_fft win hop ...", file=sys.stderr)
            sys.exit(1)
        hp["mrstft_resolutions"] = [
            args.mrstft_resolutions[i : i + 3] for i in range(0, len(args.mrstft_resolutions), 3)
        ]
    if args.mse_weight is not None:
        hp["w_linear_mag"] = args.mse_weight
    if args.mss_weight is not None:
        hp["w_mrstft"] = args.mss_weight
    if args.mss_fft_sizes is not None:
        hp["mrstft_resolutions"] = [[v, v, max(1, v // 4)] for v in args.mss_fft_sizes]
    hp["epochs"] = int(max(1, args.epochs))

    prev_tag = str(old_cfg.get("run_tag") or "resume")
    hp["run_tag"] = args.run_tag or f"{prev_tag}_resume"

    random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])

    out_dir = resolve_run_dir(_RUNS_ROOT, args.out_dir, hp["run_tag"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"[resume] checkpoint={checkpoint}")
    print(f"[resume] output_dir={out_dir}")

    train_ds, val_ds, info = prepare_train_val_datasets(
        hp["noisy_root"],
        hp["clean_root"],
        hp["val_fraction"],
        hp["log_eps"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=hp["workers"],
        collate_fn=collate_padded_utterances,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=hp["workers"],
        collate_fn=collate_padded_utterances,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if len(train_loader) == 0:
        print("Train DataLoader is empty.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(hp["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GRUChunkDenoiser(
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["gru_num_layers"],
        dropout=hp["gru_dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        ckpt_epoch = int(ckpt.get("epoch", 0))
        if not args.reset_optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
    else:
        model.load_state_dict(ckpt)
        ckpt_epoch = 0

    cfg = dict(hp)
    cfg["out_dir_resolved"] = out_dir
    cfg["resume_from_checkpoint"] = checkpoint
    cfg["resume_from_epoch"] = ckpt_epoch
    cfg.update(
        {
            "freq_bins": FREQ_BINS,
            "model": model_info(model)["name"],
            "batches_per_train_epoch": len(train_loader),
            **info,
        }
    )
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    write_model_info_txt(os.path.join(out_dir, "model_info.txt"), model)

    print(
        f"[resume] from_epoch={ckpt_epoch}  add_epochs={hp['epochs']}  "
        f"loss=log*{hp.get('w_log_mag', 1.0)} + lin*{hp.get('w_linear_mag', 0.0)} + mr*{hp.get('w_mrstft', 0.0)}"
    )
    run_training_loop(
        hp,
        out_dir,
        model,
        optimizer,
        train_loader,
        val_loader,
        len(val_ds),
        device,
    )
    print(f"[done] saved under {out_dir}")


if __name__ == "__main__":
    main()

