"""Train GRUChunkDenoiser on paired wavs (utterance batches, padded STFT sequences)."""

from __future__ import annotations

import json
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

from core.model import FREQ_BINS, GRUChunkDenoiser, model_info
from training.artifacts import resolve_run_dir, write_model_info_txt
from training.data import collate_padded_utterances, prepare_train_val_datasets
from training.loop import run_training_loop

_BASE = os.path.dirname(os.path.abspath(__file__))
_RUNS_ROOT = os.path.join(_BASE, "runs")
_DEFAULT_NOISY = os.path.normpath(os.path.join(_BASE, "..", "data", "train", "noisy_trainset_56spk_wav"))
_DEFAULT_CLEAN = os.path.normpath(os.path.join(_BASE, "..", "data", "train", "clean_trainset_56spk_wav"))

HYPERPARAMS = {
    "noisy_root": _DEFAULT_NOISY,
    "clean_root": _DEFAULT_CLEAN,
    "out_dir": None,
    "run_tag": "",
    "val_fraction": 0.1,
    "hidden_dim": 128,
    "gru_num_layers": 3,
    "gru_dropout": 0.05,
    "epochs": 30,
    "batch_size": 16,
    "lr": 1e-5,
    "workers": 0,
    "seed": 0,
    "device": "cuda",
    "log_eps": 1e-8,
    "loss_log_eps": 1e-8,
    "w_log_mag": 1.0,
    "w_linear_mag": 0.05,
    "w_mrstft": 0.05,
    "mrstft_resolutions": [[240, 240, 60], [480, 480, 120], [960, 960, 240]],
}
HYPERPARAMS["run_tag"] = (
    f"gru_h{HYPERPARAMS['hidden_dim']}_L{HYPERPARAMS['gru_num_layers']}"
    f"_bs{HYPERPARAMS['batch_size']}_lr{HYPERPARAMS['lr']}"
)


def main() -> None:
    hp = HYPERPARAMS
    random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])

    out_dir = resolve_run_dir(_RUNS_ROOT, hp["out_dir"], hp["run_tag"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run] output_dir={out_dir}")

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

    cfg = dict(hp)
    cfg["out_dir_resolved"] = out_dir
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
    print(f"[run] device={device}  batches/epoch={len(train_loader)}")

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
