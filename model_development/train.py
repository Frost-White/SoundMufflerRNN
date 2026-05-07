import copy
import json
import os
import random
import sys
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from core import TinyDenoiser, WavPairDataset, pair_wav_files, snr_db

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NOISY_ROOT = "../data/train/noisy_trainset_56spk_wav"
CLEAN_ROOT = "../data/train/clean_trainset_56spk_wav"
SR = 48000
CHUNK = 960  # 20 ms
HOP = 720  # 15 ms hop, 5 ms overlap
EPOCHS = 10
HIDDEN_SIZE = 16
LR = 1e-3
BATCH_SIZE = 16
VAL_RATIO = 0.1
SEED = 42
OUT_DIR = "runs"
MODEL_VERSION = 1
PREPROCESS_VERSION = 1


def enable_unbuffered_output():
    # Match python -u behavior when running this script directly.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)


def rel_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_snr_gain = 0.0
    total = 0
    with torch.no_grad():
        for step, (noisy, clean) in enumerate(loader, start=1):
            noisy = noisy.reshape(-1, noisy.size(-2), noisy.size(-1)).to(device)
            clean = clean.reshape(-1, clean.size(-2), clean.size(-1)).to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            noisy_snr = snr_db(clean, noisy).mean()
            pred_snr = snr_db(clean, pred).mean()
            total_loss += loss.item()
            total_snr_gain += (pred_snr - noisy_snr).item()
            total += 1
    return total_loss / total, total_snr_gain / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def collate_chunk_batches(batch):
    noisy_batch, clean_batch = zip(*batch)
    return torch.cat(noisy_batch, dim=0), torch.cat(clean_batch, dim=0)


def save_training_plots(metrics_history, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        with open(os.path.join(output_dir, "plot_warning.txt"), "w", encoding="utf-8") as f:
            f.write("matplotlib is not installed; plots were not generated.\n")
        return

    epochs = [m["epoch"] for m in metrics_history]
    train_losses = [m["train_loss"] for m in metrics_history]
    val_losses = [m["val_loss"] for m in metrics_history]
    val_snr_gains = [m["val_snr_gain_db"] for m in metrics_history]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, val_snr_gains, label="val_snr_gain_db")
    plt.xlabel("Epoch")
    plt.ylabel("SNR Gain (dB)")
    plt.title("Validation SNR Gain")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "snr_curve.png"), dpi=150)
    plt.close()


def main():
    print("Preparing dataset...", flush=True)
    noisy_root = rel_path(NOISY_ROOT)
    clean_root = rel_path(CLEAN_ROOT)
    out_dir = rel_path(OUT_DIR)
    run_tag = f"bs{BATCH_SIZE}_hidden{HIDDEN_SIZE}"
    report_dir = os.path.join(out_dir, run_tag)
    checkpoint_path = os.path.join(report_dir, f"{run_tag}.pt")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    pairs = pair_wav_files(noisy_root, clean_root)
    print(f"Found {len(pairs)} paired wav files", flush=True)
    idxs = list(range(len(pairs)))
    random.shuffle(idxs)
    split = int(len(idxs) * (1.0 - VAL_RATIO))
    train_pairs = [pairs[i] for i in idxs[:split]]
    val_pairs = [pairs[i] for i in idxs[split:]]

    train_ds = WavPairDataset(train_pairs, chunk_size=CHUNK, hop=HOP, target_sr=SR)
    val_ds = WavPairDataset(val_pairs, chunk_size=CHUNK, hop=HOP, target_sr=SR)
    print(f"Dataset ready: train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_chunk_batches,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_chunk_batches,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyDenoiser(hidden_size=HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    print(f"Training started on {device} with batch_size={BATCH_SIZE}", flush=True)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "train.py",
        "model_name": model.__class__.__name__,
        "model_version": MODEL_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "device": device,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "learning_rate": LR,
        "val_ratio": VAL_RATIO,
        "noisy_root": noisy_root,
        "clean_root": clean_root,
        "out_dir": out_dir,
        "run_tag": run_tag,
        "report_dir": report_dir,
        "checkpoint_path": checkpoint_path,
        "sr": SR,
        "chunk_samples": CHUNK,
        "hop_samples": HOP,
        "total_pairs": len(pairs),
        "train_pairs": len(train_ds),
        "val_pairs": len(val_ds),
        "trainable_parameters": count_parameters(model),
        "optimizer": "Adam",
        "loss": "MSELoss",
    }
    save_json(os.path.join(report_dir, "run_config.json"), run_config)

    with open(os.path.join(report_dir, "model_info.txt"), "w", encoding="utf-8") as f:
        f.write("Model Run Info\n")
        f.write(f"created_at: {run_config['created_at']}\n")
        f.write(f"model_name: {run_config['model_name']}\n")
        f.write(f"device: {run_config['device']}\n")
        f.write(f"batch_size: {run_config['batch_size']}\n")
        f.write(f"hidden_size: {run_config['hidden_size']}\n")
        f.write(f"learning_rate: {run_config['learning_rate']}\n")
        f.write(f"epochs: {run_config['epochs']}\n")
        f.write(f"trainable_parameters: {run_config['trainable_parameters']}\n")
        f.write(f"train_pairs: {run_config['train_pairs']}\n")
        f.write(f"val_pairs: {run_config['val_pairs']}\n")
        f.write(f"noisy_root: {run_config['noisy_root']}\n")
        f.write(f"clean_root: {run_config['clean_root']}\n")

    metrics_csv_path = os.path.join(report_dir, "metrics_train.csv")
    with open(metrics_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_snr_gain_db,epoch_time_s\n")

    metrics_history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_payload = None
    for epoch in range(1, EPOCHS + 1):
        print(f"epoch={epoch:02d} started", flush=True)
        epoch_start = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        total_steps = 0
        for step, (noisy, clean) in enumerate(train_loader, start=1):
            noisy = noisy.reshape(-1, noisy.size(-2), noisy.size(-1)).to(device)
            clean = clean.reshape(-1, clean.size(-2), clean.size(-1)).to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
            total_steps += 1

        train_loss = total_train_loss / total_steps
        val_loss, val_snr_gain = run_eval(model, val_loader, criterion, device)
        epoch_time_s = time.perf_counter() - epoch_start
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_snr_gain={val_snr_gain:.3f}dB "
            f"epoch_time_s={epoch_time_s:.2f}",
            flush=True,
        )

        metrics_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_snr_gain_db": val_snr_gain,
            "epoch_time_s": epoch_time_s,
        }
        metrics_history.append(metrics_row)
        with open(metrics_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.8f},{val_loss:.8f},{val_snr_gain:.8f},{epoch_time_s:.4f}\n"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_checkpoint_payload = {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "epoch": epoch,
                "metrics": metrics_row,
                "config": run_config,
            }

    if best_checkpoint_payload is None:
        raise RuntimeError("Training produced no checkpoint payload.")
    torch.save(best_checkpoint_payload, checkpoint_path)

    save_json(os.path.join(report_dir, "metrics_train.json"), metrics_history)
    save_json(
        os.path.join(report_dir, "train_summary.json"),
        {
            "run_tag": run_tag,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": EPOCHS,
            "checkpoint_path": checkpoint_path,
        },
    )
    save_training_plots(metrics_history, report_dir)


if __name__ == "__main__":
    enable_unbuffered_output()
    main()
