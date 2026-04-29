import argparse
import copy
import json
import os
import random
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from core import ChunkPairDataset, TinyDenoiser, pair_chunk_files, snr_db

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def rel_path(path: str) -> str:
    # Resolve relative paths from this script directory.
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def run_eval(model, loader, criterion, device):
    # Run validation and return loss plus SNR gain.
    model.eval()
    total_loss = 0.0
    total_snr_gain = 0.0
    total = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            noisy_snr = snr_db(clean, noisy).mean()
            pred_snr = snr_db(clean, pred).mean()
            total_loss += loss.item() * noisy.size(0)
            total_snr_gain += (pred_snr - noisy_snr).item() * noisy.size(0)
            total += noisy.size(0)
    return total_loss / total, total_snr_gain / total


def count_parameters(model):
    # Count trainable model parameters.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(path, payload):
    # Save dict/list payload as pretty JSON.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_config(path):
    # Load train config JSON from disk.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_training_plots(metrics_history, output_dir):
    # Save train/val loss and val SNR gain plots.
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
    # Train for N epochs and save one hyperparameter-tagged checkpoint.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.json")
    args0, _ = parser.parse_known_args()
    config_path = rel_path(args0.config)
    config = load_config(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.json")
    parser.add_argument("--noisy-root", default=config.get("noisy_root", "../data/chunk_train/noisy_trainset_56spk_wav"))
    parser.add_argument("--clean-root", default=config.get("clean_root", "../data/chunk_train/clean_trainset_56spk_wav"))
    parser.add_argument("--epochs", type=int, default=int(config.get("epochs", 10)))
    parser.add_argument("--batch-size", type=int, default=int(config.get("batch_size", 32)))
    parser.add_argument("--hidden-size", type=int, default=int(config.get("hidden_size", 8)))
    parser.add_argument("--lr", type=float, default=float(config.get("lr", 1e-3)))
    parser.add_argument("--val-ratio", type=float, default=float(config.get("val_ratio", 0.1)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 42)))
    parser.add_argument("--out-dir", default=config.get("out_dir", "runs"))
    args = parser.parse_args()

    noisy_root = rel_path(args.noisy_root)
    clean_root = rel_path(args.clean_root)
    out_dir = rel_path(args.out_dir)
    run_tag = f"bs{args.batch_size}_hidden{args.hidden_size}"
    report_dir = os.path.join(out_dir, run_tag)
    checkpoint_path = os.path.join(report_dir, f"{run_tag}.pt")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = pair_chunk_files(noisy_root, clean_root)
    dataset = ChunkPairDataset(pairs)
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    split = int(len(idxs) * (1.0 - args.val_ratio))
    train_ds = Subset(dataset, idxs[:split])
    val_ds = Subset(dataset, idxs[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyDenoiser(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "train.py",
        "model_name": model.__class__.__name__,
        "device": device,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.lr,
        "val_ratio": args.val_ratio,
        "noisy_root": noisy_root,
        "clean_root": clean_root,
        "out_dir": out_dir,
        "run_tag": run_tag,
        "report_dir": report_dir,
        "checkpoint_path": checkpoint_path,
        "total_pairs": len(dataset),
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
        f.write("epoch,train_loss,val_loss,val_snr_gain_db\n")

    metrics_history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_payload = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total = 0
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item() * noisy.size(0)
            total += noisy.size(0)

        train_loss = total_train_loss / total
        val_loss, val_snr_gain = run_eval(model, val_loader, criterion, device)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_snr_gain={val_snr_gain:.3f}dB"
        )

        metrics_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_snr_gain_db": val_snr_gain,
        }
        metrics_history.append(metrics_row)
        with open(metrics_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{val_snr_gain:.8f}\n")

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
            "total_epochs": args.epochs,
            "checkpoint_path": checkpoint_path,
        },
    )
    save_training_plots(metrics_history, report_dir)


if __name__ == "__main__":
    main()
