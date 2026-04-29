import argparse
import json
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from core import ChunkPairDataset, TinyDenoiser, pair_chunk_files, snr_db

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def rel_path(path: str) -> str:
    # Resolve relative paths from this script directory.
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def main():
    # Load checkpoint and report test loss/SNR gain.
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy-root", default="../data/chunk_test/noisy_testset_wav")
    parser.add_argument("--clean-root", default="../data/chunk_test/clean_testset_wav")
    parser.add_argument(
        "--checkpoint",
        default="runs/bs32_hidden8/bs32_hidden8.pt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    noisy_root = rel_path(args.noisy_root)
    clean_root = rel_path(args.clean_root)
    checkpoint = rel_path(args.checkpoint)

    pairs = pair_chunk_files(noisy_root, clean_root)
    loader = DataLoader(ChunkPairDataset(pairs), batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device)
    config = ckpt.get("config", {})
    hidden_size = int(config.get("hidden_size", 8))
    model = TinyDenoiser(hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_snr_gain = 0.0
    total = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            total_loss += criterion(pred, clean).item() * noisy.size(0)
            total_snr_gain += (snr_db(clean, pred).mean() - snr_db(clean, noisy).mean()).item() * noisy.size(0)
            total += noisy.size(0)

    test_loss = total_loss / total
    test_snr_gain = total_snr_gain / total
    print(f"test_loss={test_loss:.6f}")
    print(f"test_snr_gain={test_snr_gain:.3f}dB")

    report_dir = os.path.dirname(checkpoint)
    os.makedirs(report_dir, exist_ok=True)
    test_metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "evaluate.py",
        "checkpoint": checkpoint,
        "noisy_root": noisy_root,
        "clean_root": clean_root,
        "batch_size": args.batch_size,
        "hidden_size": hidden_size,
        "test_loss": test_loss,
        "test_snr_gain_db": test_snr_gain,
        "num_samples": total,
    }
    with open(os.path.join(report_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
