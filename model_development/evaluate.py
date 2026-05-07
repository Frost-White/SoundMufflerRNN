import json
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from core import TinyDenoiser, WavPairDataset, pair_wav_files, snr_db

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NOISY_ROOT = "../data/test/noisy_testset_wav"
CLEAN_ROOT = "../data/test/clean_testset_wav"
CHECKPOINT = "runs/"
SR = 48000
CHUNK = 960
HOP = 720


def rel_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def required_config_int(config: dict, key: str) -> int:
    if key not in config:
        raise KeyError(f"Checkpoint config is missing required key: {key}")
    return int(config[key])


def required_config_value(config: dict, key: str):
    if key not in config:
        raise KeyError(f"Checkpoint config is missing required key: {key}")
    return config[key]


def collate_chunk_batches(batch):
    noisy_batch, clean_batch = zip(*batch)
    return torch.cat(noisy_batch, dim=0), torch.cat(clean_batch, dim=0)


def main():
    noisy_root = rel_path(NOISY_ROOT)
    clean_root = rel_path(CLEAN_ROOT)
    checkpoint = rel_path(CHECKPOINT)
    if os.path.isdir(checkpoint):
        raise ValueError("CHECKPOINT must point to a .pt file, not a directory.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device)
    config = ckpt.get("config", {})
    hidden_size = required_config_int(config, "hidden_size")
    target_sr = required_config_int(config, "sr")
    chunk_size = required_config_int(config, "chunk_samples")
    hop_size = required_config_int(config, "hop_samples")
    batch_size = int(config.get("batch_size", 1))
    required_config_value(config, "model_version")
    required_config_value(config, "preprocess_version")

    pairs = pair_wav_files(noisy_root, clean_root)
    loader = DataLoader(
        WavPairDataset(pairs, chunk_size=chunk_size, hop=hop_size, target_sr=target_sr),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_chunk_batches,
    )

    model = TinyDenoiser(hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_snr_gain = 0.0
    total = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.reshape(-1, noisy.size(-2), noisy.size(-1)).to(device)
            clean = clean.reshape(-1, clean.size(-2), clean.size(-1)).to(device)
            pred = model(noisy)
            total_loss += criterion(pred, clean).item()
            total_snr_gain += (snr_db(clean, pred).mean() - snr_db(clean, noisy).mean()).item()
            total += 1

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
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "test_loss": test_loss,
        "test_snr_gain_db": test_snr_gain,
        "num_samples": total,
    }
    with open(os.path.join(report_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
