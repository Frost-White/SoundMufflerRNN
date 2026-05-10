"""Train TinyDenoiser on noisy/clean pairs (chunk-level batches, aligned STFT)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from audio_pipeline import CHUNK_HOP, CHUNK_SAMPLES, chunk_waveform, load_audio, stft_chunks
from model import FREQ_BINS, TinyDenoiser, model_info

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NOISY = os.path.normpath(os.path.join(_BASE, "..", "data", "train", "noisy_trainset_56spk_wav"))
_DEFAULT_CLEAN = os.path.normpath(os.path.join(_BASE, "..", "data", "train", "clean_trainset_56spk_wav"))

# --- hiperparametreler: önce burayı düzenle (CLI aynı anahtarlarla bunların üstüne yazabilir) ---
HYPERPARAMS = {
    # veri
    "noisy_root": _DEFAULT_NOISY,
    "clean_root": _DEFAULT_CLEAN,
    # None -> yeni klasör: runs/<YYYYMMDD_HHMMSS>_<run_tag> (eski run'lara dokunmaz)
    "out_dir": None,
    "run_tag": "",
    "val_fraction": 0.1,
    # model (TinyDenoiser)
    "hidden_dim": 32,
    # eğitim
    "epochs": 5,
    "batch_size": 512,
    "lr": 1e-3,
    "workers": 0,
    "seed": 0,
    "device": "cuda",  # None = cuda varsa cuda, yok cpu
    # özellik / kayıp
    "log_eps": 1e-8,
}

_LOG_EPS = HYPERPARAMS["log_eps"]
_RUNS_ROOT = os.path.join(_BASE, "runs")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_run_dir(out_dir: str | None, run_tag: str) -> str:
    """Yeni eğitim çıktısı; out_dir verilmezse runs altında benzersiz klasör."""
    if out_dir:
        return os.path.abspath(out_dir)
    tag = run_tag.strip().replace(" ", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{tag}" if tag else stamp
    return os.path.abspath(os.path.join(_RUNS_ROOT, name))


def write_model_info_txt(path: str, model: TinyDenoiser) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for k, v in model_info(model).items():
            f.write(f"{k}: {v}\n")


def write_train_summary(path: str, payload: dict) -> None:
    out = {**payload, "last_updated_at": _utc_now_iso()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _progress_tick(label: str, done: int, total: int, *, max_ticks: int = 80) -> None:
    """Tek satırda ilerleme; total küçükken her adımda günceller."""
    if total <= 0:
        return
    step = max(1, total // max_ticks)
    if done == total or done % step == 0 or total <= 30:
        pct = 100.0 * done / total
        sys.stdout.write(f"\r[{label}] {done}/{total} ({pct:.1f}%)")
        sys.stdout.flush()


def _progress_end() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


def find_wav_by_basename(root: str, basename: str) -> str | None:
    for cur, _, names in os.walk(root):
        if basename in names:
            return os.path.join(cur, basename)
    return None


def collect_pairs(noisy_root: str, clean_root: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Scan noisy tree; for each .wav resolve clean twin by filename (same basename)."""
    noisy_paths: list[str] = []
    for cur, _, names in os.walk(noisy_root):
        for name in sorted(names):
            if name.lower().endswith(".wav"):
                noisy_paths.append(os.path.join(cur, name))

    total = len(noisy_paths)
    print(f"[pairs] {total} gürültülü dosya bulundu; temiz çiftleri aranıyor...")
    pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    for i, noisy_path in enumerate(noisy_paths, start=1):
        name = os.path.basename(noisy_path)
        clean_path = find_wav_by_basename(clean_root, name)
        if clean_path is None:
            missing.append(noisy_path)
        else:
            pairs.append((noisy_path, clean_path))
        _progress_tick("pairs", i, total)
    _progress_end()
    return pairs, missing


def num_chunks_for_length(num_samples: int) -> int:
    """Match `chunk_waveform` stride/hop chunk count without loading audio."""
    if num_samples < CHUNK_SAMPLES:
        return 0
    n_win = num_samples - CHUNK_SAMPLES + 1
    return (n_win + CHUNK_HOP - 1) // CHUNK_HOP


def build_chunk_rows(
    pairs: list[tuple[str, str]], label: str = "chunk_idx"
) -> list[tuple[str, str, int]]:
    """
    One row = one aligned chunk index for a (noisy_path, clean_path) pair.
    Chunk index k is always the same temporal frame for noisy and clean after cropping.
    """
    total = len(pairs)
    if total:
        print(f"[{label}] {total} çift için chunk sayıları okunuyor (metadata)...")
    rows: list[tuple[str, str, int]] = []
    for i, (noisy_path, clean_path) in enumerate(pairs, start=1):
        try:
            ln = sf.info(noisy_path).frames
            lc = sf.info(clean_path).frames
        except Exception as e:
            print(f"[skip] metadata read failed: {noisy_path} ({e})", file=sys.stderr)
            _progress_tick(label, i, total)
            continue
        L = min(ln, lc)
        n = num_chunks_for_length(L)
        for k in range(n):
            rows.append((noisy_path, clean_path, k))
        _progress_tick(label, i, total)
    if total:
        _progress_end()
    return rows


class AlignedChunkDataset(Dataset):
    """
    Each sample is one chunk from an aligned pair; tensors built from the same chunk_idx.
    Rows must come from `build_chunk_rows` so paths + chunk_idx stay matched.
    """

    def __init__(self, rows: list[tuple[str, str, int]], log_eps: float = _LOG_EPS):
        self._rows = rows
        self._log_eps = log_eps

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        noisy_path, clean_path, chunk_idx = self._rows[idx]

        noisy_wav, _ = load_audio(noisy_path)
        clean_wav, _ = load_audio(clean_path)
        L = min(len(noisy_wav), len(clean_wav))
        noisy_wav = noisy_wav[:L].astype(np.float32, copy=False)
        clean_wav = clean_wav[:L].astype(np.float32, copy=False)

        chunks_n = chunk_waveform(noisy_wav)
        chunks_c = chunk_waveform(clean_wav)
        assert chunks_n.shape == chunks_c.shape, (noisy_path, chunks_n.shape, chunks_c.shape)

        cn = chunks_n[chunk_idx]
        cc = chunks_c[chunk_idx]
        spec_n = stft_chunks(cn.reshape(1, -1))
        spec_c = stft_chunks(cc.reshape(1, -1))
        mag_n = np.abs(spec_n[0]).astype(np.float32)
        mag_c = np.abs(spec_c[0]).astype(np.float32)
        x = np.log(mag_n + self._log_eps).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "mag_noisy": torch.from_numpy(mag_n),
            "mag_clean": torch.from_numpy(mag_c),
        }


def collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([s["x"] for s in samples], dim=0),
        "mag_noisy": torch.stack([s["mag_noisy"] for s in samples], dim=0),
        "mag_clean": torch.stack([s["mag_clean"] for s in samples], dim=0),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """One epoch = one full pass over `loader` (incomplete last batch dropped if drop_last)."""
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        mag_n = batch["mag_noisy"].to(device)
        mag_c = batch["mag_clean"].to(device)
        bs = x.shape[0]

        optimizer.zero_grad(set_to_none=True)
        mask = model(x)
        pred_mag = mask * mag_n
        loss = loss_fn(pred_mag, mag_c)
        loss.backward()
        optimizer.step()
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        mag_n = batch["mag_noisy"].to(device)
        mag_c = batch["mag_clean"].to(device)
        mask = model(x)
        pred_mag = mask * mag_n
        loss = loss_fn(pred_mag, mag_c)
        bs = x.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def main() -> None:
    h = HYPERPARAMS
    p = argparse.ArgumentParser(description="Train TinyDenoiser on noisy/clean wav pairs.")
    p.add_argument("--noisy-root", default=h["noisy_root"])
    p.add_argument("--clean-root", default=h["clean_root"])
    p.add_argument("--epochs", type=int, default=h["epochs"])
    p.add_argument("--batch-size", type=int, default=h["batch_size"])
    p.add_argument("--lr", type=float, default=h["lr"])
    p.add_argument("--hidden-dim", type=int, default=h["hidden_dim"])
    p.add_argument("--workers", type=int, default=h["workers"])
    p.add_argument("--val-fraction", type=float, default=h["val_fraction"])
    p.add_argument("--seed", type=int, default=h["seed"])
    p.add_argument("--device", default=h["device"], help="cuda | cpu (default: auto)")
    p.add_argument("--out-dir", default=h["out_dir"])
    p.add_argument("--run-tag", default=h["run_tag"])
    p.add_argument("--log-eps", type=float, default=h["log_eps"])
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = resolve_run_dir(args.out_dir, args.run_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run] output_dir={out_dir}")

    pairs, missing = collect_pairs(args.noisy_root, args.clean_root)
    if missing:
        print(f"[warn] {len(missing)} noisy files without clean twin (skipped).")
    if not pairs:
        print("No paired wav files found.", file=sys.stderr)
        sys.exit(1)

    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_fraction)
    n_val = max(min(n_val, len(pairs) - 1), 0) if len(pairs) > 1 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_rows = build_chunk_rows(train_pairs, label="train_chunks")
    val_rows = build_chunk_rows(val_pairs, label="val_chunks")
    print(
        f"[prep] chunk indeksleri: train={len(train_rows)} satır "
        f"({len(train_pairs)} çift), val={len(val_rows)} satır ({len(val_pairs)} çift)"
    )
    if not train_rows:
        print("No training chunks (files too short or unreadable).", file=sys.stderr)
        sys.exit(1)

    train_ds = AlignedChunkDataset(train_rows, log_eps=args.log_eps)
    val_ds = AlignedChunkDataset(val_rows, log_eps=args.log_eps)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    if len(train_loader) == 0:
        print(
            "Train DataLoader is empty (batch_size > number of train chunks). "
            "Lower --batch-size.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TinyDenoiser(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    cfg = vars(args).copy()
    cfg["out_dir_resolved"] = out_dir
    cfg["hyperparams_preset"] = {k: HYPERPARAMS[k] for k in HYPERPARAMS}
    cfg.update(
        {
            "freq_bins": FREQ_BINS,
            "num_pairs": len(pairs),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "train_chunks": len(train_rows),
            "val_chunks": len(val_rows),
            "batches_per_train_epoch": len(train_loader),
        }
    )
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    write_model_info_txt(os.path.join(out_dir, "model_info.txt"), model)

    started_at = _utc_now_iso()
    epoch_records: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    sum_epoch_sec = 0.0
    summary_path = os.path.join(out_dir, "train_summary.json")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device) if len(val_ds) else float("nan")
        epoch_sec = time.perf_counter() - t0
        sum_epoch_sec += epoch_sec

        rec = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "epoch_sec": round(epoch_sec, 3),
        }
        epoch_records.append(rec)

        if not math.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best_weights.pt"))

        msg = (
            f"epoch {epoch}/{args.epochs}  "
            f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  "
            f"epoch_sec={epoch_sec:.1f}"
        )
        print(msg)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_mse": train_loss,
            "val_mse": val_loss,
        }
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, "last_weights.pt"))

        has_best = os.path.isfile(os.path.join(out_dir, "best_weights.pt"))
        summary_payload = {
            "run_tag": args.run_tag or None,
            "run_dir": out_dir,
            "started_at": started_at,
            "epochs_done": epoch,
            "epochs_total_planned": args.epochs,
            "epochs": epoch_records,
            "best_val_mse": best_val if has_best and math.isfinite(best_val) else None,
            "best_epoch": best_epoch if has_best else None,
            "total_epoch_time_sec": round(sum_epoch_sec, 3),
            "artifacts": {
                "run_config": "run_config.json",
                "model_info": "model_info.txt",
                "train_summary": "train_summary.json",
                "last_pt": "last.pt",
                "last_weights": "last_weights.pt",
                "best_weights": "best_weights.pt" if has_best else None,
            },
        }
        if epoch == args.epochs:
            summary_payload["finished_at"] = _utc_now_iso()
        write_train_summary(summary_path, summary_payload)

    print(f"[done] saved under {out_dir}")


if __name__ == "__main__":
    main()
