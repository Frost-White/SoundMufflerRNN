"""Train TinyDenoiser on noisy/clean pairs (chunk-level batches, aligned STFT)."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from audio_pipeline import chunk_waveform, load_audio, stft_chunks
from model import FREQ_BINS, TinyDenoiser, model_info

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NOISY = os.path.normpath(os.path.join(_BASE, "..", "data", "train", "noisy_trainset_56spk_wav"))
_DEFAULT_CLEAN = os.path.normpath(os.path.join(_BASE, "..", "data", "train", "clean_trainset_56spk_wav"))

# --- hiperparametreler: buradan düzenle ---
HYPERPARAMS = {
    # veri
    "noisy_root": _DEFAULT_NOISY,
    "clean_root": _DEFAULT_CLEAN,
    # None -> yeni klasör: runs/<YYYYMMDD_HHMMSS>_<run_tag>
    "out_dir": None,
    "run_tag": "",  # otomatik: h{hidden_dim}_bs{batch_size}_lr{lr}
    "val_fraction": 0.1,
    # model
    "hidden_dim": 2048,
    # eğitim
    "epochs": 20,
    "batch_size": 1024,
    "lr": 1e-3,
    "workers": 0,
    "seed": 0,
    "device": "cuda",  # "cuda" | "cpu" | None (None -> cuda varsa cuda)
    # özellik / kayıp
    "log_eps": 1e-8,
}
HYPERPARAMS["run_tag"] = f"h{HYPERPARAMS['hidden_dim']}_bs{HYPERPARAMS['batch_size']}_lr{HYPERPARAMS['lr']}"

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


def _csv_metric_float(v: object) -> float | int | str:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return v if isinstance(v, (int, float)) else ""


def write_run_plots(out_dir: str, epoch_records: list[dict]) -> None:
    """Kayıt: loss_curve.png (train/val MSE), snr_curve.png (val SNR kazancı dB)."""
    if not epoch_records:
        return
    epochs = [int(r["epoch"]) for r in epoch_records]
    train = [r.get("train_mse", r.get("train_loss")) for r in epoch_records]
    val = [r.get("val_mse", r.get("val_loss")) for r in epoch_records]
    snr_db = [r.get("val_snr_gain_db") for r in epoch_records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, label="train MSE")
    val_xy = [(e, v) for e, v in zip(epochs, val) if isinstance(v, (int, float)) and math.isfinite(v)]
    if val_xy:
        ex, vx = zip(*val_xy)
        ax.plot(ex, vx, label="val MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    snr_xy = [(e, s) for e, s in zip(epochs, snr_db) if isinstance(s, (int, float)) and math.isfinite(s)]
    if snr_xy:
        ex2, sy = zip(*snr_xy)
        ax2.plot(ex2, sy, color="C1", label="val SNR gain (dB)")
        ax2.legend()
    else:
        ax2.text(
            0.5,
            0.5,
            "no val SNR (empty val set or NaN)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("SNR gain (dB)")
    ax2.set_title("Validation SNR gain vs clean")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "snr_curve.png"), dpi=120)
    plt.close(fig2)


def write_metrics_train_csv(path: str, epoch_records: list[dict]) -> None:
    """Per-epoch metrics (matches legacy runs e.g. no_stft_bs32_hidden8/metrics_train.csv)."""
    fieldnames = ["epoch", "train_loss", "val_loss", "val_snr_gain_db", "epoch_time_s"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_records:
            w.writerow(
                {
                    "epoch": r["epoch"],
                    "train_loss": _csv_metric_float(r.get("train_mse", r.get("train_loss"))),
                    "val_loss": _csv_metric_float(r.get("val_mse", r.get("val_loss"))),
                    "val_snr_gain_db": _csv_metric_float(r.get("val_snr_gain_db")),
                    "epoch_time_s": r.get("epoch_sec", r.get("epoch_time_s", "")),
                }
            )


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


def _scan_wavs(root: str) -> dict[str, str]:
    """basename -> full path under `root` for every .wav file."""
    out: dict[str, str] = {}
    for cur, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(".wav"):
                out[name] = os.path.join(cur, name)
    return out


def collect_pairs(noisy_root: str, clean_root: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Pair noisy/clean wavs by basename (tek O(N) tarama her tarafta)."""
    noisy_map = _scan_wavs(noisy_root)
    clean_map = _scan_wavs(clean_root)
    pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    for name in sorted(noisy_map):
        np_path = noisy_map[name]
        cp = clean_map.get(name)
        if cp is None:
            missing.append(np_path)
        else:
            pairs.append((np_path, cp))
    return pairs, missing


def preload_stft_features(
    pairs: list[tuple[str, str]], label: str = "preload"
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Disk -> chunk -> |STFT| -> RAM. Per pair: (mag_noisy[N,F], mag_clean[N,F]) float32."""
    total = len(pairs)
    if total:
        print(f"[{label}] {total} çift için |STFT| magnitüdleri RAM'e hazırlanıyor...")
    feats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for i, (noisy_path, clean_path) in enumerate(pairs, start=1):
        try:
            nw, _ = load_audio(noisy_path)
            cw, _ = load_audio(clean_path)
        except Exception as e:
            print(f"[skip] read failed: {noisy_path} ({e})", file=sys.stderr)
            _progress_tick(label, i, total)
            continue
        L = min(len(nw), len(cw))
        chunks_n = chunk_waveform(nw[:L])
        chunks_c = chunk_waveform(cw[:L])
        if chunks_n.shape[0] == 0:
            _progress_tick(label, i, total)
            continue
        mag_n = np.abs(stft_chunks(chunks_n)).astype(np.float32, copy=False)
        mag_c = np.abs(stft_chunks(chunks_c)).astype(np.float32, copy=False)
        feats[(noisy_path, clean_path)] = (mag_n, mag_c)
        _progress_tick(label, i, total)
    if total:
        _progress_end()
    return feats


def concat_split_features(
    pairs: list[tuple[str, str]],
    feats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-pair magnitudes into two contiguous arrays for a train/val split."""
    parts_n: list[np.ndarray] = []
    parts_c: list[np.ndarray] = []
    for key in pairs:
        item = feats.get(key)
        if item is None:
            continue
        mn, mc = item
        parts_n.append(mn)
        parts_c.append(mc)
    if not parts_n:
        empty = np.empty((0, FREQ_BINS), dtype=np.float32)
        return empty, empty
    return np.concatenate(parts_n, axis=0), np.concatenate(parts_c, axis=0)


class AlignedChunkDataset(Dataset):
    """Each sample = one chunk of precomputed |STFT| magnitudes living in RAM."""

    def __init__(
        self,
        mag_noisy: np.ndarray,
        mag_clean: np.ndarray,
        log_eps: float = _LOG_EPS,
    ):
        assert mag_noisy.shape == mag_clean.shape, (mag_noisy.shape, mag_clean.shape)
        self._mn = mag_noisy
        self._mc = mag_clean
        self._log_eps = log_eps

    def __len__(self) -> int:
        return self._mn.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mn = self._mn[idx]
        mc = self._mc[idx]
        x = np.log(mn + self._log_eps)
        return {
            "x": torch.from_numpy(x),
            "mag_noisy": torch.from_numpy(mn),
            "mag_clean": torch.from_numpy(mc),
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
) -> tuple[float, float]:
    """Returns (val MSE, val SNR gain vs noisy in dB).

    SNR gain = 10*log10( MSE(noisy,clean) / MSE(pred,clean) ); 0 dB ≈ identity mask.
    """
    model.eval()
    total = 0.0
    n = 0
    sse_noisy = 0.0
    sse_pred = 0.0
    n_elem = 0
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
        sse_noisy += ((mag_n - mag_c) ** 2).sum().item()
        sse_pred += ((pred_mag - mag_c) ** 2).sum().item()
        n_elem += mag_n.numel()
    if n == 0:
        return float("nan"), float("nan")
    avg_loss = total / n
    if n_elem == 0:
        return avg_loss, float("nan")
    mse_noisy = sse_noisy / n_elem
    mse_pred = sse_pred / n_elem
    gain_db = 10.0 * math.log10(mse_noisy / (mse_pred + 1e-20))
    return avg_loss, gain_db


def prepare_datasets(hp: dict) -> tuple[AlignedChunkDataset, AlignedChunkDataset, dict]:
    """Çiftleri tara, |STFT|'leri RAM'e yükle, train/val split + dataset kur."""
    print("[step] (1/5) çift listesi taranıyor...")
    pairs, missing = collect_pairs(hp["noisy_root"], hp["clean_root"])
    if missing:
        print(f"[warn] {len(missing)} noisy files without clean twin (skipped).")
    if not pairs:
        print("No paired wav files found.", file=sys.stderr)
        sys.exit(1)
    print(f"[step] (1/5) tamam: {len(pairs)} eşleşmiş çift")

    print("[step] (2/5) disk -> |STFT| -> RAM ön yükleme...")
    feats = preload_stft_features(pairs, label="preload")
    pairs = [p for p in pairs if p in feats]
    if not pairs:
        print("All pairs failed to load.", file=sys.stderr)
        sys.exit(1)
    print(f"[step] (2/5) tamam: {len(pairs)} çift RAM'de")

    print("[step] (3/5) train/val split ve birleştirme...")
    random.shuffle(pairs)
    n_val = int(len(pairs) * hp["val_fraction"])
    n_val = max(min(n_val, len(pairs) - 1), 0) if len(pairs) > 1 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_mag_n, train_mag_c = concat_split_features(train_pairs, feats)
    val_mag_n, val_mag_c = concat_split_features(val_pairs, feats)
    del feats  # per-pair arrays artık concat'lerin içinde

    print(
        f"[step] (3/5) tamam: train={train_mag_n.shape[0]} chunk ({len(train_pairs)} çift), "
        f"val={val_mag_n.shape[0]} chunk ({len(val_pairs)} çift)"
    )
    if train_mag_n.shape[0] == 0:
        print("No training chunks (files too short or unreadable).", file=sys.stderr)
        sys.exit(1)

    train_ds = AlignedChunkDataset(train_mag_n, train_mag_c, log_eps=hp["log_eps"])
    val_ds = AlignedChunkDataset(val_mag_n, val_mag_c, log_eps=hp["log_eps"])

    info = {
        "num_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_chunks": int(train_mag_n.shape[0]),
        "val_chunks": int(val_mag_n.shape[0]),
    }
    return train_ds, val_ds, info


def run_training_loop(
    hp: dict,
    out_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_ds_len: int,
    device: torch.device,
) -> None:
    epochs = hp["epochs"]
    started_at = _utc_now_iso()
    epoch_records: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    sum_epoch_sec = 0.0
    summary_path = os.path.join(out_dir, "train_summary.json")
    metrics_csv_path = os.path.join(out_dir, "metrics_train.csv")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        if val_ds_len:
            val_loss, val_snr_gain_db = validate(model, val_loader, loss_fn, device)
        else:
            val_loss, val_snr_gain_db = float("nan"), float("nan")
        epoch_sec = time.perf_counter() - t0
        sum_epoch_sec += epoch_sec

        rec = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "val_snr_gain_db": val_snr_gain_db,
            "epoch_sec": round(epoch_sec, 3),
        }
        epoch_records.append(rec)
        write_metrics_train_csv(metrics_csv_path, epoch_records)
        write_run_plots(out_dir, epoch_records)

        if not math.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best_weights.pt"))

        sg = (
            f"{val_snr_gain_db:.4f}"
            if isinstance(val_snr_gain_db, float) and math.isfinite(val_snr_gain_db)
            else "nan"
        )
        print(
            f"epoch {epoch}/{epochs}  "
            f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  "
            f"val_snr_gain_db={sg}  epoch_sec={epoch_sec:.1f}"
        )

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
            "run_tag": hp["run_tag"] or None,
            "run_dir": out_dir,
            "started_at": started_at,
            "epochs_done": epoch,
            "epochs_total_planned": epochs,
            "best_val_mse": best_val if has_best and math.isfinite(best_val) else None,
            "best_epoch": best_epoch if has_best else None,
            "total_epoch_time_sec": round(sum_epoch_sec, 3),
            "artifacts": {
                "run_config": "run_config.json",
                "model_info": "model_info.txt",
                "train_summary": "train_summary.json",
                "metrics_train_csv": "metrics_train.csv",
                "loss_curve_png": "loss_curve.png",
                "snr_curve_png": "snr_curve.png",
                "last_pt": "last.pt",
                "last_weights": "last_weights.pt",
                "best_weights": "best_weights.pt" if has_best else None,
            },
        }
        if epoch == epochs:
            summary_payload["finished_at"] = _utc_now_iso()
        write_train_summary(summary_path, summary_payload)


def main() -> None:
    hp = HYPERPARAMS
    random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])

    out_dir = resolve_run_dir(hp["out_dir"], hp["run_tag"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run] output_dir={out_dir}")

    train_ds, val_ds, info = prepare_datasets(hp)

    print("[step] (4/5) model ve loader hazırlanıyor...")
    train_loader = DataLoader(
        train_ds,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=hp["workers"],
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=hp["workers"],
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    if len(train_loader) == 0:
        print(
            "Train DataLoader is empty (batch_size > number of train chunks). "
            "Lower batch_size.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device(hp["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TinyDenoiser(hidden_dim=hp["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.MSELoss()

    cfg = dict(hp)
    cfg["out_dir_resolved"] = out_dir
    cfg.update({"freq_bins": FREQ_BINS, "batches_per_train_epoch": len(train_loader), **info})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    write_model_info_txt(os.path.join(out_dir, "model_info.txt"), model)
    print(f"[step] (4/5) tamam: device={device}  batches/epoch={len(train_loader)}")

    print(f"[step] (5/5) eğitim başlıyor ({hp['epochs']} epoch)")
    run_training_loop(
        hp, out_dir, model, optimizer, loss_fn, train_loader, val_loader, len(val_ds), device
    )
    print(f"[done] saved under {out_dir}")


if __name__ == "__main__":
    main()
