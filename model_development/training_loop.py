"""Masked MSE training / validation for padded utterance batches."""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run_artifacts import utc_now_iso, write_metrics_train_csv, write_run_plots, write_train_summary


def _valid_time_mask(lengths: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
    """(B, T) bool: true where t < lengths[b]."""
    lengths = lengths.to(device)
    return torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)


def masked_mean_squared_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Scalar mean over valid (batch, time, freq) cells only."""
    _, T, F = pred.shape
    device = pred.device
    valid = _valid_time_mask(lengths, T, device).unsqueeze(-1).to(dtype=pred.dtype)
    se = ((pred - target) ** 2 * valid).sum()
    denom = (valid.sum() * F).clamp_min(1.0)
    return se / denom


def masked_sse_sum(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, T, F = pred.shape
    device = pred.device
    valid = _valid_time_mask(lengths, T, device).unsqueeze(-1).to(dtype=pred.dtype)
    se = ((pred - target) ** 2 * valid).sum()
    return se, valid.sum() * F


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_se = 0.0
    total_n = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        mag_n = batch["mag_noisy"].to(device)
        mag_c = batch["mag_clean"].to(device)
        lengths = batch["lengths"]

        optimizer.zero_grad(set_to_none=True)
        mask = model(x, lengths)
        pred_mag = mask * mag_n
        loss = masked_mean_squared_error(pred_mag, mag_c, lengths)
        loss.backward()
        optimizer.step()

        se, cnt = masked_sse_sum(pred_mag, mag_c, lengths)
        total_se += se.detach().item()
        total_n += cnt.detach().item()
    return total_se / max(total_n, 1.0)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """(mean MSE on valid cells, SNR gain dB vs noisy on same cells)."""
    model.eval()
    total_se = 0.0
    total_n = 0.0
    sse_noisy = 0.0
    sse_pred = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        mag_n = batch["mag_noisy"].to(device)
        mag_c = batch["mag_clean"].to(device)
        lengths = batch["lengths"]

        mask = model(x, lengths)
        pred_mag = mask * mag_n

        _, T, F = pred_mag.shape
        valid = _valid_time_mask(lengths, T, device).unsqueeze(-1).to(dtype=pred_mag.dtype)
        total_se += ((pred_mag - mag_c) ** 2 * valid).sum().item()
        sse_noisy += ((mag_n - mag_c) ** 2 * valid).sum().item()
        sse_pred += ((pred_mag - mag_c) ** 2 * valid).sum().item()
        total_n += (valid.sum() * F).item()

    if total_n <= 0:
        return float("nan"), float("nan")
    mse = total_se / total_n
    mse_noisy = sse_noisy / total_n
    mse_pred = sse_pred / total_n
    gain_db = 10.0 * math.log10(mse_noisy / (mse_pred + 1e-20))
    return mse, gain_db


def run_training_loop(
    hp: dict,
    out_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_ds_len: int,
    device: torch.device,
) -> None:
    epochs = hp["epochs"]
    started_at = utc_now_iso()
    epoch_records: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    sum_epoch_sec = 0.0
    summary_path = os.path.join(out_dir, "train_summary.json")
    metrics_csv_path = os.path.join(out_dir, "metrics_train.csv")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        if val_ds_len:
            val_loss, val_snr_gain_db = validate(model, val_loader, device)
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
            summary_payload["finished_at"] = utc_now_iso()
        write_train_summary(summary_path, summary_payload)
