"""Masked log-magnitude objective with optional linear and MR-STFT terms."""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.audio import CHUNK_HOP, N_FFT, analysis_stft_chunks_torch, synthesis_istft_chunks_torch
from training.artifacts import utc_now_iso, write_metrics_train_csv, write_run_plots, write_train_summary


def _valid_time_mask(lengths: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
    """(B, T) bool: true where t < lengths[b]."""
    lengths = lengths.to(device)
    return torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)


def _masked_sse_sum(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(sum squared error, valid cell count) over valid (batch,time,freq) cells."""
    _, T, F = pred.shape
    device = pred.device
    valid = _valid_time_mask(lengths, T, device).unsqueeze(-1).to(dtype=pred.dtype)
    return ((pred - target) ** 2 * valid).sum(), valid.sum() * F


def masked_mean_squared_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    se, denom = _masked_sse_sum(pred, target, lengths)
    return se / denom.clamp_min(1.0)


def masked_log_mean_squared_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return masked_mean_squared_error(torch.log(pred + eps), torch.log(target + eps), lengths)


def _pred_chunks_from_mask(
    mask: torch.Tensor,
    chunks_noisy: torch.Tensor,
) -> torch.Tensor:
    flat_chunks = chunks_noisy.reshape(-1, chunks_noisy.shape[-1])
    noisy_spec_flat = analysis_stft_chunks_torch(flat_chunks)
    noisy_spec = noisy_spec_flat.reshape(*chunks_noisy.shape[:-1], noisy_spec_flat.shape[-1])
    pred_spec = mask.to(dtype=noisy_spec.dtype) * noisy_spec
    pred_spec_flat = pred_spec.reshape(-1, pred_spec.shape[-1])
    pred_chunks_flat = synthesis_istft_chunks_torch(pred_spec_flat, length=chunks_noisy.shape[-1])
    return pred_chunks_flat.reshape(*pred_spec.shape[:-1], chunks_noisy.shape[-1])


def _overlap_add_average_torch(chunks: torch.Tensor, hop: int = CHUNK_HOP) -> torch.Tensor:
    if chunks.shape[0] == 0:
        return chunks.new_zeros(0)
    n_chunks, wlen = chunks.shape
    total = (n_chunks - 1) * hop + wlen
    out = chunks.new_zeros(total)
    weight = chunks.new_zeros(total)
    win = torch.hann_window(wlen, periodic=True, dtype=chunks.dtype, device=chunks.device)
    for i in range(n_chunks):
        start = i * hop
        out[start : start + wlen] = out[start : start + wlen] + chunks[i]
        weight[start : start + wlen] = weight[start : start + wlen] + win
    min_weight = 1e-3
    stable = weight >= min_weight
    out_stable = out.clone()
    out_stable[stable] = out_stable[stable] / weight[stable]
    out_stable[~stable] = 0.0
    return out_stable


def _mrstft_one(
    pred_wave: torch.Tensor,
    clean_wave: torch.Tensor,
    resolutions: list[tuple[int, int, int]],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not resolutions:
        return pred_wave.new_tensor(0.0)
    total = pred_wave.new_tensor(0.0)
    for n_fft, win_length, hop_length in resolutions:
        win = torch.hann_window(win_length, device=pred_wave.device, dtype=pred_wave.dtype)
        pred_stft = torch.stft(
            pred_wave,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=win,
            center=False,
            return_complex=True,
        )
        clean_stft = torch.stft(
            clean_wave,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=win,
            center=False,
            return_complex=True,
        )
        pred_mag = pred_stft.abs()
        clean_mag = clean_stft.abs()
        spectral_convergence = torch.linalg.norm(pred_mag - clean_mag) / (
            torch.linalg.norm(clean_mag) + eps
        )
        log_mag = torch.mean(torch.abs(torch.log(pred_mag + eps) - torch.log(clean_mag + eps)))
        total = total + spectral_convergence + log_mag
    return total / float(len(resolutions))


def _multi_resolution_stft_loss(
    pred_chunks: torch.Tensor,
    clean_chunks: torch.Tensor,
    lengths: torch.Tensor,
    resolutions: list[tuple[int, int, int]],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not resolutions:
        return pred_chunks.new_tensor(0.0)
    B = pred_chunks.shape[0]
    total = pred_chunks.new_tensor(0.0)
    count = 0
    for b in range(B):
        t = int(lengths[b].item())
        if t <= 0:
            continue
        pred_wave = _overlap_add_average_torch(pred_chunks[b, :t])
        clean_wave = _overlap_add_average_torch(clean_chunks[b, :t])
        n = min(pred_wave.shape[0], clean_wave.shape[0])
        if n <= 0:
            continue
        total = total + _mrstft_one(pred_wave[:n], clean_wave[:n], resolutions, eps=eps)
        count += 1
    if count == 0:
        return pred_chunks.new_tensor(0.0)
    return total / float(count)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    w_log_mag: float,
    w_linear_mag: float,
    w_mrstft: float,
    mrstft_resolutions: list[tuple[int, int, int]],
    loss_log_eps: float,
) -> tuple[float, float, float, float]:
    model.train()
    total_lin_se = 0.0
    total_log_se = 0.0
    total_n = 0.0
    total_mrstft = 0.0
    num_batches = 0
    for batch in loader:
        x = batch["x"].to(device)
        mag_n = batch["mag_noisy"].to(device)
        mag_c = batch["mag_clean"].to(device)
        chunks_n = batch["chunks_noisy"].to(device)
        chunks_c = batch["chunks_clean"].to(device)
        lengths = batch["lengths"]

        optimizer.zero_grad(set_to_none=True)
        mask = model(x, lengths)
        pred_mag = mask * mag_n
        linear_mag_mse = masked_mean_squared_error(pred_mag, mag_c, lengths)
        log_mag_mse = masked_log_mean_squared_error(pred_mag, mag_c, lengths, loss_log_eps)
        pred_chunks = _pred_chunks_from_mask(mask, chunks_n)
        mrstft_loss = _multi_resolution_stft_loss(
            pred_chunks,
            chunks_c,
            lengths,
            mrstft_resolutions,
            eps=loss_log_eps,
        )
        loss = w_log_mag * log_mag_mse + w_linear_mag * linear_mag_mse + w_mrstft * mrstft_loss
        loss.backward()
        optimizer.step()

        lin_se, cnt = _masked_sse_sum(pred_mag, mag_c, lengths)
        log_se, _ = _masked_sse_sum(
            torch.log(pred_mag + loss_log_eps),
            torch.log(mag_c + loss_log_eps),
            lengths,
        )
        total_lin_se += lin_se.detach().item()
        total_log_se += log_se.detach().item()
        total_n += cnt.detach().item()
        total_mrstft += mrstft_loss.detach().item()
        num_batches += 1
    mean_linear = total_lin_se / max(total_n, 1.0)
    mean_log = total_log_se / max(total_n, 1.0)
    mean_mrstft = total_mrstft / max(num_batches, 1)
    mean_total = w_log_mag * mean_log + w_linear_mag * mean_linear + w_mrstft * mean_mrstft
    return mean_linear, mean_log, mean_mrstft, mean_total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    w_log_mag: float,
    w_linear_mag: float,
    w_mrstft: float,
    mrstft_resolutions: list[tuple[int, int, int]],
    loss_log_eps: float,
) -> tuple[float, float, float, float, float]:
    """(linear MSE, log MSE, MRSTFT, total, SNR gain dB on valid cells)."""
    model.eval()
    total_lin_se = 0.0
    total_log_se = 0.0
    total_n = 0.0
    sse_noisy = 0.0
    sse_pred = 0.0
    total_mrstft = 0.0
    num_batches = 0
    for batch in loader:
        x = batch["x"].to(device)
        mag_n = batch["mag_noisy"].to(device)
        mag_c = batch["mag_clean"].to(device)
        chunks_n = batch["chunks_noisy"].to(device)
        chunks_c = batch["chunks_clean"].to(device)
        lengths = batch["lengths"]

        mask = model(x, lengths)
        pred_mag = mask * mag_n
        pred_chunks = _pred_chunks_from_mask(mask, chunks_n)
        mrstft_loss = _multi_resolution_stft_loss(
            pred_chunks,
            chunks_c,
            lengths,
            mrstft_resolutions,
            eps=loss_log_eps,
        )

        _, T, F = pred_mag.shape
        valid = _valid_time_mask(lengths, T, device).unsqueeze(-1).to(dtype=pred_mag.dtype)
        total_lin_se += ((pred_mag - mag_c) ** 2 * valid).sum().item()
        total_log_se += (
            (torch.log(pred_mag + loss_log_eps) - torch.log(mag_c + loss_log_eps)) ** 2 * valid
        ).sum().item()
        sse_noisy += ((mag_n - mag_c) ** 2 * valid).sum().item()
        sse_pred += ((pred_mag - mag_c) ** 2 * valid).sum().item()
        total_n += (valid.sum() * F).item()
        total_mrstft += mrstft_loss.item()
        num_batches += 1

    if total_n <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    linear_mag_mse = total_lin_se / total_n
    log_mag_mse = total_log_se / total_n
    mrstft = total_mrstft / max(num_batches, 1)
    total = w_log_mag * log_mag_mse + w_linear_mag * linear_mag_mse + w_mrstft * mrstft
    mse_noisy = sse_noisy / total_n
    mse_pred = sse_pred / total_n
    gain_db = 10.0 * math.log10(mse_noisy / (mse_pred + 1e-20))
    return linear_mag_mse, log_mag_mse, mrstft, total, gain_db


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
    w_log_mag = float(hp.get("w_log_mag", 1.0))
    w_linear_mag = float(hp.get("w_linear_mag", hp.get("mse_weight", 0.0)))
    w_mrstft = float(hp.get("w_mrstft", hp.get("mss_weight", 0.0)))
    loss_log_eps = float(hp.get("loss_log_eps", hp.get("log_eps", 1e-8)))
    if "mrstft_resolutions" in hp:
        mrstft_resolutions = [tuple(int(v) for v in r) for r in hp.get("mrstft_resolutions", [])]
    else:
        fft_sizes = [int(v) for v in hp.get("mss_fft_sizes", [240, 480, N_FFT])]
        mrstft_resolutions = [(n, n, max(1, n // 4)) for n in fft_sizes]
    started_at = utc_now_iso()
    epoch_records: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    sum_epoch_sec = 0.0
    summary_path = os.path.join(out_dir, "train_summary.json")
    metrics_csv_path = os.path.join(out_dir, "metrics_train.csv")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_linear, train_log, train_mrstft, train_total = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            w_log_mag,
            w_linear_mag,
            w_mrstft,
            mrstft_resolutions,
            loss_log_eps,
        )
        if val_ds_len:
            val_linear, val_log, val_mrstft, val_total, val_snr_gain_db = validate(
                model,
                val_loader,
                device,
                w_log_mag,
                w_linear_mag,
                w_mrstft,
                mrstft_resolutions,
                loss_log_eps,
            )
        else:
            val_linear, val_log, val_mrstft, val_total, val_snr_gain_db = (
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )
        epoch_sec = time.perf_counter() - t0
        sum_epoch_sec += epoch_sec

        rec = {
            "epoch": epoch,
            "train_linear_mag_mse": train_linear,
            "train_log_mag_mse": train_log,
            "train_mrstft": train_mrstft,
            "train_total": train_total,
            "val_linear_mag_mse": val_linear,
            "val_log_mag_mse": val_log,
            "val_mrstft": val_mrstft,
            "val_total": val_total,
            "val_snr_gain_db": val_snr_gain_db,
            "epoch_sec": round(epoch_sec, 3),
        }
        epoch_records.append(rec)
        write_metrics_train_csv(metrics_csv_path, epoch_records)
        write_run_plots(out_dir, epoch_records)

        if not math.isnan(val_total) and val_total < best_val:
            best_val = val_total
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best_weights.pt"))

        sg = (
            f"{val_snr_gain_db:.4f}"
            if isinstance(val_snr_gain_db, float) and math.isfinite(val_snr_gain_db)
            else "nan"
        )
        print(
            f"epoch {epoch}/{epochs}  "
            f"train_total={train_total:.6f} (log={train_log:.6f}, lin={train_linear:.6f}, mr={train_mrstft:.6f})  "
            f"val_total={val_total:.6f} (log={val_log:.6f}, lin={val_linear:.6f}, mr={val_mrstft:.6f})  "
            f"val_snr_gain_db={sg}  epoch_sec={epoch_sec:.1f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_linear_mag_mse": train_linear,
            "train_log_mag_mse": train_log,
            "train_mrstft": train_mrstft,
            "train_total": train_total,
            "val_linear_mag_mse": val_linear,
            "val_log_mag_mse": val_log,
            "val_mrstft": val_mrstft,
            "val_total": val_total,
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
            "best_val_total": best_val if has_best and math.isfinite(best_val) else None,
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
