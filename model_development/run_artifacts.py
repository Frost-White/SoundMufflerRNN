"""Run directory layout, metrics CSV, plots, JSON summaries, model_info text."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.nn import Module

from model import model_info


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_run_dir(runs_root: str, out_dir: str | None, run_tag: str) -> str:
    if out_dir:
        return os.path.abspath(out_dir)
    tag = run_tag.strip().replace(" ", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{tag}" if tag else stamp
    return os.path.abspath(os.path.join(runs_root, name))


def write_model_info_txt(path: str, model: Module) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for k, v in model_info(model).items():
            f.write(f"{k}: {v}\n")


def write_train_summary(path: str, payload: dict) -> None:
    out = {**payload, "last_updated_at": utc_now_iso()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _csv_metric_float(v: object) -> float | int | str:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return v if isinstance(v, (int, float)) else ""


def write_run_plots(out_dir: str, epoch_records: list[dict]) -> None:
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
