"""Paired wav indexing, STFT preload in RAM, utterance-level Dataset for GRU training."""

from __future__ import annotations

import os
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from core.audio import chunk_waveform, load_audio, stft_chunks
from utils.progress import endline, tick


def scan_wavs_by_basename(root: str) -> dict[str, str]:
    out: dict[str, str] = {}
    dup_count = 0
    for cur, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(".wav"):
                if name in out:
                    dup_count += 1
                out[name] = os.path.join(cur, name)
    if dup_count:
        print(
            f"[warn] {root}: {dup_count} duplicate basename(s) detected; last path wins.",
            file=sys.stderr,
        )
    return out


def collect_pairs(
    noisy_root: str, clean_root: str
) -> tuple[list[tuple[str, str]], list[str]]:
    noisy_map = scan_wavs_by_basename(noisy_root)
    clean_map = scan_wavs_by_basename(clean_root)
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


def preload_stft_mag_pairs(
    pairs: list[tuple[str, str]],
    label: str = "preload",
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per pair: (mag_noisy[T,F], mag_clean[T,F], noisy_chunks[T,S], clean_chunks[T,S])."""
    total = len(pairs)
    if total:
        print(f"[{label}] {total} çift için |STFT| magnitüdleri RAM'e hazırlanıyor...")
    feats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, (noisy_path, clean_path) in enumerate(pairs, start=1):
        try:
            nw, _ = load_audio(noisy_path)
            cw, _ = load_audio(clean_path)
        except Exception as e:
            print(f"[skip] read failed: {noisy_path} ({e})", file=sys.stderr)
            tick(label, i, total)
            continue
        L = min(len(nw), len(cw))
        chunks_n = chunk_waveform(nw[:L], pad_end=True)
        chunks_c = chunk_waveform(cw[:L], pad_end=True)
        if chunks_n.shape[0] == 0:
            tick(label, i, total)
            continue
        mag_n = np.abs(stft_chunks(chunks_n)).astype(np.float32, copy=False)
        mag_c = np.abs(stft_chunks(chunks_c)).astype(np.float32, copy=False)
        chunks_n = np.ascontiguousarray(chunks_n.astype(np.float32, copy=False))
        chunks_c = np.ascontiguousarray(chunks_c.astype(np.float32, copy=False))
        feats[(noisy_path, clean_path)] = (mag_n, mag_c, chunks_n, chunks_c)
        tick(label, i, total)
    if total:
        endline()
    return feats


class UtteranceMagDataset(Dataset):
    """One sample = one wav pair's full chunk sequence (T, F)."""

    def __init__(
        self,
        pair_keys: list[tuple[str, str]],
        feats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        log_eps: float,
    ):
        self._keys = [k for k in pair_keys if k in feats]
        self._feats = feats
        self._log_eps = log_eps

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        key = self._keys[idx]
        mn, mc, chunks_n, chunks_c = self._feats[key]
        x = np.log(mn + self._log_eps).astype(np.float32, copy=False)
        return {
            "x": torch.from_numpy(x),
            "mag_noisy": torch.from_numpy(np.ascontiguousarray(mn)),
            "mag_clean": torch.from_numpy(np.ascontiguousarray(mc)),
            "chunks_noisy": torch.from_numpy(chunks_n),
            "chunks_clean": torch.from_numpy(chunks_c),
        }


def collate_padded_utterances(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([s["x"].shape[0] for s in samples], dtype=torch.long)
    x = pad_sequence([s["x"] for s in samples], batch_first=True)
    mag_n = pad_sequence([s["mag_noisy"] for s in samples], batch_first=True)
    mag_c = pad_sequence([s["mag_clean"] for s in samples], batch_first=True)
    chunks_n = pad_sequence([s["chunks_noisy"] for s in samples], batch_first=True)
    chunks_c = pad_sequence([s["chunks_clean"] for s in samples], batch_first=True)
    return {
        "x": x,
        "mag_noisy": mag_n,
        "mag_clean": mag_c,
        "chunks_noisy": chunks_n,
        "chunks_clean": chunks_c,
        "lengths": lengths,
    }


def prepare_train_val_datasets(
    noisy_root: str,
    clean_root: str,
    val_fraction: float,
    log_eps: float,
) -> tuple[UtteranceMagDataset, UtteranceMagDataset, dict[str, Any]]:
    print("[data] çift listesi taranıyor...")
    pairs, missing = collect_pairs(noisy_root, clean_root)
    if missing:
        print(f"[warn] {len(missing)} noisy files without clean twin (skipped).")
    if not pairs:
        print("No paired wav files found.", file=sys.stderr)
        sys.exit(1)
    print(f"[data] {len(pairs)} eşleşmiş çift")

    feats = preload_stft_mag_pairs(pairs, label="preload")
    pairs = [p for p in pairs if p in feats]
    if not pairs:
        print("All pairs failed to load.", file=sys.stderr)
        sys.exit(1)
    print(f"[data] {len(pairs)} çift RAM'de")

    random.shuffle(pairs)
    n_val = int(len(pairs) * val_fraction)
    n_val = max(min(n_val, len(pairs) - 1), 0) if len(pairs) > 1 else 0
    val_keys = pairs[:n_val]
    train_keys = pairs[n_val:]

    train_feats = {k: feats[k] for k in train_keys}
    val_feats = {k: feats[k] for k in val_keys}
    del feats

    train_ds = UtteranceMagDataset(train_keys, train_feats, log_eps)
    val_ds = UtteranceMagDataset(val_keys, val_feats, log_eps)

    n_chunks_train = sum(train_feats[k][0].shape[0] for k in train_keys)
    n_chunks_val = sum(val_feats[k][0].shape[0] for k in val_keys)

    info = {
        "num_pairs": len(pairs),
        "train_pairs": len(train_keys),
        "val_pairs": len(val_keys),
        "train_utterances": len(train_keys),
        "val_utterances": len(val_keys),
        "train_chunks_total": int(n_chunks_train),
        "val_chunks_total": int(n_chunks_val),
    }
    if len(train_ds) == 0:
        print("No training utterances.", file=sys.stderr)
        sys.exit(1)

    print(
        f"[data] train={len(train_ds)} dosya ({n_chunks_train} chunk), "
        f"val={len(val_ds)} dosya ({n_chunks_val} chunk)"
    )
    return train_ds, val_ds, info
