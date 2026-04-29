import glob
import os
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import Dataset


def pair_chunk_files(noisy_root: str, clean_root: str) -> List[Tuple[str, str]]:
    # Match noisy-clean wav files by relative path.
    noisy_files = [
        p for p in glob.glob(os.path.join(noisy_root, "**", "*.wav"), recursive=True) if os.path.isfile(p)
    ]
    pairs = []
    for noisy_path in sorted(noisy_files):
        rel = os.path.relpath(noisy_path, noisy_root)
        clean_path = os.path.join(clean_root, rel)
        if os.path.exists(clean_path):
            pairs.append((noisy_path, clean_path))
    if not pairs:
        raise RuntimeError(f"No paired wav files found in {noisy_root} and {clean_root}")
    return pairs


def read_mono_wav(path: str, target_sr: int = 48000) -> np.ndarray:
    # Load wav as mono float32 and validate sample rate.
    audio, sr = sf.read(path, dtype="float32")
    if sr != target_sr:
        raise ValueError(f"Expected {target_sr}Hz but got {sr}Hz: {path}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


class ChunkPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        # Return number of paired chunks.
        return len(self.pairs)

    def __getitem__(self, idx: int):
        # Read one noisy-clean pair and return tensors.
        noisy_path, clean_path = self.pairs[idx]
        noisy = read_mono_wav(noisy_path)
        clean = read_mono_wav(clean_path)
        if noisy.shape != clean.shape:
            n = min(len(noisy), len(clean))
            noisy = noisy[:n]
            clean = clean[:n]
        return torch.from_numpy(noisy).unsqueeze(0), torch.from_numpy(clean).unsqueeze(0)


class TinyDenoiser(nn.Module):
    def __init__(self, hidden_size: int = 8):
        # Tiny Conv1d baseline denoiser.
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run forward pass through conv stack.
        return self.net(x)


def snr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Compute SNR in decibels.
    noise = reference - estimate
    ref_power = torch.mean(reference * reference, dim=-1) + eps
    noise_power = torch.mean(noise * noise, dim=-1) + eps
    return 10.0 * torch.log10(ref_power / noise_power)
