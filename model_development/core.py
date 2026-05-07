import glob
import os
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import Dataset


def pair_wav_files(noisy_root: str, clean_root: str) -> List[Tuple[str, str]]:
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
    audio, sr = sf.read(path, dtype="float32")
    if sr != target_sr:
        raise ValueError(f"Expected {target_sr}Hz but got {sr}Hz: {path}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def chunk_waveform(audio: np.ndarray, chunk_size: int, hop: int) -> np.ndarray:
    if len(audio) < chunk_size:
        return np.empty((0, chunk_size), dtype=np.float32)
    starts = range(0, len(audio) - chunk_size + 1, hop)
    out = np.empty((len(starts), chunk_size), dtype=np.float32)
    for i, start in enumerate(starts):
        out[i] = audio[start : start + chunk_size]
    return out


class WavPairDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        chunk_size: int = 960,
        hop: int = 720,
        target_sr: int = 48000,
    ):
        self.chunk_size = chunk_size
        self.hop = hop
        self.target_sr = target_sr
        self.pairs = []
        for noisy_path, clean_path in pairs:
            n = sf.info(noisy_path).frames
            m = sf.info(clean_path).frames
            if min(n, m) >= chunk_size:
                self.pairs.append((noisy_path, clean_path))
        if not self.pairs:
            raise RuntimeError("No wav pairs long enough for chunk_size")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx]
        noisy = read_mono_wav(noisy_path, self.target_sr)
        clean = read_mono_wav(clean_path, self.target_sr)
        if len(noisy) != len(clean):
            n = min(len(noisy), len(clean))
            noisy = noisy[:n]
            clean = clean[:n]
        noisy_c = chunk_waveform(noisy, self.chunk_size, self.hop)
        clean_c = chunk_waveform(clean, self.chunk_size, self.hop)
        x = torch.from_numpy(noisy_c).unsqueeze(1).float()
        y = torch.from_numpy(clean_c).unsqueeze(1).float()
        return x, y


class TinyDenoiser(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def snr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    noise = reference - estimate
    ref_power = torch.mean(reference * reference, dim=-1) + eps
    noise_power = torch.mean(noise * noise, dim=-1) + eps
    return 10.0 * torch.log10(ref_power / noise_power)
