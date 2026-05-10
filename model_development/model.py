"""Tiny spectral denoiser for 481-bin features from audio_pipeline STFT."""

from __future__ import annotations

import torch
import torch.nn as nn

from audio_pipeline import N_FFT

# One-sided STFT bins: matches stft_chunks output last dim when shape is (num_chunks, 481)
FREQ_BINS = N_FFT // 2 + 1


class TinyDenoiser(nn.Module):
    """
    Per-chunk denoiser on magnitude-domain features.

    Expected input: log(|X| + eps) with shape (batch, FREQ_BINS), float32.
    Output: multiplicative mask in (0, 1) with shape (batch, FREQ_BINS);
    apply as clean_mag = mask * noisy_mag (or same in log domain via user choice).
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FREQ_BINS, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, FREQ_BINS),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_info(model: TinyDenoiser | None = None) -> dict:
    """Static + optional instantiated model stats (params, shapes)."""
    m = model if model is not None else TinyDenoiser()
    info = {
        "name": "TinyDenoiser",
        "input_shape": f"(batch, {FREQ_BINS})",
        "output_shape": f"(batch, {FREQ_BINS})",
        "freq_bins": FREQ_BINS,
        "hidden_dim": m.net[0].out_features,
        "num_parameters": count_parameters(m),
        "notes": (
            "Input: log-magnitude of noisy STFT per chunk; "
            "output: sigmoid mask for magnitude denoising."
        ),
    }
    return info


def print_model_info(model: TinyDenoiser | None = None) -> None:
    for k, v in model_info(model).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    torch.manual_seed(0)
    net = TinyDenoiser()
    x = torch.randn(4, FREQ_BINS)
    y = net(x)
    print_model_info(net)
    print(f"  sanity forward: in {tuple(x.shape)} -> out {tuple(y.shape)}")
