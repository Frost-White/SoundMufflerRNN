"""GRU sequence denoiser on per-chunk log-magnitude STFT features (FREQ_BINS)."""

from __future__ import annotations

import torch
import torch.nn as nn

N_FFT = 960
FREQ_BINS = N_FFT // 2 + 1


class GRUChunkDenoiser(nn.Module):
    """
    Utterance-level model: input (batch, time, FREQ_BINS) log(|X|+eps).
    Output multiplicative mask in (0,1), same shape; pred_mag = mask * mag_noisy.
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            FREQ_BINS,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, FREQ_BINS),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, FREQ_BINS). lengths: (B,) long, each <= T (CPU long ok for pack).
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        return self.mask_head(out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_info(model: GRUChunkDenoiser | None = None) -> dict:
    m = model if model is not None else GRUChunkDenoiser()
    return {
        "name": "GRUChunkDenoiser",
        "input_shape": f"(batch, time, {FREQ_BINS})",
        "output_shape": f"(batch, time, {FREQ_BINS})",
        "freq_bins": FREQ_BINS,
        "hidden_dim": m.hidden_dim,
        "gru_num_layers": m.num_layers,
        "num_parameters": count_parameters(m),
        "notes": (
            "Sequence of log-magnitude STFT rows per wav chunk; "
            "sigmoid mask per time step for magnitude denoising."
        ),
    }


def print_model_info(model: GRUChunkDenoiser | None = None) -> None:
    for k, v in model_info(model).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    torch.manual_seed(0)
    net = GRUChunkDenoiser(hidden_dim=64, num_layers=1)
    x = torch.randn(2, 5, FREQ_BINS)
    lengths = torch.tensor([5, 3])
    y = net(x, lengths)
    print_model_info(net)
    print(f"  sanity forward: in {tuple(x.shape)} -> out {tuple(y.shape)}")
