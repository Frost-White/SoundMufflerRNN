from __future__ import annotations

import torch
import torch.nn as nn

N_FFT = 960
FREQ_BINS = N_FFT // 2 + 1


class GRUChunkDenoiser(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
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
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        return self.mask_head(out)
