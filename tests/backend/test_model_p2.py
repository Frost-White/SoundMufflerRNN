import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.inference.model import FREQ_BINS, GRUChunkDenoiser  # noqa: E402


def test_model_forward_shape_and_mask_range() -> None:
    model = GRUChunkDenoiser(hidden_dim=16, num_layers=2, dropout=0.1)
    x = torch.randn(2, 5, FREQ_BINS)
    lengths = torch.tensor([5, 3], dtype=torch.long)
    out = model(x, lengths)
    assert tuple(out.shape) == (2, 5, FREQ_BINS)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_model_ignores_dropout_for_single_layer() -> None:
    model = GRUChunkDenoiser(hidden_dim=8, num_layers=1, dropout=0.7)
    assert model.gru.dropout == 0.0


def test_model_handles_unsorted_lengths() -> None:
    model = GRUChunkDenoiser(hidden_dim=8, num_layers=2, dropout=0.0)
    x = torch.randn(3, 4, FREQ_BINS)
    lengths = torch.tensor([2, 4, 3], dtype=torch.long)
    out = model(x, lengths)
    assert tuple(out.shape) == (3, 4, FREQ_BINS)
