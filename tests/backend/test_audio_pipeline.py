import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.inference import audio_pipeline as ap  # noqa: E402


def test_chunk_waveform_rejects_non_1d() -> None:
    wav = np.zeros((2, ap.CHUNK_SAMPLES), dtype=np.float32)
    with pytest.raises(ValueError):
        ap.chunk_waveform(wav)


def test_chunk_waveform_empty_with_pad_end_returns_empty_matrix() -> None:
    wav = np.array([], dtype=np.float32)
    chunks = ap.chunk_waveform(wav, pad_end=True)
    assert chunks.shape == (0, ap.CHUNK_SAMPLES)


def test_chunk_waveform_short_without_pad_end_returns_empty_matrix() -> None:
    wav = np.ones(ap.CHUNK_SAMPLES - 8, dtype=np.float32)
    chunks = ap.chunk_waveform(wav, pad_end=False)
    assert chunks.shape == (0, ap.CHUNK_SAMPLES)


def test_chunk_waveform_short_with_pad_end_returns_one_padded_chunk() -> None:
    wav = np.array([0.25, -0.25, 0.5], dtype=np.float32)
    chunks = ap.chunk_waveform(wav, pad_end=True)
    assert chunks.shape == (1, ap.CHUNK_SAMPLES)
    assert np.allclose(chunks[0, : len(wav)], wav)
    assert np.allclose(chunks[0, len(wav) :], 0.0)


def test_analysis_stft_chunks_empty_returns_expected_shape() -> None:
    chunks = np.empty((0, ap.CHUNK_SAMPLES), dtype=np.float32)
    spec = ap.analysis_stft_chunks(chunks)
    assert spec.shape == (0, ap.N_FFT // 2 + 1)
    assert spec.dtype == np.complex64


def test_analysis_stft_chunks_rejects_unknown_window() -> None:
    chunks = np.zeros((1, ap.CHUNK_SAMPLES), dtype=np.float32)
    with pytest.raises(ValueError):
        ap.analysis_stft_chunks(chunks, window="rect")


def test_synthesis_istft_chunks_empty_returns_zero_matrix() -> None:
    spectra = np.empty((0, ap.N_FFT // 2 + 1), dtype=np.complex64)
    out = ap.synthesis_istft_chunks(spectra)
    assert out.shape == (0, ap.CHUNK_SAMPLES)
    assert out.dtype == np.float32


def test_project_to_stft_consistency_torch_rejects_invalid_rank() -> None:
    bad = torch.zeros((1, 2, 3, 4), dtype=torch.complex64)
    with pytest.raises(ValueError):
        ap.project_to_stft_consistency_torch(bad)


def test_blend_consistent_spectra_clamps_blend_into_unit_interval() -> None:
    raw = np.array([[1 + 0j]], dtype=np.complex64)
    projected = np.array([[5 + 0j]], dtype=np.complex64)
    over = ap.blend_consistent_spectra(raw, projected, blend=2.0)
    under = ap.blend_consistent_spectra(raw, projected, blend=-1.0)
    assert np.allclose(over, projected)
    assert np.allclose(under, raw)


def test_overlap_add_average_zero_chunks_returns_empty_waveform() -> None:
    chunks = np.empty((0, ap.CHUNK_SAMPLES), dtype=np.float32)
    out = ap.overlap_add_average(chunks)
    assert out.shape == (0,)
    assert out.dtype == np.float32
