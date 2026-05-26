from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.inference.audio_pipeline import (
    SR,
    analysis_stft_chunks,
    blend_consistent_spectra,
    chunk_waveform,
    overlap_add_average,
    project_to_stft_consistency,
    synthesis_istft_chunks,
)
from app.inference.model import GRUChunkDenoiser

_LOG_EPS = 1e-8
_BOUNDARY_PAD = 240


class InferenceError(Exception):
    pass


def _default_weights_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "best_weights.pt"


def _load_model(weights_path: Path) -> GRUChunkDenoiser:
    model = GRUChunkDenoiser(hidden_dim=128, num_layers=3).to(torch.device("cpu"))
    obj = torch.load(str(weights_path), map_location=torch.device("cpu"))
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_model() -> GRUChunkDenoiser:
    path = Path(os.getenv("SOUNDMUFFLER_WEIGHTS_PATH", str(_default_weights_path())))
    if not path.is_file():
        raise FileNotFoundError(f"Weights not found: {path}")
    return _load_model(path)


def _resample_linear(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav.astype(np.float32, copy=False)
    if wav.size == 0:
        return np.zeros(0, dtype=np.float32)
    dst_len = max(1, int(round(len(wav) * float(dst_sr) / float(src_sr))))
    src_idx = np.arange(len(wav), dtype=np.float64)
    dst_idx = np.linspace(0, len(wav) - 1, num=dst_len, dtype=np.float64)
    out = np.interp(dst_idx, src_idx, wav.astype(np.float64, copy=False))
    return out.astype(np.float32, copy=False)


def _enhance_waveform(noisy_wav: np.ndarray) -> np.ndarray:
    work_wav = noisy_wav.astype(np.float32, copy=False)
    work_wav = np.pad(work_wav, (_BOUNDARY_PAD, _BOUNDARY_PAD))
    chunks = chunk_waveform(work_wav, pad_end=True)
    if chunks.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    noisy_spec = analysis_stft_chunks(chunks)
    mag = np.abs(noisy_spec)
    x_np = np.log(mag + _LOG_EPS).astype(np.float32, copy=False)
    x = torch.from_numpy(x_np).unsqueeze(0).to(torch.device("cpu"))
    lengths = torch.tensor([x_np.shape[0]], dtype=torch.long)
    with torch.no_grad():
        mask = get_model()(x, lengths).squeeze(0).cpu().numpy()

    enhanced_spec_raw = mask * noisy_spec
    enhanced_spec = blend_consistent_spectra(
        enhanced_spec_raw,
        project_to_stft_consistency(enhanced_spec_raw),
        blend=1.0,
    )
    chunk_out = synthesis_istft_chunks(enhanced_spec)
    wav_out = overlap_add_average(chunk_out, min_weight=0.0)
    if len(wav_out) > len(work_wav):
        wav_out = wav_out[: len(work_wav)]
    start = _BOUNDARY_PAD
    stop = start + len(noisy_wav)
    return wav_out[start:stop]


def enhance_audio_bytes(file_bytes: bytes) -> bytes:
    try:
        with sf.SoundFile(io.BytesIO(file_bytes)) as f:
            sr = int(f.samplerate)
            data = f.read(dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError("Unsupported or invalid audio file.") from exc

    if data.ndim > 1:
        data = data.mean(axis=1)

    input_wav = np.asarray(data, dtype=np.float32)
    model_input = _resample_linear(input_wav, sr, SR) if sr != SR else input_wav
    enhanced_model_sr = _enhance_waveform(model_input)
    enhanced = _resample_linear(enhanced_model_sr, SR, sr) if sr != SR else enhanced_model_sr

    buf = io.BytesIO()
    sf.write(buf, enhanced, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
