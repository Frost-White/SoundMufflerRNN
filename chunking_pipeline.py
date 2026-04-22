import math
import wave
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


SAMPLE_RATE = 48000
CHUNK_SIZE = 960
TRAIN_HOP = 480
INFER_HOP = 960
TOO_SHORT_MARKER = ".too_short"


@dataclass(frozen=True)
class PairJob:
    noisy_path: Path
    clean_path: Path
    out_dir: Path
    chunk_size: int
    hop_size: int
    sample_rate: int


def _read_wav_mono_int16(path: Path, expected_sr: int) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if channels != 1:
        raise ValueError(f"{path} mono degil (channels={channels})")
    if sample_width != 2:
        raise ValueError(f"{path} 16-bit PCM degil (sample_width={sample_width})")
    if frame_rate != expected_sr:
        raise ValueError(f"{path} sample rate {frame_rate}, beklenen {expected_sr}")

    return np.frombuffer(raw, dtype=np.int16)


def _write_wav_mono_int16(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.Wave_write(str(path)) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.asarray(audio, dtype=np.int16).tobytes())


def _chunk_count(total_len: int, chunk_size: int, hop_size: int) -> int:
    if total_len < chunk_size:
        return 0
    return 1 + (total_len - chunk_size) // hop_size


def _iter_chunks(audio: np.ndarray, chunk_size: int, hop_size: int) -> Iterable[np.ndarray]:
    count = _chunk_count(len(audio), chunk_size, hop_size)
    for i in range(count):
        start = i * hop_size
        end = start + chunk_size
        yield audio[start:end]


def _already_processed(path: Path) -> bool:
    if not path.exists():
        return False
    if (path / TOO_SHORT_MARKER).exists():
        return True
    return any(path.iterdir())


def _job_output_folder_name(noisy_path: Path) -> str:
    return noisy_path.stem


def _collect_jobs(
    split_root: Path,
    out_root: Path,
    chunk_size: int,
    hop_size: int,
    sample_rate: int,
) -> List[PairJob]:
    noisy_root = split_root / "noisy"
    clean_root = split_root / "clean"
    if not noisy_root.exists() or not clean_root.exists():
        raise FileNotFoundError(f"Beklenen klasorler yok: {noisy_root} veya {clean_root}")

    noisy_files = sorted(noisy_root.rglob("*.wav"))
    jobs: List[PairJob] = []

    for noisy_path in noisy_files:
        rel = noisy_path.relative_to(noisy_root)
        clean_path = clean_root / rel
        if not clean_path.exists():
            raise FileNotFoundError(f"Eslesmis clean dosyasi yok: {clean_path}")

        out_dir = out_root / _job_output_folder_name(noisy_path)
        jobs.append(
            PairJob(
                noisy_path=noisy_path,
                clean_path=clean_path,
                out_dir=out_dir,
                chunk_size=chunk_size,
                hop_size=hop_size,
                sample_rate=sample_rate,
            )
        )
    return jobs


def _process_single_pair(job: PairJob) -> Tuple[str, int, str]:
    if _already_processed(job.out_dir):
        return (str(job.noisy_path), 0, "skipped")

    noisy = _read_wav_mono_int16(job.noisy_path, expected_sr=job.sample_rate)
    clean = _read_wav_mono_int16(job.clean_path, expected_sr=job.sample_rate)
    usable_len = min(len(noisy), len(clean))
    noisy = noisy[:usable_len]
    clean = clean[:usable_len]

    n_chunks = _chunk_count(usable_len, job.chunk_size, job.hop_size)
    if n_chunks == 0:
        job.out_dir.mkdir(parents=True, exist_ok=True)
        (job.out_dir / TOO_SHORT_MARKER).touch()
        return (str(job.noisy_path), 0, "too_short")

    job.out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (n_chunk, c_chunk) in enumerate(
        zip(
            _iter_chunks(noisy, job.chunk_size, job.hop_size),
            _iter_chunks(clean, job.chunk_size, job.hop_size),
        )
    ):
        noisy_name = job.out_dir / f"noisy_{idx:06d}.wav"
        clean_name = job.out_dir / f"clean_{idx:06d}.wav"
        _write_wav_mono_int16(noisy_name, n_chunk, sr=job.sample_rate)
        _write_wav_mono_int16(clean_name, c_chunk, sr=job.sample_rate)

    return (str(job.noisy_path), n_chunks, "ok")


def _print_progress(done: int, total: int) -> None:
    width = 30
    ratio = done / total if total else 1.0
    filled = int(math.floor(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100.0
    print(f"\r[{bar}] {percent:6.2f}% ({done}/{total})", end="", flush=True)
    if done == total:
        print()


def _run_split(
    split_root: Path,
    out_root: Path,
    hop_size: int,
    chunk_size: int = CHUNK_SIZE,
    sample_rate: int = SAMPLE_RATE,
    max_workers: int | None = None,
) -> None:
    jobs = _collect_jobs(
        split_root=split_root,
        out_root=out_root,
        chunk_size=chunk_size,
        hop_size=hop_size,
        sample_rate=sample_rate,
    )

    if not jobs:
        print(f"{split_root} icin islenecek wav bulunamadi.")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    total = len(jobs)
    done = 0
    skipped = 0
    too_short = 0
    total_chunks = 0
    failures: List[Tuple[str, str]] = []

    print(f"\nIsleniyor: {split_root} -> {out_root} (hop={hop_size})")
    _print_progress(done, total)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_process_single_pair, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            done += 1
            try:
                _, count, status = future.result()
                total_chunks += count
                if status == "skipped":
                    skipped += 1
                elif status == "too_short":
                    too_short += 1
            except (ValueError, wave.Error, OSError) as exc:
                failures.append((str(job.noisy_path), str(exc)))
            _print_progress(done, total)

    print(
        f"Bitti: toplam={total}, chunk={total_chunks}, skipped={skipped}, too_short={too_short}, errors={len(failures)}"
    )
    if failures:
        print("Hata listesi:")
        for file_path, err in failures:
            print(f"- {file_path}: {err}")


def process_and_save(
    data_root: str | Path = "data",
    max_workers: int | None = None,
) -> None:
    """
    data/train/noisy-clean ve data/test/noisy-clean ciftlerini chunk'layip diske kaydeder.

    - Train: chunk=960, hop=480 (overlap)
    - Test/Inference: chunk=960, hop=960 (non-overlap)
    """
    data_root = Path(data_root)
    train_root = data_root / "train"
    test_root = data_root / "test"
    chunk_train_root = data_root / "chunk_train"
    chunk_test_root = data_root / "chunk_test"

    if train_root.exists():
        _run_split(
            split_root=train_root,
            out_root=chunk_train_root,
            hop_size=TRAIN_HOP,
            chunk_size=CHUNK_SIZE,
            sample_rate=SAMPLE_RATE,
            max_workers=max_workers,
        )
    else:
        print(f"Train klasoru yok, atlandi: {train_root}")

    if test_root.exists():
        _run_split(
            split_root=test_root,
            out_root=chunk_test_root,
            hop_size=INFER_HOP,
            chunk_size=CHUNK_SIZE,
            sample_rate=SAMPLE_RATE,
            max_workers=max_workers,
        )
    else:
        print(f"Test klasoru yok, atlandi: {test_root}")


if __name__ == "__main__":
    # Kisa kullanim:
    # max_workers=None -> sistemdeki CPU sayisini kullanir.
    process_and_save(data_root="data", max_workers=None)
