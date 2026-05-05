import argparse
import os

import h5py
import numpy as np
import soundfile as sf


EXPECTED_SR = 48000
PROGRESS_EVERY = 10000


def read_wav_float32(path):
    data, sr = sf.read(path, dtype="float32")
    if sr != EXPECTED_SR:
        raise RuntimeError(f"Sample rate mismatch in {path}: expected {EXPECTED_SR}, got {sr}")
    if getattr(data, "ndim", 1) != 1:
        raise RuntimeError(f"Expected mono audio in {path}, got shape {data.shape}")
    return data.astype(np.float32, copy=False)


def list_wavs(folder):
    return {name for name in os.listdir(folder) if name.lower().endswith(".wav")}


def convert_split(split_name, noisy_dir, clean_dir, out_noisy_path, out_clean_path):
    noisy_files = list_wavs(noisy_dir)
    clean_files = list_wavs(clean_dir)

    common = sorted(noisy_files & clean_files)
    noisy_only = sorted(noisy_files - clean_files)
    clean_only = sorted(clean_files - noisy_files)

    skipped_unmatched = len(noisy_only) + len(clean_only)
    if skipped_unmatched:
        print(
            f"[{split_name}] Unmatched files skipped: {skipped_unmatched} "
            f"(noisy-only={len(noisy_only)}, clean-only={len(clean_only)})"
        )

    vlen_dtype = h5py.vlen_dtype(np.float32)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_noisy_path, "w") as h5_noisy, h5py.File(out_clean_path, "w") as h5_clean:
        noisy_ds = h5_noisy.create_dataset("audio", shape=(0,), maxshape=(None,), dtype=vlen_dtype)
        clean_ds = h5_clean.create_dataset("audio", shape=(0,), maxshape=(None,), dtype=vlen_dtype)
        noisy_name_ds = h5_noisy.create_dataset("filenames", shape=(0,), maxshape=(None,), dtype=str_dtype)
        clean_name_ds = h5_clean.create_dataset("filenames", shape=(0,), maxshape=(None,), dtype=str_dtype)

        written = 0
        corrupt_skipped = 0

        for filename in common:
            noisy_path = os.path.join(noisy_dir, filename)
            clean_path = os.path.join(clean_dir, filename)

            try:
                noisy_audio = read_wav_float32(noisy_path)
                clean_audio = read_wav_float32(clean_path)
            except RuntimeError:
                raise
            except Exception as exc:
                corrupt_skipped += 1
                print(f"[{split_name}] Corrupt/unreadable pair skipped: {filename} ({exc})")
                continue

            noisy_ds.resize((written + 1,))
            clean_ds.resize((written + 1,))
            noisy_name_ds.resize((written + 1,))
            clean_name_ds.resize((written + 1,))

            noisy_ds[written] = noisy_audio
            clean_ds[written] = clean_audio
            noisy_name_ds[written] = filename
            clean_name_ds[written] = filename
            written += 1

            if written % PROGRESS_EVERY == 0:
                print(f"[{split_name}] Written {written} chunks...")

    print(
        f"[{split_name}] Done. Written: {written}, "
        f"skipped_unmatched: {skipped_unmatched}, skipped_corrupt: {corrupt_skipped}"
    )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.normpath(os.path.join(base_dir, "..", "data"))
    default_output_dir = os.path.join(default_data_root, "h5")

    parser = argparse.ArgumentParser(description="Convert chunked wav pairs into HDF5 files.")
    parser.add_argument(
        "--data_root",
        default=default_data_root,
        help="Full path to data/ folder. Defaults to ../data relative to this script.",
    )
    parser.add_argument(
        "--output_dir",
        default=default_output_dir,
        help="Directory to write HDF5 files. Defaults to <data_root>/h5.",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using data_root: {data_root}")
    print(f"Using output_dir: {output_dir}")

    train_noisy_dir = os.path.join(data_root, "chunk_train", "noisy_trainset_56spk_wav")
    train_clean_dir = os.path.join(data_root, "chunk_train", "clean_trainset_56spk_wav")
    test_noisy_dir = os.path.join(data_root, "chunk_test", "noisy_testset_wav")
    test_clean_dir = os.path.join(data_root, "chunk_test", "clean_testset_wav")

    required_dirs = [train_noisy_dir, train_clean_dir, test_noisy_dir, test_clean_dir]
    for folder in required_dirs:
        if not os.path.isdir(folder):
            raise RuntimeError(f"Missing required folder: {folder}")

    convert_split(
        split_name="train",
        noisy_dir=train_noisy_dir,
        clean_dir=train_clean_dir,
        out_noisy_path=os.path.join(output_dir, "train_noisy.h5"),
        out_clean_path=os.path.join(output_dir, "train_clean.h5"),
    )
    convert_split(
        split_name="test",
        noisy_dir=test_noisy_dir,
        clean_dir=test_clean_dir,
        out_noisy_path=os.path.join(output_dir, "test_noisy.h5"),
        out_clean_path=os.path.join(output_dir, "test_clean.h5"),
    )


if __name__ == "__main__":
    main()
