import multiprocessing
import os
import sys
import soundfile as sf

# Configuration
SR = 48000                    # fixed sample rate of all audio files
CHUNK_SECONDS = 0.020         # 20 ms chunk length
HOP_SECONDS = 0.005           # 5 ms hop length (overlap)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
INPUT_ROOTS = [os.path.join(DATA_ROOT, "train"), os.path.join(DATA_ROOT, "test")]
OUTPUT_ROOTS = [os.path.join(DATA_ROOT, "chunk_train"), os.path.join(DATA_ROOT, "chunk_test")]
EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
NUM_WORKERS = os.cpu_count() or 1

CHUNK_SIZE = int(CHUNK_SECONDS * SR)  # 960 samples
HOP_SIZE = int(HOP_SECONDS * SR)      # 240 samples


def ensure_dir(path):
    """Create output directories if they do not exist."""
    os.makedirs(path, exist_ok=True)


def print_progress(prefix, current, total, width=40):
    """Print a simple terminal progress bar."""
    if total == 0:
        return
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"{prefix} |{bar}| {current}/{total}"
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()


def chunk_audio(waveform):
    """Return a list of overlapping chunks from one audio waveform."""
    chunks = []
    for start in range(0, len(waveform) - CHUNK_SIZE + 1, HOP_SIZE):
        chunks.append(waveform[start:start + CHUNK_SIZE])
    return chunks


def get_audio_files(input_root):
    """Collect supported audio files under the input root."""
    files = []
    for root, _, names in os.walk(input_root):
        for filename in sorted(names):
            ext = os.path.splitext(filename)[1].lower()
            if ext in EXTENSIONS:
                files.append((root, filename))
    return files


def process_file(item):
    """Process a single audio file and write its chunks."""
    root, filename, input_root, output_root = item
    rel_dir = os.path.relpath(root, input_root)
    out_dir = os.path.join(output_root, rel_dir)
    ensure_dir(out_dir)

    ext = os.path.splitext(filename)[1].lower()
    input_path = os.path.join(root, filename)
    audio, sr = sf.read(input_path)
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {sr} Hz for {input_path}")

    chunks = chunk_audio(audio)
    base = os.path.splitext(filename)[0]

    for idx, chunk in enumerate(chunks):
        chunk_name = f"{base}_ch{idx}{ext}"
        output_path = os.path.join(out_dir, chunk_name)
        sf.write(output_path, chunk, SR)

    return True


def process_input_output(input_root, output_root):
    """Walk a data folder and write chunked audio to the output folder."""
    audio_files = get_audio_files(input_root)
    total_files = len(audio_files)
    if total_files == 0:
        return

    tasks = [(root, filename, input_root, output_root) for root, filename in audio_files]
    workers = min(NUM_WORKERS, total_files)

    with multiprocessing.Pool(processes=workers) as pool:
        for count, _ in enumerate(pool.imap_unordered(process_file, tasks), start=1):
            print_progress(os.path.basename(input_root), count, total_files)

    sys.stdout.write("\n")


if __name__ == "__main__":
    for input_root, output_root in zip(INPUT_ROOTS, OUTPUT_ROOTS):
        process_input_output(input_root, output_root)

    print("Done. Chunked files written to:")
    for out in OUTPUT_ROOTS:
        print(" -", out)
