import os
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_sampling_rate(file_path):
    """Worker function to get sampling rate for a single file."""
    try:
        with wave.open(file_path, 'rb') as wf:
            return wf.getframerate()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None  # Or handle differently

def check_sampling_rates_parallel(base_path, max_workers=None):
    # Collect all WAV file paths first
    wav_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    sampling_counts = {16000: 0, 48000: 0}
    other_rates = {}
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(get_sampling_rate, fp): fp for fp in wav_files}
        
        # Collect results as they complete
        for future in as_completed(futures):
            sr = future.result()
            if sr is not None:
                if sr in sampling_counts:
                    sampling_counts[sr] += 1
                else:
                    other_rates[sr] = other_rates.get(sr, 0) + 1
    
    return sampling_counts, other_rates

if __name__ == "__main__":
    base_path = "data"
    counts, others = check_sampling_rates_parallel(base_path)  # max_workers=None uses CPU count
    
    print("Sampling rate counts:")
    for sr, count in counts.items():
        print(f"{sr} Hz: {count} files")
    
    if others:
        print("Other sampling rates:")
        for sr, count in others.items():
            print(f"{sr} Hz: {count} files")