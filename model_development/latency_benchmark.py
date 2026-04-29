import argparse
import os
import time

import torch

from core import TinyDenoiser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def rel_path(path: str) -> str:
    # Resolve relative paths from this script directory.
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def percentile(values, q):
    # Return an approximate percentile from a list.
    values = sorted(values)
    idx = int((len(values) - 1) * q)
    return values[idx]


def main():
    # Measure avg and p95 per-chunk inference latency.
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyDenoiser().to(device).eval()
    if args.checkpoint:
        ckpt = torch.load(rel_path(args.checkpoint), map_location=device)
        model.load_state_dict(ckpt["model"])

    x = torch.randn(1, 1, 960, device=device)  # 20ms @ 48kHz

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    avg = sum(times_ms) / len(times_ms)
    p95 = percentile(times_ms, 0.95)
    print(f"device={device}")
    print(f"avg_latency_ms={avg:.4f}")
    print(f"p95_latency_ms={p95:.4f}")


if __name__ == "__main__":
    main()
