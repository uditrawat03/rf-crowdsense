from __future__ import annotations

import time

import numpy as np

from ..preprocessing.spectrogram import iq_to_spectrogram


def run(iterations: int = 200) -> dict:
    rng = np.random.default_rng(42)
    iq = (rng.normal(size=4096) + 1j * rng.normal(size=4096)).astype(np.complex64)
    start = time.perf_counter()
    for _ in range(iterations):
        iq_to_spectrogram(iq, 1_000_000.0)
    elapsed = time.perf_counter() - start
    return {
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "iterations_per_second": iterations / elapsed,
        "mean_ms": 1000.0 * elapsed / iterations,
    }


def gpu_matmul(size: int = 4096, iterations: int = 20) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA is not available"}

    device = torch.device("cuda")
    a = torch.randn((size, size), device=device, dtype=torch.float16)
    b = torch.randn((size, size), device=device, dtype=torch.float16)
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "available": True,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "float16",
        "matrix_size": size,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "mean_ms": 1000.0 * elapsed / iterations,
    }
