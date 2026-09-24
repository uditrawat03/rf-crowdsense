from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from ..inference.common import prepare_sample_input
from ..inference.onnx_predict import create_session, load_onnx_metadata
from ..models.pytorch_models import build_model


def latency_summary(samples_seconds: list[float], *, batch_size: int) -> dict:
    values = np.asarray(samples_seconds, dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one latency sample is required")
    mean_seconds = float(values.mean())
    return {
        "iterations": int(values.size),
        "batch_size": int(batch_size),
        "mean_ms": mean_seconds * 1000.0,
        "p50_ms": float(np.percentile(values, 50)) * 1000.0,
        "p95_ms": float(np.percentile(values, 95)) * 1000.0,
        "samples_per_second": float(batch_size / mean_seconds),
    }


def benchmark_pytorch(
    checkpoint_path: Path,
    sample_path: Path,
    *,
    device: str = "auto",
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 100,
) -> dict:
    import torch

    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    if device == "auto":
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
        target = torch.device("cuda")
    elif device == "cpu":
        target = torch.device("cpu")
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")

    model = build_model(str(checkpoint["model_name"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target).eval()
    input_array, _truth = prepare_sample_input(sample_path, checkpoint.get("preprocessing"))
    input_array = np.repeat(input_array, batch_size, axis=0)
    x = torch.from_numpy(input_array).to(target)

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if target.type == "cuda":
            torch.cuda.synchronize()

        timings: list[float] = []
        for _ in range(iterations):
            if target.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            model(x)
            if target.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)

    return {
        "engine": "pytorch-eager",
        "device": str(target),
        "model": str(checkpoint["model_name"]),
        **latency_summary(timings, batch_size=batch_size),
    }


def benchmark_onnx(
    model_path: Path,
    sample_path: Path,
    *,
    provider: str = "auto",
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 100,
) -> dict:
    metadata = load_onnx_metadata(model_path)
    max_batch = int(metadata.get("onnx", {}).get("max_batch", 1))
    if batch_size > max_batch:
        raise ValueError(f"batch_size {batch_size} exceeds exported max_batch {max_batch}")

    input_array, _truth = prepare_sample_input(sample_path, metadata.get("preprocessing"))
    input_array = np.repeat(input_array, batch_size, axis=0)
    session, selected = create_session(model_path, provider=provider)
    input_name = str(metadata.get("onnx", {}).get("input_name", "spectrogram"))
    feeds = {input_name: input_array}

    for _ in range(warmup):
        session.run(None, feeds)

    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        session.run(None, feeds)
        timings.append(time.perf_counter() - started)

    return {
        "engine": "onnxruntime",
        "provider": selected,
        "providers": session.get_providers(),
        "model": str(metadata["model_name"]),
        **latency_summary(timings, batch_size=batch_size),
    }


def benchmark_engines(
    checkpoint_path: Path,
    onnx_model_path: Path,
    sample_path: Path,
    *,
    device: str = "auto",
    provider: str = "auto",
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 100,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    pytorch_result = benchmark_pytorch(
        checkpoint_path,
        sample_path,
        device=device,
        batch_size=batch_size,
        warmup=warmup,
        iterations=iterations,
    )
    onnx_result = benchmark_onnx(
        onnx_model_path,
        sample_path,
        provider=provider,
        batch_size=batch_size,
        warmup=warmup,
        iterations=iterations,
    )
    speedup = pytorch_result["mean_ms"] / onnx_result["mean_ms"]
    return {
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "results": [pytorch_result, onnx_result],
        "onnx_vs_pytorch_mean_latency_speedup": float(speedup),
        "note": "Benchmark covers model inference only; IQ-to-spectrogram preprocessing is excluded.",
    }
