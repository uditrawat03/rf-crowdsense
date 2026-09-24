from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np

from ..inference.common import prepare_sample_input
from ..models.pytorch_models import build_model
from ..precision import normalize_precision_list, precision_support, torch_dtype
from .inference import latency_summary


def summarize_precision_results(results: list[dict]) -> dict:
    completed = [row for row in results if row.get("status") == "ok"]
    if not completed:
        return {"fastest_precision": None, "fp32_baseline_available": False}

    fastest = min(completed, key=lambda row: float(row["mean_ms"]))
    fp32 = next((row for row in completed if row.get("precision") == "fp32"), None)
    fp32_mean = float(fp32["mean_ms"]) if fp32 is not None else None

    for row in completed:
        if fp32_mean is not None:
            row["speedup_vs_fp32"] = float(fp32_mean / float(row["mean_ms"]))
        else:
            row["speedup_vs_fp32"] = None

    return {
        "fastest_precision": str(fastest["precision"]),
        "fastest_mean_ms": float(fastest["mean_ms"]),
        "fp32_baseline_available": fp32 is not None,
    }


def benchmark_precision_modes(
    checkpoint_path: Path,
    sample_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 100,
    precisions: str | list[str] | tuple[str, ...] = "fp32,fp16,bf16",
) -> dict:
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    requested = normalize_precision_list(precisions)
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

    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    model = build_model(str(checkpoint["model_name"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target).eval()

    input_array, _truth = prepare_sample_input(sample_path, checkpoint.get("preprocessing"))
    input_array = np.repeat(input_array, batch_size, axis=0)
    x = torch.from_numpy(input_array).to(target)

    results: list[dict] = []
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for precision in requested:
        supported, reason = precision_support(torch, precision, target)
        if not supported:
            results.append(
                {
                    "precision": precision,
                    "status": "skipped",
                    "reason": reason,
                }
            )
            continue

        use_autocast = precision != "fp32"
        dtype = torch_dtype(torch, precision)

        def autocast_context():
            if not use_autocast:
                return nullcontext()
            return torch.autocast(device_type=target.type, dtype=dtype)

        if target.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(target)

        with torch.inference_mode():
            for _ in range(warmup):
                with autocast_context():
                    model(x)
            if target.type == "cuda":
                torch.cuda.synchronize(target)

            timings: list[float] = []
            for _ in range(iterations):
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                started = time.perf_counter()
                with autocast_context():
                    model(x)
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                timings.append(time.perf_counter() - started)

            with autocast_context():
                score, logits = model(x)
            if target.type == "cuda":
                torch.cuda.synchronize(target)

        outputs[precision] = (
            score.detach().float().cpu().numpy(),
            logits.detach().float().cpu().numpy(),
        )
        row = {
            "precision": precision,
            "status": "ok",
            "autocast": use_autocast,
            "torch_dtype": str(dtype),
            **latency_summary(timings, batch_size=batch_size),
        }
        if target.type == "cuda":
            row["peak_memory_allocated_mib"] = float(
                torch.cuda.max_memory_allocated(target) / (1024**2)
            )
            row["peak_memory_reserved_mib"] = float(
                torch.cuda.max_memory_reserved(target) / (1024**2)
            )
        results.append(row)

    fp32_output = outputs.get("fp32")
    if fp32_output is not None:
        fp32_score, fp32_logits = fp32_output
        for row in results:
            precision = row.get("precision")
            if row.get("status") != "ok" or precision == "fp32":
                continue
            score, logits = outputs[str(precision)]
            row["activity_score_max_abs_diff_vs_fp32"] = float(
                np.max(np.abs(score - fp32_score))
            )
            row["class_logits_max_abs_diff_vs_fp32"] = float(
                np.max(np.abs(logits - fp32_logits))
            )

    summary = summarize_precision_results(results)
    return {
        "engine": "pytorch-eager",
        "model": str(checkpoint["model_name"]),
        "device": str(target),
        "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "iterations": int(iterations),
        "requested_precisions": requested,
        "results": results,
        **summary,
        "note": (
            "Benchmark covers model inference only; IQ-to-spectrogram preprocessing is excluded. "
            "Fastest precision is a measurement for this model, batch size, and machine only."
        ),
    }
