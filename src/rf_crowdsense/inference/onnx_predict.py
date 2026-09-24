from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .common import build_prediction_result, prepare_sample_input
from ..deployment.onnx_export import onnx_metadata_path


def resolve_provider(requested: str, available: list[str] | tuple[str, ...]) -> str:
    requested = requested.lower()
    available_set = set(available)
    if requested == "auto":
        if "CUDAExecutionProvider" in available_set:
            return "CUDAExecutionProvider"
        if "CPUExecutionProvider" in available_set:
            return "CPUExecutionProvider"
        raise RuntimeError(f"No supported ONNX Runtime provider is available: {sorted(available_set)}")
    mapping = {"cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}
    if requested not in mapping:
        raise ValueError("provider must be one of: auto, cpu, cuda")
    selected = mapping[requested]
    if selected not in available_set:
        raise RuntimeError(
            f"{selected} was requested but is unavailable. Available providers: {sorted(available_set)}"
        )
    return selected


def load_onnx_metadata(model_path: Path) -> dict:
    path = onnx_metadata_path(model_path.resolve())
    if not path.exists():
        raise FileNotFoundError(f"ONNX metadata sidecar not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_session(model_path: Path, *, provider: str = "auto"):
    # Importing torch first lets ONNX Runtime reuse the CUDA/cuDNN DLLs shipped with
    # the PyTorch CUDA wheel on Windows when both use the same CUDA major version.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            # Provider selection below produces a clearer error if CUDA still cannot load.
            pass

    selected = resolve_provider(provider, ort.get_available_providers())
    providers = [selected]
    if selected == "CUDAExecutionProvider":
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(model_path.resolve()), providers=providers)
    return session, selected


def predict_sample_onnx(
    model_path: Path,
    sample_path: Path,
    *,
    provider: str = "auto",
) -> dict:
    model_path = model_path.resolve()
    sample_path = sample_path.resolve()
    metadata = load_onnx_metadata(model_path)
    input_array, truth = prepare_sample_input(sample_path, metadata.get("preprocessing"))
    session, selected = create_session(model_path, provider=provider)

    input_name = str(metadata.get("onnx", {}).get("input_name", "spectrogram"))
    outputs = session.run(None, {input_name: input_array})
    if len(outputs) != 2:
        raise RuntimeError(f"Expected two ONNX outputs, received {len(outputs)}")
    score = float(np.asarray(outputs[0]).reshape(-1)[0])
    class_logits = np.asarray(outputs[1], dtype=np.float32).reshape(1, -1)[0]

    return build_prediction_result(
        engine="onnxruntime",
        model_name=str(metadata["model_name"]),
        model_path=model_path,
        sample_path=sample_path,
        score=score,
        class_logits=class_logits,
        count_scale=float(metadata.get("count_scale", 1.0)),
        labels=metadata.get("activity_classes"),
        calibration=metadata.get("count_calibration"),
        device=selected,
        extra={"providers": session.get_providers()},
        ground_truth=truth,
    )
