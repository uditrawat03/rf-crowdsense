from __future__ import annotations

from pathlib import Path

import numpy as np

from ..calibration import count_interval
from ..preprocessing.spectrogram import iq_to_spectrogram

DEFAULT_ACTIVITY_CLASSES = ["0-5", "6-20", "21-50", "51-100", "101+"]


def prepare_sample_input(sample_path: Path, preprocessing: dict | None = None) -> tuple[np.ndarray, dict]:
    """Load one synthetic/authorized IQ sample and return NCHW float32 spectrogram input."""
    preprocessing = preprocessing or {"nperseg": 256, "overlap": 0.5}
    payload = np.load(sample_path.resolve())
    spec = iq_to_spectrogram(
        payload["iq"],
        float(payload["sample_rate"]),
        nperseg=int(preprocessing.get("nperseg", 256)),
        overlap=float(preprocessing.get("overlap", 0.5)),
    )
    x = np.ascontiguousarray(spec[None, None, ...], dtype=np.float32)
    truth = {}
    if "active_devices" in payload:
        truth["ground_truth_active_transmitters"] = int(payload["active_devices"])
    return x, truth


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return (exp_values / np.sum(exp_values)).astype(np.float64)


def build_prediction_result(
    *,
    engine: str,
    model_name: str,
    sample_path: Path,
    score: float,
    class_logits: np.ndarray,
    count_scale: float,
    labels: list[str] | tuple[str, ...] | None = None,
    calibration: dict | None = None,
    device: str | None = None,
    model_path: Path | None = None,
    extra: dict | None = None,
    ground_truth: dict | None = None,
) -> dict:
    labels = list(labels or DEFAULT_ACTIVITY_CLASSES)
    probabilities = softmax(class_logits)
    class_index = int(np.argmax(probabilities))
    estimated_count = max(0.0, float(score) * float(count_scale))

    result = {
        "engine": engine,
        "sample": str(sample_path.resolve()),
        "model": model_name,
        "activity_score": float(score),
        "estimated_active_transmitters": estimated_count,
        "activity_class": {
            "index": class_index,
            "label": str(labels[class_index]),
            "probability": float(probabilities[class_index]),
            "probability_note": "Softmax probability is not separately calibrated in this milestone.",
        },
    }
    if model_path is not None:
        result["model_path"] = str(model_path.resolve())
    if device is not None:
        result["device"] = device

    if calibration is not None:
        lower, upper = count_interval(
            estimated_count,
            calibration,
            lower_bound=0.0,
            upper_bound=float(count_scale),
        )
        result["count_interval"] = {
            "lower": lower,
            "upper": upper,
            "nominal_coverage": float(calibration["nominal_coverage"]),
            "method": str(calibration["method"]),
            "calibration_size": int(calibration["calibration_size"]),
            "interpretation": (
                "Dataset-level split-conformal coverage target; not a per-sample probability."
            ),
        }
    else:
        result["count_interval"] = None

    if ground_truth:
        result.update(ground_truth)
    if extra:
        result.update(extra)
    return result
