from __future__ import annotations

from pathlib import Path

import numpy as np

from ..calibration import count_interval
from ..models.pytorch_models import build_model
from ..preprocessing.spectrogram import iq_to_spectrogram


def _resolve_device(torch, requested: str):
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def predict_sample(
    checkpoint_path: Path,
    sample_path: Path,
    *,
    device: str = "auto",
) -> dict:
    import torch

    checkpoint_path = checkpoint_path.resolve()
    sample_path = sample_path.resolve()
    target_device = _resolve_device(torch, device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

    model_name = str(checkpoint["model_name"])
    model = build_model(model_name).to(target_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preprocessing = checkpoint.get("preprocessing", {"nperseg": 256, "overlap": 0.5})
    payload = np.load(sample_path)
    iq = payload["iq"]
    sample_rate = float(payload["sample_rate"])
    spec = iq_to_spectrogram(
        iq,
        sample_rate,
        nperseg=int(preprocessing.get("nperseg", 256)),
        overlap=float(preprocessing.get("overlap", 0.5)),
    )
    x = torch.from_numpy(spec[None, None, ...]).to(target_device)

    with torch.inference_mode():
        predicted_score, class_logits = model(x)
        score = float(predicted_score.squeeze().detach().cpu())
        probabilities = torch.softmax(class_logits, dim=1).squeeze(0).detach().cpu().numpy()

    count_scale = float(checkpoint.get("count_scale", 1.0))
    estimated_count = max(0.0, score * count_scale)
    labels = checkpoint.get("activity_classes", ["0-5", "6-20", "21-50", "51-100", "101+"])
    class_index = int(np.argmax(probabilities))

    result = {
        "checkpoint": str(checkpoint_path),
        "sample": str(sample_path),
        "device": str(target_device),
        "model": model_name,
        "activity_score": score,
        "estimated_active_transmitters": estimated_count,
        "activity_class": {
            "index": class_index,
            "label": str(labels[class_index]),
            "probability": float(probabilities[class_index]),
            "probability_note": "Softmax probability is not separately calibrated in this milestone.",
        },
    }

    calibration = checkpoint.get("count_calibration")
    if calibration is not None:
        lower, upper = count_interval(
            estimated_count,
            calibration,
            lower_bound=0.0,
            upper_bound=count_scale,
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

    if "active_devices" in payload:
        result["ground_truth_active_transmitters"] = int(payload["active_devices"])
    return result
