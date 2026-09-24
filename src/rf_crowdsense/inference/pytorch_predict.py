from __future__ import annotations

from pathlib import Path

from .common import build_prediction_result, prepare_sample_input
from ..models.pytorch_models import build_model


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

    input_array, truth = prepare_sample_input(sample_path, checkpoint.get("preprocessing"))
    x = torch.from_numpy(input_array).to(target_device)

    with torch.inference_mode():
        predicted_score, class_logits = model(x)
        score = float(predicted_score.squeeze().detach().cpu())
        logits = class_logits.squeeze(0).detach().cpu().numpy()

    return build_prediction_result(
        engine="pytorch",
        model_name=model_name,
        model_path=checkpoint_path,
        sample_path=sample_path,
        score=score,
        class_logits=logits,
        count_scale=float(checkpoint.get("count_scale", 1.0)),
        labels=checkpoint.get("activity_classes"),
        calibration=checkpoint.get("count_calibration"),
        device=str(target_device),
        ground_truth=truth,
    )
