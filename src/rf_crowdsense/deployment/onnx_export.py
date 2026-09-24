from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from ..inference.common import prepare_sample_input
from ..models.pytorch_models import build_model


def onnx_metadata_path(model_path: Path) -> Path:
    return Path(str(model_path) + ".json")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(
    checkpoint_path: Path,
    sample_path: Path,
    output_path: Path,
    *,
    opset: int = 18,
    max_batch: int = 256,
    verify: bool = True,
) -> dict:
    """Export a trained RF CrowdSense PyTorch checkpoint to ONNX.

    The ONNX graph accepts a dynamic batch dimension while keeping spectrogram
    frequency/time dimensions fixed to the preprocessing shape derived from the sample.
    """
    import torch
    import onnx

    checkpoint_path = checkpoint_path.resolve()
    sample_path = sample_path.resolve()
    output_path = output_path.resolve()
    if max_batch < 2:
        raise ValueError("max_batch must be at least 2")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["model_name"])
    model = build_model(model_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preprocessing = checkpoint.get("preprocessing", {"nperseg": 256, "overlap": 0.5})
    input_array, _truth = prepare_sample_input(sample_path, preprocessing)
    # A batch >1 prevents the exporter from specializing a declared dynamic batch to 1.
    export_array = np.repeat(input_array, 2, axis=0)
    example = torch.from_numpy(export_array)
    batch_dim = torch.export.Dim("batch", min=1, max=max_batch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(output_path),
        input_names=["spectrogram"],
        output_names=["activity_score", "class_logits"],
        opset_version=opset,
        dynamo=True,
        dynamic_shapes=({0: batch_dim},),
        external_data=False,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    metadata = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "model_name": model_name,
        "preprocessing": dict(preprocessing),
        "count_scale": float(checkpoint.get("count_scale", 1.0)),
        "activity_classes": list(
            checkpoint.get("activity_classes", ["0-5", "6-20", "21-50", "51-100", "101+"])
        ),
        "count_calibration": checkpoint.get("count_calibration"),
        "onnx": {
            "opset": int(opset),
            "input_name": "spectrogram",
            "output_names": ["activity_score", "class_logits"],
            "input_shape_example": list(input_array.shape),
            "dynamic_batch": True,
            "max_batch": int(max_batch),
        },
    }

    if verify:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            torch_score, torch_logits = model(torch.from_numpy(input_array))
        ort_score, ort_logits = session.run(None, {"spectrogram": input_array})
        score_error = float(
            np.max(np.abs(torch_score.detach().cpu().numpy().astype(np.float32) - ort_score))
        )
        logits_error = float(
            np.max(np.abs(torch_logits.detach().cpu().numpy().astype(np.float32) - ort_logits))
        )
        tolerance = 1e-4
        if score_error > tolerance or logits_error > tolerance:
            raise RuntimeError(
                "ONNX verification failed: "
                f"score_max_abs_error={score_error:.6g}, logits_max_abs_error={logits_error:.6g}"
            )
        metadata["verification"] = {
            "provider": "CPUExecutionProvider",
            "score_max_abs_error": score_error,
            "logits_max_abs_error": logits_error,
            "tolerance": tolerance,
        }

    metadata_path = onnx_metadata_path(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "model": str(output_path),
        "metadata": str(metadata_path),
        "model_name": model_name,
        "opset": int(opset),
        "dynamic_batch": True,
        "max_batch": int(max_batch),
        "verified": bool(verify),
        "verification": metadata.get("verification"),
    }
