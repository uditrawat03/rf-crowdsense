from pathlib import Path

import pytest

from rf_crowdsense.deployment.onnx_export import onnx_metadata_path
from rf_crowdsense.inference.onnx_predict import resolve_provider


def test_onnx_metadata_path_is_sidecar():
    assert onnx_metadata_path(Path("model.onnx")) == Path("model.onnx.json")


def test_provider_auto_prefers_cuda():
    available = ["CPUExecutionProvider", "CUDAExecutionProvider"]
    assert resolve_provider("auto", available) == "CUDAExecutionProvider"


def test_provider_auto_falls_back_to_cpu():
    assert resolve_provider("auto", ["CPUExecutionProvider"]) == "CPUExecutionProvider"


def test_provider_rejects_missing_cuda():
    with pytest.raises(RuntimeError):
        resolve_provider("cuda", ["CPUExecutionProvider"])


def test_provider_rejects_unknown_value():
    with pytest.raises(ValueError):
        resolve_provider("directml", ["CPUExecutionProvider"])
