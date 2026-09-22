import pytest


def test_invalid_model_name_without_importing_torch_when_unavailable():
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("PyTorch extra is not installed")

    from rf_crowdsense.models.pytorch_models import build_model

    with pytest.raises(ValueError):
        build_model("not-a-model")
