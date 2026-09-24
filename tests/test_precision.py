import pytest

from rf_crowdsense.benchmark.precision import summarize_precision_results
from rf_crowdsense.precision import (
    grad_scaler_enabled,
    normalize_amp_dtype,
    normalize_precision,
    normalize_precision_list,
)


def test_precision_aliases_and_list_deduplication():
    assert normalize_precision("float32") == "fp32"
    assert normalize_precision("half") == "fp16"
    assert normalize_precision("bfloat16") == "bf16"
    assert normalize_precision_list("fp32, half, bf16, fp32") == ["fp32", "fp16", "bf16"]


def test_precision_validation():
    with pytest.raises(ValueError):
        normalize_precision("fp8")
    with pytest.raises(ValueError):
        normalize_precision_list("")
    with pytest.raises(ValueError):
        normalize_amp_dtype("fp32")


def test_grad_scaler_is_only_required_for_fp16_amp():
    assert grad_scaler_enabled(amp_enabled=True, amp_dtype="fp16") is True
    assert grad_scaler_enabled(amp_enabled=True, amp_dtype="bf16") is False
    assert grad_scaler_enabled(amp_enabled=False, amp_dtype="fp16") is False


def test_precision_summary_sets_speedups_and_fastest_mode():
    rows = [
        {"precision": "fp32", "status": "ok", "mean_ms": 4.0},
        {"precision": "fp16", "status": "ok", "mean_ms": 2.0},
        {"precision": "bf16", "status": "skipped", "reason": "unsupported"},
    ]
    summary = summarize_precision_results(rows)

    assert summary["fastest_precision"] == "fp16"
    assert summary["fp32_baseline_available"] is True
    assert rows[0]["speedup_vs_fp32"] == pytest.approx(1.0)
    assert rows[1]["speedup_vs_fp32"] == pytest.approx(2.0)
    assert "speedup_vs_fp32" not in rows[2]


def test_precision_summary_without_completed_results():
    rows = [{"precision": "bf16", "status": "skipped", "reason": "unsupported"}]
    assert summarize_precision_results(rows) == {
        "fastest_precision": None,
        "fp32_baseline_available": False,
    }
