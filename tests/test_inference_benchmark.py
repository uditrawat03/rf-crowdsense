import numpy as np
import pytest

from rf_crowdsense.benchmark.inference import latency_summary


def test_latency_summary_reports_latency_and_throughput():
    result = latency_summary([0.01, 0.02, 0.03], batch_size=2)
    assert result["iterations"] == 3
    assert result["batch_size"] == 2
    assert result["mean_ms"] == pytest.approx(20.0)
    assert result["p50_ms"] == pytest.approx(20.0)
    assert result["samples_per_second"] == pytest.approx(100.0)
    assert np.isfinite(result["p95_ms"])


def test_latency_summary_requires_values():
    with pytest.raises(ValueError):
        latency_summary([], batch_size=1)
