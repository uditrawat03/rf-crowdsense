from pathlib import Path

import numpy as np

from rf_crowdsense.inference.common import build_prediction_result, softmax


def test_softmax_is_normalized_and_stable():
    probs = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.isclose(probs.sum(), 1.0)
    assert int(np.argmax(probs)) == 1


def test_build_prediction_result_adds_count_interval():
    calibration = {
        "radius": 5.0,
        "nominal_coverage": 0.9,
        "method": "split_conformal_absolute_residual",
        "calibration_size": 100,
    }
    result = build_prediction_result(
        engine="test",
        model_name="cnn",
        sample_path=Path("sample.npz"),
        score=0.25,
        class_logits=np.array([0.0, 0.1, 2.0, 0.2, -1.0]),
        count_scale=120.0,
        calibration=calibration,
    )

    assert result["estimated_active_transmitters"] == 30.0
    assert result["activity_class"]["index"] == 2
    assert result["count_interval"]["lower"] == 25.0
    assert result["count_interval"]["upper"] == 35.0
