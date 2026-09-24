import numpy as np
import pytest

from rf_crowdsense.calibration import (
    count_interval,
    empirical_interval_coverage,
    fit_count_interval,
)


def test_fit_count_interval_uses_finite_sample_conformal_rank():
    truth = np.array([0.0, 10.0, 20.0, 30.0])
    prediction = np.array([0.0, 9.0, 18.0, 27.0])

    calibration = fit_count_interval(truth, prediction, coverage=0.75)

    # Residuals are [0, 1, 2, 3]. ceil((4 + 1) * .75) = 4.
    assert calibration.radius == 3.0
    assert calibration.nominal_coverage == 0.75
    assert calibration.calibration_size == 4


def test_count_interval_clips_to_physical_bounds():
    calibration = {
        "radius": 8.0,
        "nominal_coverage": 0.9,
        "method": "split_conformal_absolute_residual",
        "calibration_size": 20,
    }

    assert count_interval(3.0, calibration, lower_bound=0.0) == (0.0, 11.0)
    assert count_interval(118.0, calibration, lower_bound=0.0, upper_bound=120.0) == (
        110.0,
        120.0,
    )


def test_empirical_interval_coverage():
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    prediction = np.array([11.0, 23.0, 31.0, 50.0])
    calibration = {
        "radius": 3.0,
        "nominal_coverage": 0.9,
        "method": "split_conformal_absolute_residual",
        "calibration_size": 10,
    }

    assert empirical_interval_coverage(truth, prediction, calibration) == 0.75


def test_calibration_validation():
    with pytest.raises(ValueError):
        fit_count_interval(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        fit_count_interval(np.array([1.0]), np.array([1.0]), coverage=1.0)
    with pytest.raises(ValueError):
        fit_count_interval(np.array([1.0, 2.0]), np.array([1.0]))


def test_small_calibration_set_uses_unbounded_radius_when_needed():
    calibration = fit_count_interval(
        np.array([10.0]),
        np.array([12.0]),
        coverage=0.90,
    )

    assert np.isinf(calibration.radius)
    assert count_interval(50.0, calibration, lower_bound=0.0, upper_bound=120.0) == (
        0.0,
        120.0,
    )
