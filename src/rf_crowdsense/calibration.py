from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class CountCalibration:
    method: str
    nominal_coverage: float
    radius: float
    calibration_size: int

    def to_dict(self) -> dict:
        return asdict(self)


def fit_count_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    coverage: float = 0.90,
) -> CountCalibration:
    """Fit an absolute-residual split-conformal interval for aggregate counts.

    The returned interval has a dataset-level nominal coverage target. It is not a
    per-example probability that a particular interval contains the true count.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")

    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size == 0:
        raise ValueError("at least one calibration example is required")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("calibration values must be finite")

    residuals = np.sort(np.abs(truth - prediction))
    # Standard finite-sample split-conformal rank:
    # ceil((n + 1) * (1 - alpha)), clipped to the available calibration scores.
    rank = math.ceil((truth.size + 1) * coverage)
    # With a very small calibration set the requested finite-sample coverage may
    # be unattainable with a finite residual quantile. In that case, use an
    # infinite radius; physical count bounds can still clip the final interval.
    radius = float("inf") if rank > truth.size else float(residuals[rank - 1])
    return CountCalibration(
        method="split_conformal_absolute_residual",
        nominal_coverage=float(coverage),
        radius=radius,
        calibration_size=int(truth.size),
    )


def count_interval(
    prediction: float,
    calibration: CountCalibration | dict,
    *,
    lower_bound: float = 0.0,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    if isinstance(calibration, dict):
        radius = float(calibration["radius"])
    else:
        radius = calibration.radius

    lower = max(float(lower_bound), float(prediction) - radius)
    upper = float(prediction) + radius
    if upper_bound is not None:
        upper = min(float(upper_bound), upper)
    return lower, max(lower, upper)


def empirical_interval_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calibration: CountCalibration | dict,
) -> float:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size == 0:
        raise ValueError("at least one example is required")

    radius = float(calibration["radius"] if isinstance(calibration, dict) else calibration.radius)
    return float(np.mean(np.abs(truth - prediction) <= radius))
