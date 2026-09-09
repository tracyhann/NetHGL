"""Transparent scaling helpers for interpretable statistical effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ScaledPredictor:
    """Scaled values and the parameters required to reproduce the transform."""

    values: NDArray[np.float64]
    mean: float
    standard_deviation: float
    mean_centered: bool


def scale_by_standard_deviation(
    values: ArrayLike,
    *,
    mean_center: bool = False,
    ddof: int = 1,
) -> ScaledPredictor:
    """Express a predictor per SD, optionally preserving its original zero.

    Without mean-centering, this computes ``x / SD(x)``. With centering, it
    computes ``(x - mean(x)) / SD(x)``. Both make a one-unit coefficient a
    one-standard-deviation effect; centering changes the model intercept and
    lower-order terms in interactions.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) <= ddof:
        raise ValueError("values must be a one-dimensional array longer than ddof")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=ddof))
    if standard_deviation <= 0:
        raise ValueError("values must have non-zero standard deviation")
    numerator = array - mean if mean_center else array
    return ScaledPredictor(
        values=np.asarray(numerator / standard_deviation, dtype=float),
        mean=mean,
        standard_deviation=standard_deviation,
        mean_centered=mean_center,
    )

