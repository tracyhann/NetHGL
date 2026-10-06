"""ROI-level summary, autocorrelation, and autoregressive features."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import toeplitz
from scipy.stats import iqr, kurtosis, skew

from nethgl.config import GraphConfig


def _safe_standardize(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    means = values.mean(axis=axis, keepdims=True)
    scales = values.std(axis=axis, ddof=1, keepdims=True)
    scales = np.where(scales > 0, scales, 1.0)
    return (values - means) / scales


def _series_features(
    series: NDArray[np.float64],
    lags: tuple[int, ...],
    ar_order: int,
) -> NDArray[np.float64]:
    standard_deviation = float(series.std(ddof=1))
    if standard_deviation == 0:
        skewness = 0.0
        excess_kurtosis = 0.0
    else:
        skewness = float(skew(series, bias=False))
        excess_kurtosis = float(kurtosis(series, fisher=True, bias=False))

    features = [
        float(series.mean()),
        standard_deviation,
        float(np.median(series)),
        float(iqr(series)),
        float(series.min()),
        float(series.max()),
        skewness,
        excess_kurtosis,
    ]

    denominator = float(np.dot(series, series))
    correlations = []
    for lag in range(ar_order + 1):
        numerator = float(np.dot(series[: len(series) - lag], series[lag:]))
        correlations.append(numerator / denominator if denominator > 0 else 0.0)

    for lag in lags:
        features.append(correlations[lag] if lag <= ar_order else (
            float(np.dot(series[:-lag], series[lag:])) / denominator if denominator > 0 else 0.0
        ))

    if denominator > 0:
        matrix = toeplitz(np.asarray(correlations[:-1], dtype=float))
        coefficients = np.linalg.solve(
            matrix + 1e-8 * np.eye(ar_order),
            np.asarray(correlations[1:], dtype=float),
        )
    else:
        coefficients = np.zeros(ar_order, dtype=float)
    features.extend(coefficients.tolist())
    return np.asarray(features, dtype=float)


def extract_roi_timeseries_features(
    timeseries: ArrayLike,
    config: GraphConfig | None = None,
) -> NDArray[np.float32]:
    """Create the study's two-view ROI feature representation.

    One feature view is computed after standardizing each ROI over time. The
    second is computed after standardizing all ROIs within each timepoint.
    Each view contains eight distribution summaries, configured lagged
    autocorrelations, and configured Yule–Walker AR coefficients.
    """

    graph_config = config or GraphConfig()
    values = np.asarray(timeseries, dtype=float)
    if values.ndim != 2:
        raise ValueError("timeseries must be a two-dimensional ROI-by-time array")
    if not np.isfinite(values).all():
        raise ValueError("timeseries must contain only finite values for feature extraction")
    minimum_length = max(max(graph_config.lags), graph_config.ar_order) + 1
    if values.shape[1] < minimum_length:
        raise ValueError(f"timeseries must contain at least {minimum_length} timepoints")
    if values.shape[0] < 2:
        raise ValueError("at least two ROIs are required for cross-ROI standardization")

    within_roi = _safe_standardize(values.copy(), axis=1)
    within_timepoint = _safe_standardize(values.copy(), axis=0)
    rows = []
    for roi_index in range(values.shape[0]):
        first = _series_features(within_roi[roi_index], graph_config.lags, graph_config.ar_order)
        second = _series_features(
            within_timepoint[roi_index], graph_config.lags, graph_config.ar_order
        )
        rows.append(np.concatenate([first, second]))
    return np.asarray(rows, dtype=np.float32)

