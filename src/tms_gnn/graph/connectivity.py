"""Functional-connectivity estimation from parcellated time series."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pairwise_pearson_connectivity(
    timeseries: ArrayLike,
    min_overlap: int = 30,
) -> NDArray[np.float32]:
    """Estimate a symmetric ROI×ROI Pearson matrix with pairwise deletion.

    Parameters
    ----------
    timeseries:
        Array shaped ``(n_rois, n_timepoints)``. Missing samples may be NaN.
    min_overlap:
        Minimum number of finite samples required for an off-diagonal pair.

    Returns
    -------
    numpy.ndarray
        Float32 correlation matrix. Insufficient-overlap and constant pairs are
        assigned zero; the diagonal is one.
    """

    values = np.asarray(timeseries, dtype=float)
    if values.ndim != 2:
        raise ValueError("timeseries must be a two-dimensional ROI-by-time array")
    if values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("timeseries must contain at least one ROI and two timepoints")
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least 2")

    n_rois = values.shape[0]
    result = np.zeros((n_rois, n_rois), dtype=np.float32)
    for left in range(n_rois):
        for right in range(left + 1, n_rois):
            finite = np.isfinite(values[left]) & np.isfinite(values[right])
            if int(finite.sum()) < min_overlap:
                continue
            x = values[left, finite]
            y = values[right, finite]
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
            if denominator > 0:
                correlation = float(np.dot(x_centered, y_centered) / denominator)
                result[left, right] = result[right, left] = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result

