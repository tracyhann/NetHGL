"""Atlas-agnostic parcellation of grayordinate or voxel time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ParcellatedTimeseries:
    """ROI×time signals, finite-sample counts, and ordered parcel labels."""

    timeseries: NDArray[np.float32]
    valid_counts: NDArray[np.int32]
    parcel_ids: NDArray[np.int64]


def parcellate_timeseries(
    samples_by_grayordinate: ArrayLike,
    parcel_labels: ArrayLike,
    *,
    background_label: int = 0,
) -> ParcellatedTimeseries:
    """Average aligned samples within each non-background atlas parcel.

    The first input is shaped ``(timepoints, grayordinates)``. Labels must
    already be aligned to the same grayordinate axis; atlas-specific alignment
    belongs in the user's governed neuroimaging workflow.
    """

    values = np.asarray(samples_by_grayordinate, dtype=float)
    labels = np.asarray(parcel_labels)
    if values.ndim != 2:
        raise ValueError("samples_by_grayordinate must be a two-dimensional array")
    if labels.ndim != 1 or values.shape[1] != len(labels):
        raise ValueError("parcel_labels must align with the grayordinate axis")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("parcel_labels must contain integer labels")

    parcel_ids = np.asarray(
        [label for label in np.unique(labels) if label != background_label], dtype=np.int64
    )
    if len(parcel_ids) == 0:
        raise ValueError("parcel_labels contain no non-background parcels")
    timeseries_rows = []
    count_rows = []
    for parcel_id in parcel_ids:
        parcel_values = values[:, labels == parcel_id]
        finite = np.isfinite(parcel_values)
        counts = finite.sum(axis=1)
        sums = np.where(finite, parcel_values, 0.0).sum(axis=1)
        means = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan, dtype=float),
            where=counts > 0,
        )
        timeseries_rows.append(means)
        count_rows.append(counts)
    return ParcellatedTimeseries(
        timeseries=np.asarray(timeseries_rows, dtype=np.float32),
        valid_counts=np.asarray(count_rows, dtype=np.int32),
        parcel_ids=parcel_ids,
    )

