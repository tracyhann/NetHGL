"""Classical and permutation-based multiple-testing corrections."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_p_values(p_values: ArrayLike) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    return values


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Benjamini–Hochberg false-discovery-rate adjusted p-values."""

    values = _validate_p_values(p_values)
    adjusted = np.full_like(values, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) == 0:
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def holm_adjust(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm familywise-error adjusted p-values."""

    values = _validate_p_values(p_values)
    adjusted = np.full_like(values, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) == 0:
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices])]
    scaled = values[order] * np.arange(len(order), 0, -1)
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


def max_stat_fwer(
    observed_statistics: ArrayLike,
    permuted_statistics: ArrayLike,
    *,
    two_sided: bool = True,
) -> NDArray[np.float64]:
    """Max-stat permutation FWER with a finite-sample plus-one correction."""

    observed = np.asarray(observed_statistics, dtype=float)
    permutations = np.asarray(permuted_statistics, dtype=float)
    if observed.ndim != 1 or permutations.ndim != 2:
        raise ValueError("observed must be 1D and permutations must be 2D")
    if permutations.shape[1] != len(observed) or permutations.shape[0] == 0:
        raise ValueError("permutations must be shaped (iterations, observed tests)")
    if not np.isfinite(observed).all() or not np.isfinite(permutations).all():
        raise ValueError("statistics must be finite")
    if two_sided:
        reference = np.max(np.abs(permutations), axis=1)
        targets = np.abs(observed)
    else:
        reference = np.max(permutations, axis=1)
        targets = observed
    return np.asarray(
        [(1 + np.sum(reference >= target)) / (len(reference) + 1) for target in targets],
        dtype=float,
    )


def permute_participant_blocks(
    values: ArrayLike,
    participant_ids: ArrayLike,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle a participant-level variable while preserving repeated rows."""

    observations = np.asarray(values)
    participants = np.asarray(participant_ids)
    if observations.ndim != 1 or participants.ndim != 1 or observations.shape != participants.shape:
        raise ValueError("values and participant_ids must be same-length one-dimensional arrays")
    unique_participants = np.unique(participants)
    participant_values = []
    for participant in unique_participants:
        observed = np.unique(observations[participants == participant])
        if len(observed) != 1:
            raise ValueError("values must be constant within each participant block")
        participant_values.append(observed[0])
    shuffled = rng.permutation(np.asarray(participant_values))
    mapping = dict(zip(unique_participants.tolist(), shuffled.tolist(), strict=False))
    return np.asarray(
        [mapping[participant] for participant in participants], dtype=observations.dtype
    )
