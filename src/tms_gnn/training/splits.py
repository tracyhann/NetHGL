"""Participant-level data splitting for repeated-graph datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ParticipantSplit:
    """Row indices and unique participant identifiers for three partitions."""

    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    train_participants: NDArray
    validation_participants: NDArray
    test_participants: NDArray


def _participant_strata(participants: np.ndarray, stratify: np.ndarray) -> np.ndarray:
    values = []
    for participant in np.unique(participants):
        observed = np.unique(stratify[participants == participant])
        if len(observed) != 1:
            raise ValueError("stratification labels must be constant within each participant")
        values.append(observed[0])
    return np.asarray(values)


def participant_level_split(
    participant_ids: ArrayLike,
    *,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 0,
    stratify: ArrayLike | None = None,
) -> ParticipantSplit:
    """Split graph rows without placing one participant in multiple sets.

    ``test_size`` and ``validation_size`` are fractions of all unique
    participants. ``stratify`` is optional participant-level information
    repeated on graph rows; it is not assumed to be the graph outcome.
    """

    participants = np.asarray(participant_ids)
    if participants.ndim != 1 or len(participants) == 0:
        raise ValueError("participant_ids must be a non-empty one-dimensional array")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must lie in (0, 1)")
    if test_size + validation_size >= 1:
        raise ValueError("test_size plus validation_size must be less than 1")

    unique_participants = np.unique(participants)
    participant_strata = None
    if stratify is not None:
        strata = np.asarray(stratify)
        if strata.shape != participants.shape:
            raise ValueError("stratify must have one value per graph row")
        participant_strata = _participant_strata(participants, strata)
    if len(unique_participants) < 5:
        raise ValueError("at least five participants are required for a three-way split")

    train_validation, test = train_test_split(
        unique_participants,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=participant_strata,
    )
    validation_fraction_remaining = validation_size / (1.0 - test_size)
    remaining_strata = None
    if participant_strata is not None:
        strata_by_participant = dict(
            zip(unique_participants.tolist(), participant_strata.tolist(), strict=False)
        )
        remaining_strata = np.asarray(
            [strata_by_participant[participant] for participant in train_validation]
        )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction_remaining,
        random_state=random_state + 1,
        shuffle=True,
        stratify=remaining_strata,
    )

    def row_indices(selected: np.ndarray) -> NDArray[np.int64]:
        return np.flatnonzero(np.isin(participants, selected)).astype(np.int64)

    return ParticipantSplit(
        train_indices=row_indices(train),
        validation_indices=row_indices(validation),
        test_indices=row_indices(test),
        train_participants=np.asarray(train),
        validation_participants=np.asarray(validation),
        test_participants=np.asarray(test),
    )
