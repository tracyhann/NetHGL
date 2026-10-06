"""Network-component removal and retain-only perturbation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ConfidenceInterval:
    """Point estimate and percentile confidence interval."""

    estimate: float
    lower: float
    upper: float


def component_mask(
    edge_index: torch.Tensor,
    *,
    target_network: int,
    component: str,
    partner_networks: Iterable[int] | None = None,
) -> torch.Tensor:
    """Select outgoing, incoming, or all incident target edges.

    Self-edges are never part of a perturbation component.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be shaped (2, edges)")
    if component not in {"outgoing", "incoming", "incident"}:
        raise ValueError("component must be 'outgoing', 'incoming', or 'incident'")
    source, destination = edge_index
    nonself = source != destination
    outgoing = (source == target_network) & nonself
    incoming = (destination == target_network) & nonself
    if partner_networks is not None:
        partners = torch.as_tensor(
            tuple(partner_networks), dtype=edge_index.dtype, device=edge_index.device
        )
        outgoing &= torch.isin(destination, partners)
        incoming &= torch.isin(source, partners)
    if component == "outgoing":
        return outgoing
    if component == "incoming":
        return incoming
    return outgoing | incoming


def apply_perturbation(
    edge_index: torch.Tensor,
    component: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Remove a component or retain only that component plus self-edges."""

    if component.ndim != 1 or len(component) != edge_index.shape[1]:
        raise ValueError("component must be a boolean mask with one value per edge")
    component = component.bool()
    if mode == "remove":
        keep = ~component
    elif mode == "retain":
        keep = component | (edge_index[0] == edge_index[1])
    else:
        raise ValueError("mode must be 'remove' or 'retain'")
    return edge_index[:, keep].contiguous()


def signed_necessity_or_sufficiency(
    target_effect: float,
    matched_control_effect: float,
    *,
    mode: str,
) -> float:
    """Orient contrasts so positive values indicate necessity or sufficiency.

    For removal, the inputs are prediction reductions and the contrast is
    target minus control. For retain-only, the inputs are prediction errors
    and the contrast is control minus target.
    """

    if mode == "remove":
        return float(target_effect - matched_control_effect)
    if mode == "retain":
        return float(matched_control_effect - target_effect)
    raise ValueError("mode must be 'remove' or 'retain'")


def cluster_bootstrap_mean(
    values: ArrayLike,
    participant_ids: ArrayLike,
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> ConfidenceInterval:
    """Bootstrap participant means, giving each participant equal weight."""

    observations = np.asarray(values, dtype=float)
    participants = np.asarray(participant_ids)
    if observations.ndim != 1 or participants.ndim != 1 or observations.shape != participants.shape:
        raise ValueError("values and participant_ids must be same-length one-dimensional arrays")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    finite = np.isfinite(observations)
    observations = observations[finite]
    participants = participants[finite]
    if len(observations) == 0:
        raise ValueError("values contain no finite observations")

    unique_participants = np.unique(participants)
    participant_means = np.asarray(
        [observations[participants == participant].mean() for participant in unique_participants]
    )
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(participant_means), size=(n_bootstrap, len(participant_means)))
    draws = participant_means[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=float(participant_means.mean()),
        lower=float(np.quantile(draws, tail)),
        upper=float(np.quantile(draws, 1.0 - tail)),
    )


def matched_control_empirical_p(
    target: float,
    matched_controls: ArrayLike,
    *,
    tail: str = "greater",
) -> float:
    """Compare one target statistic with a finite matched-control reference."""

    controls = np.asarray(matched_controls, dtype=float)
    controls = controls[np.isfinite(controls)]
    if len(controls) == 0 or not np.isfinite(target):
        raise ValueError("target and matched_controls must contain finite values")
    if tail == "greater":
        exceedances = np.sum(controls >= target)
    elif tail == "less":
        exceedances = np.sum(controls <= target)
    elif tail == "two-sided":
        exceedances = np.sum(np.abs(controls) >= abs(target))
    else:
        raise ValueError("tail must be 'greater', 'less', or 'two-sided'")
    return float((1 + exceedances) / (len(controls) + 1))
