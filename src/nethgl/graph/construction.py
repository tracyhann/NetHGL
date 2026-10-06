"""Construction of ROI graphs and complete directed network graphs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

try:
    from torch_geometric.data import Data as GraphData
except ImportError:
    class GraphData:  # type: ignore[no-redef]
        """Attribute container used when the optional PyG backend is absent."""

        def __init__(self, **attributes: Any) -> None:
            for key, value in attributes.items():
                setattr(self, key, value)

from nethgl.config import GraphConfig
from nethgl.graph.connectivity import pairwise_pearson_connectivity
from nethgl.graph.features import extract_roi_timeseries_features


def _validated_communities(communities: ArrayLike, n_rois: int) -> np.ndarray:
    values = np.asarray(communities)
    if values.ndim != 1 or len(values) != n_rois:
        raise ValueError("communities must have one entry per ROI")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("communities must contain integer network indices")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError("communities must contain non-negative network indices")
    observed = np.unique(values)
    if not np.array_equal(observed, np.arange(int(observed[-1]) + 1)):
        raise ValueError("communities must use contiguous indices beginning at zero")
    return values


def complete_directed_edge_index(n_nodes: int, include_self: bool = True) -> torch.Tensor:
    """Return every ordered node pair as a PyG edge index."""

    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    nodes = torch.arange(n_nodes, dtype=torch.long)
    edge_index = torch.cartesian_prod(nodes, nodes).T.contiguous()
    if not include_self:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index


def roi_edge_index(
    connectivity: ArrayLike,
    communities: ArrayLike,
    *,
    within_network_only: bool = True,
    edge_sign: str = "all",
    include_self: bool = True,
) -> torch.Tensor:
    """Build directed ROI edges with optional community and FC-sign filters."""

    fc = np.asarray(connectivity, dtype=float)
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError("connectivity must be a square ROI-by-ROI matrix")
    labels = _validated_communities(communities, fc.shape[0])
    if edge_sign not in {"all", "positive", "negative"}:
        raise ValueError("edge_sign must be 'all', 'positive', or 'negative'")

    mask = np.ones_like(fc, dtype=bool)
    if within_network_only:
        mask &= labels[:, None] == labels[None, :]
    if edge_sign == "positive":
        mask &= fc > 0
    elif edge_sign == "negative":
        mask &= fc < 0
    if not include_self:
        np.fill_diagonal(mask, False)

    source, destination = np.nonzero(mask)
    return torch.as_tensor(np.stack([source, destination]), dtype=torch.long).contiguous()


def build_graph(
    timeseries: ArrayLike,
    communities: ArrayLike,
    config: GraphConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GraphData:
    """Convert one parcellated ROI time series into a PyG graph."""

    graph_config = config or GraphConfig()
    values = np.asarray(timeseries, dtype=float)
    if values.ndim != 2:
        raise ValueError("timeseries must be a two-dimensional ROI-by-time array")
    labels = _validated_communities(communities, values.shape[0])
    connectivity = pairwise_pearson_connectivity(values, graph_config.min_overlap)
    features = extract_roi_timeseries_features(values, graph_config)
    edges = roi_edge_index(
        connectivity,
        labels,
        within_network_only=graph_config.within_network_only,
        edge_sign=graph_config.edge_sign,
        include_self=graph_config.include_self,
    )

    attributes: dict[str, Any] = {
        "x": torch.as_tensor(features, dtype=torch.float32),
        "fc": torch.as_tensor(connectivity, dtype=torch.float32),
        "edge_index": edges,
        "community": torch.as_tensor(labels, dtype=torch.long),
        "num_networks": int(labels.max()) + 1,
    }
    reserved = set(attributes)
    for key, value in (metadata or {}).items():
        if key in reserved:
            raise ValueError(f"metadata key {key!r} conflicts with a graph attribute")
        attributes[key] = value
    return GraphData(**attributes)
