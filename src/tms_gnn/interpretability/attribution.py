"""Directed network-edge attribution and balance definitions."""

from __future__ import annotations

import torch


def gradient_times_attention(
    attention: torch.Tensor,
    gradient: torch.Tensor | None = None,
    *,
    target: str = "depression",
) -> torch.Tensor:
    """Multiply GAT attention by its selected-logit gradient.

    The binary model emits a depression logit. ``target='remission'`` changes
    orientation by multiplying the depression-oriented attribution by -1.
    """

    if target not in {"depression", "remission"}:
        raise ValueError("target must be 'depression' or 'remission'")
    selected_gradient = attention.grad if gradient is None else gradient
    if selected_gradient is None:
        raise ValueError("gradient is required; backpropagate the selected logit first")
    if attention.shape != selected_gradient.shape:
        raise ValueError("attention and gradient must have the same shape")
    attribution = attention * selected_gradient
    return -attribution if target == "remission" else attribution


def _edge_values(attribution: torch.Tensor, n_edges: int) -> torch.Tensor:
    if attribution.shape[0] != n_edges:
        raise ValueError("attribution must provide one value per edge")
    if attribution.ndim == 1:
        return attribution
    if attribution.ndim == 2:
        return attribution.mean(dim=1)
    raise ValueError("attribution must be shaped (edges,) or (edges, heads)")


def _edge_graph_layout(
    edge_index: torch.Tensor, n_networks: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
        raise ValueError("edge_index must be a non-empty tensor shaped (2, edges)")
    if n_networks < 2:
        raise ValueError("n_networks must be at least 2")
    source, destination = edge_index.long()
    source_graph = torch.div(source, n_networks, rounding_mode="floor")
    destination_graph = torch.div(destination, n_networks, rounding_mode="floor")
    if not torch.equal(source_graph, destination_graph):
        raise ValueError("network edges cannot connect different graphs in a batch")
    n_graphs = int(source_graph.max().item()) + 1
    return source % n_networks, destination % n_networks, source_graph, n_graphs


def network_balance(
    edge_index: torch.Tensor,
    attribution: torch.Tensor,
    n_networks: int,
    *,
    exclude_self: bool = True,
) -> torch.Tensor:
    """Compute outgoing-minus-incoming attribution for every network."""

    values = _edge_values(attribution, edge_index.shape[1])
    source, destination, graph_index, n_graphs = _edge_graph_layout(edge_index, n_networks)
    keep = source != destination if exclude_self else torch.ones_like(source, dtype=torch.bool)
    flat_source = graph_index[keep] * n_networks + source[keep]
    flat_destination = graph_index[keep] * n_networks + destination[keep]
    outgoing = values.new_zeros(n_graphs * n_networks)
    incoming = values.new_zeros(n_graphs * n_networks)
    outgoing.index_add_(0, flat_source, values[keep])
    incoming.index_add_(0, flat_destination, values[keep])
    result = (outgoing - incoming).reshape(n_graphs, n_networks)
    return result[0] if n_graphs == 1 else result


def dyadic_balance(
    edge_index: torch.Tensor,
    attribution: torch.Tensor,
    *,
    source: int,
    destination: int,
    n_networks: int | None = None,
) -> torch.Tensor:
    """Compute source→destination minus destination→source attribution."""

    inferred_networks = n_networks or int(edge_index.max().item()) + 1
    if not 0 <= source < inferred_networks or not 0 <= destination < inferred_networks:
        raise ValueError("source and destination must be valid network indices")
    if source == destination:
        raise ValueError("dyadic balance requires two different networks")
    values = _edge_values(attribution, edge_index.shape[1])
    local_source, local_destination, graph_index, n_graphs = _edge_graph_layout(
        edge_index, inferred_networks
    )
    forward = (local_source == source) & (local_destination == destination)
    reverse = (local_source == destination) & (local_destination == source)
    result = values.new_zeros(n_graphs)
    result.index_add_(0, graph_index[forward], values[forward])
    result.index_add_(0, graph_index[reverse], -values[reverse])
    return result[0] if n_graphs == 1 else result

