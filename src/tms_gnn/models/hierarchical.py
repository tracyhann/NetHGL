"""Primary ROI-to-network hierarchical graph neural network."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch_geometric.nn import GATv2Conv, SAGEConv
except ImportError as error:
    raise ImportError(
        "HierarchicalBrainGNN requires the optional 'gnn' dependencies. "
        "Install the project with: pip install -e '.[gnn]'"
    ) from error

from tms_gnn.config import ModelConfig
from tms_gnn.graph.construction import complete_directed_edge_index


class ROIEncoder(nn.Module):
    """GraphSAGE encoder that preserves the configured feature dimension."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        dimensions = _encoder_dimensions(input_dim, hidden_dim, num_layers)
        self.layers = nn.ModuleList(
            SAGEConv(source, destination, aggr="mean")
            for source, destination in zip(dimensions[:-1], dimensions[1:], strict=True)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class LearnedROIToNetworkPool(nn.Module):
    """Learn one globally shared ROI weight within each atlas network."""

    def __init__(self, num_rois: int, num_networks: int) -> None:
        super().__init__()
        self.num_rois = num_rois
        self.num_networks = num_networks
        self.logits = nn.Parameter(torch.zeros(num_rois, num_networks))

    def forward(
        self,
        roi_embeddings: torch.Tensor,
        communities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if roi_embeddings.ndim != 3 or roi_embeddings.shape[1] != self.num_rois:
            raise ValueError("roi_embeddings must be shaped (batch, configured num_rois, features)")
        if communities.shape != (self.num_rois,):
            raise ValueError("communities must contain one network index per configured ROI")
        expected = torch.arange(self.num_networks, device=communities.device)
        if not torch.equal(torch.unique(communities), expected):
            raise ValueError("each configured network must contain at least one ROI")

        allowed = F.one_hot(communities, num_classes=self.num_networks).bool()
        masked_logits = self.logits.masked_fill(~allowed, -torch.inf)
        weights = torch.softmax(masked_logits, dim=0)
        network_embeddings = torch.einsum("brd,rn->bnd", roi_embeddings, weights)
        return network_embeddings, weights


class NetworkEncoder(nn.Module):
    """GATv2 encoder over a complete directed network graph."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = _encoder_dimensions(input_dim, hidden_dim, num_layers)
        self.layers = nn.ModuleList(
            GATv2Conv(
                source,
                destination,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
            )
            for source, destination in zip(dimensions[:-1], dimensions[1:], strict=True)
        )
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_edge_index = edge_index
        attention = torch.empty(0, device=x.device)
        for layer in self.layers:
            x, (attention_edge_index, attention) = layer(
                x, edge_index, return_attention_weights=True
            )
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if attention.requires_grad:
            attention.retain_grad()
        return x, attention_edge_index, attention


class NetworkReadout(nn.Module):
    """Combine class-specific contributions from all network embeddings."""

    def __init__(self, input_dim: int, output_dim: int, num_networks: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.per_network = nn.Linear(input_dim, output_dim)
        self.combine_networks = nn.Linear(num_networks, 1)

    def forward(self, network_embeddings: torch.Tensor) -> torch.Tensor:
        network_logits = self.per_network(network_embeddings).transpose(1, 2)
        output = self.combine_networks(network_logits).squeeze(-1)
        return output.squeeze(-1) if self.output_dim == 1 else output


def _encoder_dimensions(input_dim: int, hidden_dim: int, num_layers: int) -> list[int]:
    if num_layers == 1:
        return [input_dim, input_dim]
    return [input_dim, *([hidden_dim] * (num_layers - 1)), input_dim]


class HierarchicalBrainGNN(nn.Module):
    """Encode ROI graphs, pool to networks, and classify each graph."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.roi_encoder = ROIEncoder(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.roi_layers,
            self.config.dropout,
        )
        self.roi_to_network = LearnedROIToNetworkPool(
            self.config.num_rois, self.config.num_networks
        )
        self.network_encoder = NetworkEncoder(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.network_layers,
            self.config.heads,
            self.config.dropout,
        )
        self.readout = NetworkReadout(
            self.config.input_dim, self.config.output_dim, self.config.num_networks
        )

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        if data.x.shape[-1] != self.config.input_dim:
            raise ValueError("graph feature dimension does not match ModelConfig.input_dim")
        batch_vector = getattr(
            data,
            "batch",
            torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device),
        )
        n_graphs = int(batch_vector.max().item()) + 1
        counts = torch.bincount(batch_vector, minlength=n_graphs)
        if not torch.all(counts == self.config.num_rois):
            raise ValueError("every graph must contain ModelConfig.num_rois ordered ROIs")

        community_matrix = data.community.reshape(n_graphs, self.config.num_rois)
        if not torch.all(community_matrix == community_matrix[0]):
            raise ValueError("all batched graphs must use the same ordered ROI-to-network map")

        roi_embeddings = self.roi_encoder(data.x, data.edge_index)
        roi_embeddings = roi_embeddings.reshape(
            n_graphs, self.config.num_rois, self.config.input_dim
        )
        network_embeddings, pooling_weights = self.roi_to_network(
            roi_embeddings, community_matrix[0]
        )

        one_graph_edges = complete_directed_edge_index(self.config.num_networks).to(data.x.device)
        batched_edges = torch.cat(
            [
                one_graph_edges + graph_index * self.config.num_networks
                for graph_index in range(n_graphs)
            ],
            dim=1,
        )
        encoded_networks, attention_edges, attention = self.network_encoder(
            network_embeddings.reshape(-1, self.config.input_dim), batched_edges
        )
        encoded_networks = encoded_networks.reshape(
            n_graphs, self.config.num_networks, self.config.input_dim
        )
        logit = self.readout(encoded_networks)
        return {
            "logit": logit,
            "roi_pool_weights": pooling_weights,
            "network_embeddings": encoded_networks,
            "network_attention": attention,
            "network_edge_index": attention_edges,
        }
