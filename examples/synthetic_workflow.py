#!/usr/bin/env python3
"""Run graph construction, inference, and attribution on generated data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main() -> None:
    try:
        from torch_geometric.data import Batch
    except ImportError as error:
        message = "Install the full release dependencies with: pip install -e '.[all]'"
        raise SystemExit(message) from error

    from nethgl.config import GraphConfig, ModelConfig
    from nethgl.graph import build_graph
    from nethgl.interpretability import gradient_times_attention, network_balance
    from nethgl.models import HierarchicalBrainGNN

    rng = np.random.default_rng(12)
    torch.manual_seed(12)
    torch.set_num_threads(1)
    communities = np.repeat(np.arange(4), 2)
    graph_config = GraphConfig(lags=(1, 2, 3), ar_order=3, min_overlap=10)
    graphs = []
    for label in (0, 1):
        timeseries = rng.normal(size=(8, 40))
        timeseries[:2] += label * rng.normal(size=(1, 40))
        graph = build_graph(
            timeseries,
            communities,
            graph_config,
            metadata={"y": torch.tensor(float(label))},
        )
        graphs.append(graph)

    batch = Batch.from_data_list(graphs)
    model = HierarchicalBrainGNN(
        ModelConfig(
            input_dim=graph_config.feature_dim,
            hidden_dim=16,
            num_rois=8,
            num_networks=4,
            dropout=0.0,
        )
    )
    model.eval()
    result = model(batch)
    result["logit"].sum().backward()
    attribution = gradient_times_attention(result["network_attention"], target="remission")
    balance = network_balance(result["network_edge_index"], attribution, n_networks=4)

    print(f"graphs: {batch.num_graphs}")
    print(f"ROI features: {tuple(batch.x.shape)}")
    print(f"depression logits: {tuple(result['logit'].shape)}")
    print(f"network attention: {tuple(result['network_attention'].shape)}")
    print(f"remission-oriented balance: {tuple(balance.shape)}")


if __name__ == "__main__":
    main()
