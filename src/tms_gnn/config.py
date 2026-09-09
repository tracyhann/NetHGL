"""Validated, path-free configuration objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for converting parcellated time series into ROI graphs."""

    lags: tuple[int, ...] = (1, 2, 3, 4, 5, 10, 20, 30)
    ar_order: int = 16
    min_overlap: int = 30
    within_network_only: bool = True
    edge_sign: str = "all"
    include_self: bool = True

    def __post_init__(self) -> None:
        if not self.lags or any(lag <= 0 for lag in self.lags):
            raise ValueError("lags must contain positive integers")
        if tuple(sorted(set(self.lags))) != self.lags:
            raise ValueError("lags must be unique and strictly increasing")
        if self.ar_order <= 0:
            raise ValueError("ar_order must be positive")
        if self.min_overlap < 2:
            raise ValueError("min_overlap must be at least 2")
        if self.edge_sign not in {"all", "positive", "negative"}:
            raise ValueError("edge_sign must be 'all', 'positive', or 'negative'")

    @property
    def feature_dim(self) -> int:
        """Number of ROI features created by the configured extractor."""

        features_per_view = 8 + len(self.lags) + self.ar_order
        return 2 * features_per_view


@dataclass(frozen=True)
class ModelConfig:
    """Architecture configuration for :class:`HierarchicalBrainGNN`."""

    input_dim: int = 64
    hidden_dim: int = 256
    output_dim: int = 1
    num_rois: int = 450
    num_networks: int = 24
    roi_layers: int = 2
    network_layers: int = 2
    heads: int = 1
    dropout: float = 0.2

    def __post_init__(self) -> None:
        integer_fields = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_rois": self.num_rois,
            "roi_layers": self.roi_layers,
            "network_layers": self.network_layers,
            "heads": self.heads,
        }
        if any(value <= 0 for value in integer_fields.values()):
            raise ValueError("model dimensions, layers, and heads must be positive")
        if self.num_rois < 2:
            raise ValueError("num_rois must be at least 2")
        if self.num_networks < 2:
            raise ValueError("num_networks must be at least 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization defaults used in the primary experiments."""

    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 5e-3
    max_epochs: int = 200
    patience: int = 15

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
