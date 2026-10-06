"""Compact training loop for the hierarchical binary classifier."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from tms_gnn.config import TrainingConfig
from tms_gnn.training.metrics import select_threshold


@dataclass
class FitResult:
    """Fitted model, best weights, epoch-level history, and validation threshold.

    ``threshold`` is the validation-selected decision threshold of the best
    checkpoint when ``selection_metric="min_class_recall"``, else ``None``.
    """

    model: nn.Module
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    history: list[dict[str, float]]
    threshold: float | None = None


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _batch_loss(
    model: nn.Module,
    batch: Any,
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    batch = batch.to(device)
    logits = model(batch)["logit"].reshape(-1)
    labels = batch.y.float().reshape(-1)
    if labels.shape != logits.shape:
        raise ValueError("each graph must provide one binary y value")
    return criterion(logits, labels)


def train_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device = "cpu",
) -> float:
    """Run one optimization epoch and return graph-weighted mean loss."""

    model.train()
    target_device = torch.device(device)
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, criterion, target_device)
        loss.backward()
        optimizer.step()
        n_graphs = int(batch.num_graphs)
        total_loss += float(loss.detach()) * n_graphs
        total_graphs += n_graphs
    if total_graphs == 0:
        raise ValueError("training loader is empty")
    return total_loss / total_graphs


@torch.no_grad()
def _validation_loss(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        loss = _batch_loss(model, batch, criterion, device)
        n_graphs = int(batch.num_graphs)
        total_loss += float(loss) * n_graphs
        total_graphs += n_graphs
    if total_graphs == 0:
        raise ValueError("validation loader is empty")
    return total_loss / total_graphs


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: Iterable,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """Return graph logits, probabilities, and optional labels."""

    target_device = torch.device(device)
    model.eval()
    logits_blocks = []
    label_blocks = []
    for batch in loader:
        batch = batch.to(target_device)
        logits_blocks.append(model(batch)["logit"].reshape(-1).detach().cpu())
        if hasattr(batch, "y"):
            label_blocks.append(batch.y.reshape(-1).detach().cpu())
    if not logits_blocks:
        raise ValueError("prediction loader is empty")
    logits = torch.cat(logits_blocks).numpy()
    labels = torch.cat(label_blocks).numpy() if label_blocks else np.asarray([], dtype=float)
    return {
        "logits": logits,
        "probabilities": torch.sigmoid(torch.as_tensor(logits)).numpy(),
        "labels": labels,
    }


def fit(
    model: nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable,
    config: TrainingConfig | None = None,
    *,
    device: str | torch.device = "cpu",
    positive_class_weight: float | None = None,
) -> FitResult:
    """Fit with AdamW, selecting the checkpoint and stopping early on validation.

    The selection metric is ``config.selection_metric``: the validation minimum
    class recall at its best threshold (ties broken by lower validation loss), or
    the validation BCE loss. Training stops after ``config.patience`` epochs
    without improvement or after ``config.max_epochs`` epochs.
    """

    training_config = config or TrainingConfig()
    target_device = torch.device(device)
    model = model.to(target_device)
    if positive_class_weight is not None and positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be positive")
    weight = (
        torch.tensor(float(positive_class_weight), device=target_device)
        if positive_class_weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    use_recall = training_config.selection_metric == "min_class_recall"
    best_score: tuple[float, float] = (-float("inf"), -float("inf"))
    best_epoch = -1
    best_threshold: float | None = None
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    for epoch in range(training_config.max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, target_device)
        validation_loss = _validation_loss(model, validation_loader, criterion, target_device)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
        }
        threshold = None
        if use_recall:
            threshold, min_recall = _validation_min_class_recall(
                model, validation_loader, target_device
            )
            row["validation_min_class_recall"] = min_recall
            score = (min_recall, -validation_loss)
        else:
            score = (-validation_loss, 0.0)
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.patience:
                break

    if not best_state:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(copy.deepcopy(best_state))
    return FitResult(
        model=model,
        best_state=best_state,
        best_epoch=best_epoch,
        history=history,
        threshold=best_threshold,
    )


def _validation_min_class_recall(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> tuple[float, float]:
    """Return the validation threshold maximizing the smaller class recall, and that recall."""

    prediction = predict(model, loader, device=device)
    if len(np.unique(prediction["labels"])) != 2:
        raise ValueError(
            "selection_metric='min_class_recall' needs both classes in the validation set; "
            "use selection_metric='validation_loss' otherwise"
        )
    threshold, metrics = select_threshold(
        prediction["labels"], prediction["probabilities"], objective="min_class_recall"
    )
    return threshold, float(min(metrics["specificity"], metrics["recall"]))
