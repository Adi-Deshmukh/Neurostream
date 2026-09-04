"""Compact EEGNet-style convolutional baseline for preprocessed EEG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class EEGNetBaseline(nn.Module):
    """Small temporal-spatial CNN for input shaped ``(N, channels, samples)``."""

    def __init__(self, channels: int, n_classes: int = 4, temporal_filters: int = 8):
        super().__init__()
        if channels < 1 or n_classes < 2 or temporal_filters < 1:
            raise ValueError("channels, n_classes, and temporal_filters must be positive")
        self.features = nn.Sequential(
            nn.Conv2d(1, temporal_filters, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                temporal_filters,
                (channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(temporal_filters, temporal_filters * 2, (1, 16), bias=False),
            nn.BatchNorm2d(temporal_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, epochs: torch.Tensor) -> torch.Tensor:
        if epochs.ndim != 3:
            raise ValueError("epochs must have shape (trials, channels, samples)")
        features = self.features(epochs.unsqueeze(1))
        return self.classifier(torch.flatten(features, start_dim=1))


@dataclass
class BaselineResult:
    model: EEGNetBaseline
    train_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    checkpoint_path: Path
    label_values: np.ndarray


def _labels(labels: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    values = np.asarray(labels).reshape(-1)
    unique = np.unique(values)
    if unique.size < 2:
        raise ValueError("at least two classes are required")
    return torch.as_tensor(np.searchsorted(unique, values), dtype=torch.long), unique


def _split(labels: torch.Tensor, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    fit, validation = [], []
    for class_index in torch.unique(labels).tolist():
        indices = torch.where(labels == class_index)[0].numpy()
        rng.shuffle(indices)
        count = min(max(1, round(len(indices) * fraction)), len(indices) - 1)
        validation.extend(indices[:count])
        fit.extend(indices[count:])
    return np.asarray(fit), np.asarray(validation)


def _accuracy(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)
    return float((predictions == labels).float().mean())


def train_eegnet_baseline(
    train_epochs: np.ndarray,
    train_labels: np.ndarray,
    test_epochs: np.ndarray,
    test_labels: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 42,
    checkpoint_path: str | Path = "results/checkpoints/eegnet_subject_1.pt",
) -> BaselineResult:
    """Train and evaluate EEGNet using the same session protocol as Phase A."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_x = torch.as_tensor(train_epochs, dtype=torch.float32)
    test_x = torch.as_tensor(test_epochs, dtype=torch.float32)
    if train_x.ndim != 3 or test_x.ndim != 3:
        raise ValueError("epochs must have shape (trials, channels, samples)")
    train_y, label_values = _labels(train_labels)
    test_values = np.asarray(test_labels).reshape(-1)
    if not np.isin(test_values, label_values).all():
        raise ValueError("test labels contain classes absent from training labels")
    test_y = torch.as_tensor(np.searchsorted(label_values, test_values), dtype=torch.long)
    if train_x.shape[0] != train_y.numel() or test_x.shape[0] != test_y.numel():
        raise ValueError("each trial must have exactly one label")
    fit_indices, validation_indices = _split(train_y, validation_fraction, seed)
    model = EEGNetBaseline(train_x.shape[1], len(label_values))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    best_validation = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        order = fit_indices.copy()
        np.random.default_rng(seed + epoch).shuffle(order)
        for batch in np.array_split(order, max(1, int(np.ceil(len(order) / batch_size)))):
            batch_indices = torch.as_tensor(batch, dtype=torch.long)
            loss = loss_function(model(train_x[batch_indices]), train_y[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation_accuracy = _accuracy(model, train_x[validation_indices], train_y[validation_indices])
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "label_values": label_values, "seed": seed}, checkpoint)
    return BaselineResult(
        model=model,
        train_accuracy=_accuracy(model, train_x[fit_indices], train_y[fit_indices]),
        validation_accuracy=_accuracy(model, train_x[validation_indices], train_y[validation_indices]),
        test_accuracy=_accuracy(model, test_x, test_y),
        checkpoint_path=checkpoint,
        label_values=label_values,
    )
