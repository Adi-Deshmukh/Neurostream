"""Cosine prototype memory for a frozen SNN feature extractor.

Implements a 512-dim prototype memory head that replaces a frozen softmax classifier.
This is the core online adaptation mechanism: centroids update incrementally per class
as new (drifted) data arrives:

    P_c ← (1 - α) * P_c + α * f_t    (default α = 0.3)

avoiding catastrophic forgetting from full backpropagation fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PrototypeMemory:
    """Cosine classifier with EMA-updated class prototypes.

    Parameters
    ----------
    prototypes : Tensor, shape ``(classes, features)``
        Initial class centroids (typically 512-dim from Phase A).
    alpha : float
        Exponential Moving Average (EMA) adaptation rate. Default 0.3.
    momentum : float | None
        Legacy decay coefficient ``1 - alpha``. If provided, overrides ``alpha``.
    """

    def __init__(
        self,
        prototypes: torch.Tensor,
        alpha: float = 0.3,
        *,
        momentum: float | None = None,
    ):
        if prototypes.ndim != 2 or prototypes.shape[0] < 2:
            raise ValueError("prototypes must have shape (classes, features)")
        if momentum is not None:
            alpha = 1.0 - momentum
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.prototypes = prototypes.detach().clone().float()
        self.alpha = alpha

    @property
    def momentum(self) -> float:
        """Legacy decay coefficient, equal to ``1 - alpha``."""
        return 1.0 - self.alpha

    @property
    def feature_dim(self) -> int:
        """Dimensionality of each prototype vector."""
        return self.prototypes.shape[1]

    @property
    def n_classes(self) -> int:
        """Number of distinct classes."""
        return self.prototypes.shape[0]

    def logits(self, features: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Compute cosine similarity logits scaled by temperature."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return F.normalize(features, dim=-1) @ F.normalize(self.prototypes, dim=-1).T / temperature

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify with argmax cosine similarity, without temperature scaling."""
        cosine = F.normalize(features, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        return cosine.argmax(dim=-1)

    def predict(self, features: torch.Tensor, temperature: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted class labels and softmax confidence scores."""
        probabilities = self.logits(features, temperature).softmax(dim=-1)
        confidence, labels = probabilities.max(dim=-1)
        return labels, confidence

    def cosine_margin(self, features: torch.Tensor) -> torch.Tensor:
        """Compute top-1 minus top-2 cosine similarity margin per example."""
        cosine = F.normalize(features, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        top2 = torch.topk(cosine, k=2, dim=-1).values
        return top2[:, 0] - top2[:, 1]

    def pairwise_class_distances(self) -> torch.Tensor:
        """Compute pairwise cosine distance matrix between class prototypes."""
        norm_p = F.normalize(self.prototypes, dim=-1)
        similarity = norm_p @ norm_p.T
        return 1.0 - similarity

    def update(
        self, feature: torch.Tensor, pseudo_label: int | torch.Tensor
    ) -> None:
        """Move one or more class prototypes by the exact EMA update rule."""
        features = torch.as_tensor(feature, dtype=self.prototypes.dtype)
        labels = torch.as_tensor(pseudo_label, dtype=torch.long)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
        if features.ndim != 2 or labels.numel() != features.shape[0]:
            raise ValueError("feature and pseudo_label must describe matching examples")
        if (labels < 0).any() or (labels >= self.prototypes.shape[0]).any():
            raise ValueError("pseudo_label is outside the prototype class range")
        for current_feature, current_label in zip(features, labels):
            class_index = int(current_label)
            self.prototypes[class_index] = (
                (1.0 - self.alpha) * self.prototypes[class_index]
                + self.alpha * current_feature
            )

    def adapt(
        self,
        features: torch.Tensor,
        confidence_threshold: float = 0.6,
        temperature: float = 0.1,
    ) -> int:
        """Adapt prototypes using confidence-gated pseudo-labels."""
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        labels, confidence = self.predict(features, temperature)
        accepted = confidence > confidence_threshold
        if accepted.any():
            self.update(features[accepted], labels[accepted])
        return int(accepted.sum().item())
