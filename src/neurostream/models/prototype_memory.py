"""Cosine prototype memory for a frozen feature extractor."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PrototypeMemory:
	"""Cosine classifier with EMA-updated class prototypes."""

	def __init__(
		self,
		prototypes: torch.Tensor,
		alpha: float = 0.05,
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

	def logits(self, features: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
		if temperature <= 0:
			raise ValueError("temperature must be positive")
		return F.normalize(features, dim=-1) @ F.normalize(self.prototypes, dim=-1).T / temperature

	def classify(self, features: torch.Tensor) -> torch.Tensor:
		"""Classify with argmax cosine similarity, without temperature scaling."""
		cosine = F.normalize(features, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
		return cosine.argmax(dim=-1)

	def predict(self, features: torch.Tensor, temperature: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
		probabilities = self.logits(features, temperature).softmax(dim=-1)
		confidence, labels = probabilities.max(dim=-1)
		return labels, confidence

	def update(
		self, feature: torch.Tensor, pseudo_label: int | torch.Tensor
	) -> None:
		"""Move one class prototype by the exact EMA update rule."""
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
		if not 0 <= confidence_threshold <= 1:
			raise ValueError("confidence_threshold must be in [0, 1]")
		labels, confidence = self.predict(features, temperature)
		accepted = confidence > confidence_threshold
		self.update(features[accepted], labels[accepted])
		return int(accepted.sum().item())
