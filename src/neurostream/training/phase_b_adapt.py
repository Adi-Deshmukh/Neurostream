"""Frozen-SNN target-session adaptation using EMA prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neurostream.models.prototype_memory import PrototypeMemory
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor


@dataclass
class PhaseBResult:
	"""Protocol-specific outputs from a target-session adaptation run."""

	accepted_trials: int
	prototypes: torch.Tensor
	calibration_trials: int = 0
	evaluation_trials: int = 0
	acceptance_rate: float = float("nan")
	consolidation_phases: int = 0
	drift_before_after: float = float("nan")
	static_baseline_predictions: torch.Tensor | None = None
	prequential_predictions: torch.Tensor | None = None
	retrospective_predictions: torch.Tensor | None = None
	source_predictions_before_adaptation: torch.Tensor | None = None
	source_predictions_after_adaptation: torch.Tensor | None = None
	static_baseline_accuracy: float = float("nan")
	prequential_accuracy: float = float("nan")
	retrospective_accuracy: float = float("nan")
	source_accuracy_before_adaptation: float = float("nan")
	source_accuracy_after_adaptation: float = float("nan")
	source_retention_delta: float = float("nan")

	@property
	def before_accuracy(self) -> float:
		"""Compatibility alias for prequential accuracy."""
		return self.prequential_accuracy

	@property
	def after_accuracy(self) -> float:
		"""Compatibility alias for retrospective diagnostic accuracy."""
		return self.retrospective_accuracy

	@property
	def retention_accuracy(self) -> float:
		"""Compatibility alias for post-adaptation source accuracy."""
		return self.source_accuracy_after_adaptation


def adapt_target_session(
	model: SNNFeatureExtractor,
	prototypes: torch.Tensor,
	target_spikes: np.ndarray | torch.Tensor,
	*,
	momentum: float = 0.95,
	confidence_threshold: float = 0.8,
	temperature: float = 0.1,
	batch_size: int = 32,
	source_spikes: np.ndarray | torch.Tensor | None = None,
	enable_consolidation: bool = False,
	consolidation_interval: int = 50,
	sleep_steps: int = 5,
	noise_std: float = 0.05,
	pull_rate: float = 0.5,
) -> PhaseBResult:
	"""Adapt prototypes from target spikes using strictly prequential updates.

	Each target trial is predicted first, then optionally used with its
	pseudo-label for the prototype update. No evaluation labels enter this
	operational path. The SNN parameters are never modified.
	If source trials are supplied, predictions are recorded before and after
	adaptation; source labels must be supplied later to ``score_adaptation``.

	When ``enable_consolidation=True``, a gradient-free sleep-phase consolidation
	mechanism runs every ``consolidation_interval`` adaptation steps, preventing
	drift away from session-1 prototypes without storing raw EEG data.
	"""
	if batch_size < 1:
		raise ValueError("batch_size must be positive")
	spikes = torch.as_tensor(target_spikes, dtype=torch.float32)
	if spikes.ndim != 3:
		raise ValueError("target_spikes must have shape (channels, timesteps, trials)")
	if min(spikes.shape) < 1 or not torch.isfinite(spikes).all():
		raise ValueError("target_spikes must be non-empty and finite")
	if prototypes.ndim != 2 or prototypes.shape[0] < 2 or prototypes.shape[1] < 1:
		raise ValueError("prototypes must have shape (classes, features)")
	if not torch.isfinite(prototypes).all():
		raise ValueError("prototypes must contain only finite values")
	if hasattr(model, "fc1") and model.fc1.in_features != spikes.shape[0]:
		raise ValueError("target spike channel count does not match model input features")
	if hasattr(model, "out_features") and model.out_features != prototypes.shape[1]:
		raise ValueError("prototype feature count does not match model output features")
	memory = PrototypeMemory(prototypes, momentum=momentum)

	consolidator = None
	if enable_consolidation:
		from neurostream.training.sleep_consolidation import SleepConsolidator

		consolidator = SleepConsolidator(
			snapshot_prototypes=prototypes.detach().clone(),
			consolidation_interval=consolidation_interval,
			sleep_steps=sleep_steps,
			noise_std=noise_std,
			pull_rate=pull_rate,
		)

	model.eval()
	with torch.no_grad():
		features = []
		for trial_index in range(spikes.shape[-1]):
			feature = model(spikes[:, :, trial_index:trial_index + 1])
			features.append(feature)
		features = torch.cat(features, dim=0)
		static_baseline_predictions, _ = memory.predict(features, temperature)
		source_features = None
		source_predictions_before = None
		if source_spikes is not None:
			source_tensor = torch.as_tensor(source_spikes, dtype=torch.float32)
			if source_tensor.ndim != 3 or source_tensor.shape[:2] != spikes.shape[:2]:
				raise ValueError("source and target spikes must have matching channel and timestep dimensions")
			if source_tensor.shape[-1] < 1 or not torch.isfinite(source_tensor).all():
				raise ValueError("source spikes must be non-empty and finite")
			source_features = torch.cat(
				[
					model(source_tensor[:, :, start:start + batch_size])
					for start in range(0, source_tensor.shape[-1], batch_size)
				],
				dim=0,
			)
			source_predictions_before, _ = memory.predict(source_features, temperature)
		prequential_predictions = []
		accepted_trials = 0
		consolidation_count = 0
		for trial_index in range(features.shape[0]):
			feature = features[trial_index:trial_index + 1]
			prediction, confidence = memory.predict(feature, temperature)
			prequential_predictions.append(prediction.item())
			if confidence.item() > confidence_threshold:
				memory.update(feature, prediction)
				accepted_trials += 1

				if consolidator is not None:
					updated_proto, step_res = consolidator.step(
						memory.prototypes, step_count=accepted_trials
					)
					if step_res.triggered:
						memory.prototypes = updated_proto
						consolidation_count += 1

		retrospective_predictions, _ = memory.predict(features, temperature)
	source_predictions_after = None
	if source_features is not None:
		source_predictions_after, _ = memory.predict(source_features, temperature)

	drift_val = float("nan")
	if consolidator is not None:
		loss, _ = consolidator.compute_sleep_loss(memory.prototypes)
		drift_val = float(loss.item())

	return PhaseBResult(
		accepted_trials=accepted_trials,
		prototypes=memory.prototypes,
		acceptance_rate=accepted_trials / features.shape[0],
		consolidation_phases=consolidation_count,
		drift_before_after=drift_val,
		static_baseline_predictions=static_baseline_predictions.cpu(),
		prequential_predictions=torch.as_tensor(prequential_predictions, dtype=torch.long),
		retrospective_predictions=retrospective_predictions.cpu(),
		source_predictions_before_adaptation=(
			source_predictions_before.cpu() if source_predictions_before is not None else None
		),
		source_predictions_after_adaptation=(
			source_predictions_after.cpu() if source_predictions_after is not None else None
		),
	)


def score_adaptation(
	result: PhaseBResult,
	target_labels: np.ndarray | torch.Tensor,
	source_labels: np.ndarray | torch.Tensor | None = None,
) -> PhaseBResult:
	"""Score recorded prequential and retrospective predictions.

	This function is deliberately separate from adaptation so changing
	evaluation labels cannot alter prototype updates or predictions.
	"""
	labels = torch.as_tensor(np.asarray(target_labels).reshape(-1), dtype=torch.long)
	if (
		result.static_baseline_predictions is None
		or result.prequential_predictions is None
		or result.retrospective_predictions is None
	):
		raise ValueError("adaptation result does not contain recorded predictions")
	if labels.numel() != result.prequential_predictions.numel():
		raise ValueError("target labels must match the number of recorded predictions")
	result.static_baseline_accuracy = float(
		(result.static_baseline_predictions == labels).float().mean()
	)
	result.prequential_accuracy = float(
		(result.prequential_predictions == labels).float().mean()
	)
	result.retrospective_accuracy = float(
		(result.retrospective_predictions == labels).float().mean()
	)
	if source_labels is not None:
		source_y = torch.as_tensor(np.asarray(source_labels).reshape(-1), dtype=torch.long)
		if (
			result.source_predictions_before_adaptation is None
			or result.source_predictions_after_adaptation is None
			or source_y.numel() != result.source_predictions_before_adaptation.numel()
		):
			raise ValueError("source labels must match recorded source predictions")
		result.source_accuracy_before_adaptation = float(
			(result.source_predictions_before_adaptation == source_y).float().mean()
		)
		result.source_accuracy_after_adaptation = float(
			(result.source_predictions_after_adaptation == source_y).float().mean()
		)
		result.source_retention_delta = (
			result.source_accuracy_after_adaptation
			- result.source_accuracy_before_adaptation
		)
	return result


def calibrate_target_session(
	model: SNNFeatureExtractor,
	prototypes: torch.Tensor,
	target_spikes: np.ndarray | torch.Tensor,
	target_labels: np.ndarray | torch.Tensor,
	*,
	calibration_fraction: float = 0.2,
	momentum: float = 0.5,
	temperature: float = 0.1,
	batch_size: int = 32,
	seed: int = 42,
) -> PhaseBResult:
	"""Use labelled target calibration trials, then score held-out target trials.

	The SNN stays frozen. Calibration labels update prototypes only; evaluation
	labels are used only to calculate the reported before/after accuracies.
	"""
	if not 0 < calibration_fraction < 1:
		raise ValueError("calibration_fraction must be between 0 and 1")
	spikes = torch.as_tensor(target_spikes, dtype=torch.float32)
	labels = torch.as_tensor(np.asarray(target_labels).reshape(-1), dtype=torch.long)
	if spikes.ndim != 3 or labels.numel() != spikes.shape[-1]:
		raise ValueError("target spikes and labels must contain matching trial counts")
	if min(spikes.shape) < 1 or not torch.isfinite(spikes).all():
		raise ValueError("target spikes must be non-empty and finite")
	if prototypes.ndim != 2 or prototypes.shape[0] < 2 or not torch.isfinite(prototypes).all():
		raise ValueError("prototypes must be a finite (classes, features) tensor")
	rng = np.random.default_rng(seed)
	calibration, evaluation = [], []
	for class_index in torch.unique(labels).tolist():
		indices = torch.where(labels == class_index)[0].numpy()
		rng.shuffle(indices)
		n_calibration = min(max(1, int(round(len(indices) * calibration_fraction))), len(indices) - 1)
		calibration.extend(indices[:n_calibration].tolist())
		evaluation.extend(indices[n_calibration:].tolist())
	calibration_indices = torch.as_tensor(calibration, dtype=torch.long)
	evaluation_indices = torch.as_tensor(evaluation, dtype=torch.long)
	memory = PrototypeMemory(prototypes, momentum=momentum)
	model.eval()
	with torch.no_grad():
		features = torch.cat(
			[
				model(spikes[:, :, start:start + batch_size])
				for start in range(0, spikes.shape[-1], batch_size)
			],
			dim=0,
		)
		before, _ = memory.predict(features[evaluation_indices], temperature)
		before_accuracy = float((before == labels[evaluation_indices]).float().mean())
		memory.update(features[calibration_indices], labels[calibration_indices])
		after, _ = memory.predict(features[evaluation_indices], temperature)
		after_accuracy = float((after == labels[evaluation_indices]).float().mean())
	return PhaseBResult(
		accepted_trials=calibration_indices.numel(),
		prototypes=memory.prototypes,
		calibration_trials=calibration_indices.numel(),
		evaluation_trials=evaluation_indices.numel(),
		static_baseline_accuracy=before_accuracy,
		prequential_accuracy=after_accuracy,
		retrospective_accuracy=after_accuracy,
		static_baseline_predictions=before.cpu(),
		prequential_predictions=after.cpu(),
		retrospective_predictions=after.cpu(),
	)
