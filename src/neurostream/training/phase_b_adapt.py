"""Frozen-SNN target-session adaptation using EMA prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neurostream.models.prototype_memory import PrototypeMemory
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor


@dataclass
class PhaseBResult:
	"""Metrics from a target-session adaptation run."""

	before_accuracy: float
	after_accuracy: float
	accepted_trials: int
	prototypes: torch.Tensor
	calibration_trials: int = 0
	evaluation_trials: int = 0
	retention_accuracy: float = float("nan")
	acceptance_rate: float = float("nan")
	consolidation_phases: int = 0
	drift_before_after: float = float("nan")


def adapt_target_session(
	model: SNNFeatureExtractor,
	prototypes: torch.Tensor,
	target_spikes: np.ndarray | torch.Tensor,
	target_labels: np.ndarray | torch.Tensor | None = None,
	*,
	momentum: float = 0.95,
	confidence_threshold: float = 0.8,
	temperature: float = 0.1,
	batch_size: int = 32,
	source_spikes: np.ndarray | torch.Tensor | None = None,
	source_labels: np.ndarray | torch.Tensor | None = None,
	enable_consolidation: bool = False,
	consolidation_interval: int = 50,
	sleep_steps: int = 5,
	noise_std: float = 0.05,
	pull_rate: float = 0.5,
) -> PhaseBResult:
	"""Adapt prototypes from target spikes without using labels for updates.

	Target labels, when supplied, are used only to report before/after accuracy.
	The SNN parameters are never modified.

	When ``enable_consolidation=True``, a gradient-free sleep-phase consolidation
	mechanism runs every ``consolidation_interval`` adaptation steps, preventing
	drift away from session-1 prototypes without storing raw EEG data.
	"""
	if batch_size < 1:
		raise ValueError("batch_size must be positive")
	spikes = torch.as_tensor(target_spikes, dtype=torch.float32)
	if spikes.ndim != 3:
		raise ValueError("target_spikes must have shape (channels, timesteps, trials)")
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
		before, _ = memory.predict(features, temperature)
		before_accuracy = float("nan")
		labels = None
		if target_labels is not None:
			labels = torch.as_tensor(np.asarray(target_labels).reshape(-1), dtype=torch.long)
			if labels.numel() != features.shape[0]:
				raise ValueError("each target trial must have exactly one label")
			before_accuracy = float((before == labels).float().mean())
		accepted_trials = 0
		consolidation_count = 0
		for trial_index in range(features.shape[0]):
			feature = features[trial_index:trial_index + 1]
			prediction, confidence = memory.predict(feature, temperature)
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

		after, _ = memory.predict(features, temperature)
		after_accuracy = float("nan")
		if labels is not None:
			after_accuracy = float((after == labels).float().mean())
	retention_accuracy = float("nan")
	if source_spikes is not None and source_labels is not None:
		source_tensor = torch.as_tensor(source_spikes, dtype=torch.float32)
		source_y = torch.as_tensor(np.asarray(source_labels).reshape(-1), dtype=torch.long)
		with torch.no_grad():
			source_features = torch.cat(
				[
					model(source_tensor[:, :, start:start + batch_size])
					for start in range(0, source_tensor.shape[-1], batch_size)
				],
				dim=0,
			)
			retention, _ = memory.predict(source_features, temperature)
			retention_accuracy = float((retention == source_y).float().mean())

	drift_val = float("nan")
	if consolidator is not None:
		loss, _ = consolidator.compute_sleep_loss(memory.prototypes)
		drift_val = float(loss.item())

	return PhaseBResult(
		before_accuracy=before_accuracy,
		after_accuracy=after_accuracy,
		accepted_trials=accepted_trials,
		prototypes=memory.prototypes,
		retention_accuracy=retention_accuracy,
		acceptance_rate=accepted_trials / features.shape[0],
		consolidation_phases=consolidation_count,
		drift_before_after=drift_val,
	)


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
		before_accuracy=before_accuracy,
		after_accuracy=after_accuracy,
		accepted_trials=calibration_indices.numel(),
		prototypes=memory.prototypes,
		calibration_trials=calibration_indices.numel(),
		evaluation_trials=evaluation_indices.numel(),
	)
