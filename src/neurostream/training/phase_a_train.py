"""Supervised Phase A training for the NeuroStream feature extractor."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurostream.models.snn_feature_extractor import SNNFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class PhaseAResult:
	"""Artifacts and metrics produced by Phase A."""

	model: SNNFeatureExtractor
	prototypes: torch.Tensor
	train_accuracy: float
	validation_accuracy: float
	validation_size: int
	train_losses: list[float]
	label_values: np.ndarray
	checkpoint_path: Path | None
	prototype_path: Path | None
	linear_head_train_accuracy: float | None = None
	encoder_calibration: object | None = None  # EncoderCalibration | None


def _as_spike_tensor(spikes: np.ndarray | torch.Tensor) -> torch.Tensor:
	tensor = torch.as_tensor(spikes, dtype=torch.float32)
	if tensor.ndim != 3:
		raise ValueError("spikes must have shape (channels, timesteps, trials)")
	if min(tensor.shape) < 1:
		raise ValueError("spikes must contain at least one channel, timestep, and trial")
	if not torch.isfinite(tensor).all():
		raise ValueError("spikes must contain only finite values")
	return tensor


def _encode_labels(labels: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
	values = np.asarray(labels).reshape(-1)
	if values.size == 0 or not np.isfinite(values).all():
		raise ValueError("labels must be non-empty and finite")
	label_values = np.unique(values)
	if label_values.size == 0:
		raise ValueError("labels must not be empty")
	encoded = np.searchsorted(label_values, values)
	return torch.as_tensor(encoded, dtype=torch.long), label_values


def _class_prototypes(
	features: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> torch.Tensor:
	prototypes = []
	for class_index in range(n_classes):
		members = features[labels == class_index]
		if members.numel() == 0:
			raise ValueError(f"training data has no examples for class {class_index}")
		prototypes.append(members.mean(dim=0))
	return torch.stack(prototypes)


def prototype_logits(
	features: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
	"""Return cosine-similarity logits for prototype-based classification."""
	if temperature <= 0:
		raise ValueError("temperature must be positive")
	return F.normalize(features, dim=-1) @ F.normalize(prototypes, dim=-1).T / temperature


def _accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
	return float((predictions == labels).float().mean().cpu())


def _set_seed(seed: int) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _stratified_split(
	labels: torch.Tensor, validation_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return fit/validation indices while retaining every class in both sets."""
	if not 0 < validation_fraction < 1:
		raise ValueError("validation_fraction must be between 0 and 1")
	rng = np.random.default_rng(seed)
	fit_indices: list[int] = []
	validation_indices: list[int] = []
	for class_index in torch.unique(labels).cpu().tolist():
		class_indices = torch.where(labels == class_index)[0].cpu().numpy()
		if class_indices.size < 2:
			raise ValueError("each class needs at least two trials for validation")
		rng.shuffle(class_indices)
		n_validation = max(1, int(round(class_indices.size * validation_fraction)))
		if n_validation >= class_indices.size:
			n_validation = class_indices.size - 1
		validation_indices.extend(class_indices[:n_validation].tolist())
		fit_indices.extend(class_indices[n_validation:].tolist())
	return (
		torch.as_tensor(fit_indices, dtype=torch.long, device=labels.device),
		torch.as_tensor(validation_indices, dtype=torch.long, device=labels.device),
	)


def _evaluate_prototypes(
	network: SNNFeatureExtractor,
	inputs: torch.Tensor,
	labels: torch.Tensor,
	prototypes: torch.Tensor,
	temperature: float,
) -> float:
	network.eval()
	with torch.no_grad():
		predictions = prototype_logits(
			network(inputs), prototypes, temperature
		).argmax(dim=1)
	return _accuracy(predictions, labels)


def _prototype_training(
	network: SNNFeatureExtractor,
	train_x: torch.Tensor,
	train_y: torch.Tensor,
	*,
	n_classes: int,
	epochs: int,
	learning_rate: float,
	temperature: float,
	batch_size: int,
	augmentation_probability: float,
	max_grad_norm: float = 1.0,
	weight_decay: float = 1e-4,
	validation_x: torch.Tensor | None = None,
	validation_y: torch.Tensor | None = None,
	patience: int = 10,
) -> tuple[list[float], float]:
	if patience < 1:
		raise ValueError("patience must be at least 1")
	network.eval()
	with torch.no_grad():
		initial_features = network(train_x)
		initial_prototypes = _class_prototypes(initial_features, train_y, n_classes)
	learned_prototypes = nn.Parameter(initial_prototypes.detach().clone())
	optimizer = torch.optim.Adam(
		list(network.parameters()) + [learned_prototypes],
		lr=learning_rate, weight_decay=weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
	loss_function = nn.CrossEntropyLoss()
	losses: list[float] = []
	best_validation = -1.0
	best_state: dict[str, torch.Tensor] | None = None
	without_improvement = 0
	n_trials = train_x.shape[-1]
	for epoch in range(epochs):
		network.train()
		order = torch.randperm(n_trials, device=train_x.device)
		epoch_loss = 0.0
		for indices in order.split(batch_size):
			batch_x = train_x[:, :, indices]
			if augmentation_probability > 0:
				keep = torch.rand(batch_x.shape, device=batch_x.device) > augmentation_probability
				batch_x = batch_x * keep
			features = network(batch_x)
			loss = loss_function(
				prototype_logits(features, learned_prototypes, temperature),
				train_y[indices],
			)
			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(
				list(network.parameters()) + [learned_prototypes], max_grad_norm
			)
			optimizer.step()
			epoch_loss += float(loss.detach().cpu()) * indices.numel()
		scheduler.step()
		losses.append(epoch_loss / n_trials)

		# Per-epoch logging with optional validation accuracy
		if validation_x is not None and validation_y is not None:
			network.eval()
			with torch.no_grad():
				fit_feats = network(train_x)
				protos = _class_prototypes(fit_feats, train_y, n_classes)
				val_acc = _evaluate_prototypes(
					network, validation_x, validation_y, protos, temperature
				)
			if val_acc > best_validation:
				best_validation = val_acc
				best_state = {
					name: value.detach().clone()
					for name, value in network.state_dict().items()
				}
				without_improvement = 0
			else:
				without_improvement += 1
			logger.info(
				"Phase A epoch %d/%d loss=%.5f val_acc=%.2f%% lr=%.2e",
				epoch + 1, epochs, losses[-1], val_acc * 100,
				scheduler.get_last_lr()[0],
			)
			if without_improvement >= patience:
				logger.info("Early stopping after %d epochs without validation improvement", patience)
				break
		else:
			logger.info(
				"Phase A epoch %d/%d loss=%.5f lr=%.2e",
				epoch + 1, epochs, losses[-1],
				scheduler.get_last_lr()[0],
			)

	if best_state is not None:
		network.load_state_dict(best_state)
	network.eval()
	with torch.no_grad():
		features = network(train_x)
		prototypes = _class_prototypes(features, train_y, n_classes)
		predictions = prototype_logits(features, prototypes, temperature).argmax(dim=1)
	return losses, _accuracy(predictions, train_y)


def _linear_head_comparison(
	initial_state: dict[str, torch.Tensor],
	train_x: torch.Tensor,
	train_y: torch.Tensor,
	eval_x: torch.Tensor,
	eval_y: torch.Tensor,
	*,
	in_features: int,
	hidden_features: int,
	out_features: int,
	n_classes: int,
	epochs: int,
	learning_rate: float,
	batch_size: int,
	device: torch.device | str,
) -> tuple[float, float]:
	encoder = SNNFeatureExtractor(
		in_features=in_features, hidden=hidden_features, out_features=out_features
	).to(device)
	encoder.load_state_dict(initial_state)
	classifier = nn.Linear(out_features, n_classes).to(device)
	optimizer = torch.optim.Adam(
		list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate
	)
	loss_function = nn.CrossEntropyLoss()
	for _ in range(epochs):
		encoder.train()
		for indices in torch.randperm(train_x.shape[-1], device=device).split(batch_size):
			loss = loss_function(classifier(encoder(train_x[:, :, indices])), train_y[indices])
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	encoder.eval()
	with torch.no_grad():
		train_accuracy = _accuracy(classifier(encoder(train_x)).argmax(dim=1), train_y)
		eval_accuracy = _accuracy(classifier(encoder(eval_x)).argmax(dim=1), eval_y)
	return train_accuracy, eval_accuracy


def train_phase_a(
	train_spikes: np.ndarray | torch.Tensor,
	train_labels: np.ndarray | torch.Tensor,
	validation_spikes: np.ndarray | torch.Tensor | None = None,
	validation_labels: np.ndarray | torch.Tensor | None = None,
	*,
	subject: int = 1,
	epochs: int = 50,
	learning_rate: float = 1e-3,
	temperature: float = 0.1,
	n_classes: int = 4,
	batch_size: int = 32,
	seed: int = 42,
	validation_fraction: float = 0.2,
	augmentation_probability: float = 0.0,
	compare_linear_head: bool = False,
	model: SNNFeatureExtractor | None = None,
	checkpoint_dir: str | Path = "results/checkpoints",
	device: torch.device | str = "cpu",
	encoder_calibration: object | None = None,  # EncoderCalibration | None
	label_names: list[str] | None = None,
	preprocessing_contract: dict[str, object] | None = None,
) -> PhaseAResult:
	"""Train Phase A and save the frozen extractor and initial prototypes.

	Inputs use the spike encoder layout ``(channels, timesteps, trials)``.
	The labels are mapped to consecutive class indices for CrossEntropyLoss;
	the original sorted label values are returned in ``label_values``.
	No final test data is accepted or evaluated here. Use
	:func:`evaluate_phase_a` after training and model selection.

	Parameters
	----------
	encoder_calibration : EncoderCalibration | None
		Rate-encoding thresholds produced by ``fit_encoder()`` on the training
		split.  When supplied the calibration is serialised into the checkpoint
		so that downstream code can apply identical thresholds to validation,
		test, and online Session 2 data via ``apply_encoder()``.
	"""
	if epochs < 1 or n_classes < 2 or batch_size < 1:
		raise ValueError("epochs, n_classes, and batch_size must be positive")
	if not 0 <= augmentation_probability < 1:
		raise ValueError("augmentation_probability must be in [0, 1)")
	_set_seed(seed)
	train_x = _as_spike_tensor(train_spikes).to(device)
	validation_x = (
		_as_spike_tensor(validation_spikes).to(device)
		if validation_spikes is not None else None
	)
	train_y, label_values = _encode_labels(train_labels)
	if len(label_values) != n_classes:
		raise ValueError(
			f"expected {n_classes} training classes, found {len(label_values)}"
		)
	train_y = train_y.to(device)
	if train_x.shape[-1] != train_y.numel():
		raise ValueError("each spike trial must have exactly one label")
	validation_y = None
	if validation_x is not None:
		if validation_labels is None:
			raise ValueError("validation_labels are required with validation_spikes")
		validation_values = np.asarray(validation_labels).reshape(-1)
		if validation_values.size != validation_x.shape[-1] or not np.isfinite(validation_values).all():
			raise ValueError("validation labels must align with validation spikes")
		if not np.isin(validation_values, label_values).all():
			raise ValueError("validation labels contain classes absent from training labels")
		if validation_x.shape[:2] != train_x.shape[:2]:
			raise ValueError("validation spikes must match training channel and timestep dimensions")
		validation_y = torch.as_tensor(
			np.searchsorted(label_values, validation_values), dtype=torch.long, device=device
		)
	if validation_x is None:
		fit_indices, validation_indices = _stratified_split(train_y, validation_fraction, seed)
		fit_x, fit_y = train_x[:, :, fit_indices], train_y[fit_indices]
		validation_x = train_x[:, :, validation_indices]
		validation_y = train_y[validation_indices]
		validation_size = validation_indices.numel()
	else:
		fit_x, fit_y = train_x, train_y
		validation_size = validation_x.shape[-1]

	network = model or SNNFeatureExtractor(in_features=train_x.shape[0])
	network.to(device)
	initial_state = {name: value.detach().clone() for name, value in network.state_dict().items()}

	losses, train_accuracy = _prototype_training(
		network, fit_x, fit_y, n_classes=n_classes, epochs=epochs,
		learning_rate=learning_rate, temperature=temperature, batch_size=batch_size,
		augmentation_probability=augmentation_probability,
		validation_x=validation_x, validation_y=validation_y,
	)
	with torch.no_grad():
		fit_features = network(fit_x)
		final_prototypes = _class_prototypes(fit_features, fit_y, n_classes)
	validation_accuracy = _evaluate_prototypes(
		network, validation_x, validation_y, final_prototypes, temperature
	)
	linear_train_accuracy = None
	if compare_linear_head:
		linear_train_accuracy, _ = _linear_head_comparison(
			initial_state, fit_x, fit_y, validation_x, validation_y,
			in_features=train_x.shape[0], out_features=network.out_features,
			hidden_features=network.fc1.out_features,
			n_classes=n_classes, epochs=epochs, learning_rate=learning_rate,
			batch_size=batch_size, device=device,
		)

	network.eval()
	with torch.no_grad():
		final_prototypes = _class_prototypes(network(fit_x), fit_y, n_classes)

	for parameter in network.parameters():
		parameter.requires_grad_(False)
	network.eval()

	output_dir = Path(checkpoint_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = output_dir / f"phase_a_subject_{subject}.pt"
	prototype_path  = output_dir / f"prototypes_subject_{subject}.pt"

	# Serialise encoder calibration so val/test/online inference re-uses the
	# exact same per-channel thresholds that were fitted on training data.
	calibration_payload: dict | None = None
	if encoder_calibration is not None:
		from neurostream.data.spike_encoder import EncoderCalibration
		if isinstance(encoder_calibration, EncoderCalibration):
			if train_x.shape[0] != encoder_calibration.channel_count * 3:
				raise ValueError("encoder calibration channel count does not match spike channels")
			calibration_payload = {
				"alpha_thresholds": encoder_calibration.alpha_thresholds,
				"beta_thresholds":  encoder_calibration.beta_thresholds,
				"sfreq":            encoder_calibration.sfreq,
				"n_timesteps":      encoder_calibration.n_timesteps,
				"rate_threshold_pct": encoder_calibration.rate_threshold_pct,
				"channel_count": encoder_calibration.channel_count,
				"filter_order": encoder_calibration.filter_order,
			}
		else:
			raise TypeError("encoder_calibration must be an EncoderCalibration")
	label_names = label_names or [str(value) for value in label_values.tolist()]
	if len(label_names) != n_classes or len(set(label_names)) != n_classes:
		raise ValueError("label_names must contain one unique name per class")
	contract = dict(preprocessing_contract or {})
	contract.setdefault("sampling_frequency_hz", (
		encoder_calibration.sfreq if encoder_calibration is not None else None
	))
	contract.setdefault("encoder_n_timesteps", (
		encoder_calibration.n_timesteps if encoder_calibration is not None else train_x.shape[1]
	))
	contract.setdefault("encoder_rate_threshold_pct", (
		encoder_calibration.rate_threshold_pct if encoder_calibration is not None else None
	))
	contract.setdefault("channel_count", (
		encoder_calibration.channel_count if encoder_calibration is not None else train_x.shape[0] // 3
	))
	contract.setdefault("channel_order", None)
	contract.setdefault("filter_settings", {"l_freq_hz": 4.0, "h_freq_hz": 100.0, "order": 4})
	contract.setdefault("epoch_window_seconds", {"tmin": 2.0, "tmax": 6.0})
	contract.setdefault("normalization", "per-trial, per-channel z-score after baseline centering")
	contract.setdefault("band_definitions_hz", {"alpha": [8.0, 13.0], "beta": [13.0, 30.0], "gamma": [30.0, 100.0]})

	torch.save(
		{
			"model_state_dict": network.state_dict(),
			"subject": subject,
			"seed": seed,
			"train_accuracy": train_accuracy,
			"validation_accuracy": validation_accuracy,
			"validation_size": validation_size,
			"linear_head_train_accuracy": linear_train_accuracy,
			"label_values": label_values.tolist(),
			"label_names": label_names,
			"feature_dim": network.out_features,
			"encoder_calibration": calibration_payload,
			"preprocessing_contract": contract,
		},
		checkpoint_path,
	)
	torch.save(final_prototypes.cpu(), prototype_path)
	logger.info(
		"Phase A subject %s fit_accuracy=%.2f%% validation_accuracy=%.2f%% "
		"prototypes=%s",
		subject,
		train_accuracy * 100,
		validation_accuracy * 100,
		tuple(final_prototypes.shape),
	)
	return PhaseAResult(
		model=network,
		prototypes=final_prototypes.cpu(),
		train_accuracy=train_accuracy,
		validation_accuracy=validation_accuracy,
		validation_size=validation_size,
		train_losses=losses,
		label_values=label_values,
		checkpoint_path=checkpoint_path,
		prototype_path=prototype_path,
		linear_head_train_accuracy=linear_train_accuracy,
		encoder_calibration=encoder_calibration,
	)


def train(*args, **kwargs) -> PhaseAResult:
	"""Compatibility alias for :func:`train_phase_a`."""
	return train_phase_a(*args, **kwargs)


def evaluate_phase_a(
	model: SNNFeatureExtractor,
	prototypes: torch.Tensor,
	eval_spikes: np.ndarray | torch.Tensor,
	eval_labels: np.ndarray | torch.Tensor,
	label_values: np.ndarray | torch.Tensor,
	*,
	temperature: float = 0.1,
	train_spikes: np.ndarray | torch.Tensor | None = None,
	train_labels: np.ndarray | torch.Tensor | None = None,
	linear_head_epochs: int = 50,
	batch_size: int = 32,
	seed: int = 42,
) -> dict[str, float | None]:
	"""Evaluate a frozen Phase A model on data not used for training decisions.

	The prototype and optional linear-head metrics are computed here, after
	training and hyperparameter selection. This function is the only place that
	accepts final evaluation data.
	"""
	eval_x = _as_spike_tensor(eval_spikes)
	values = np.asarray(eval_labels).reshape(-1)
	raw_values = np.asarray(label_values).reshape(-1)
	if eval_x.shape[-1] != values.size or not np.isfinite(values).all():
		raise ValueError("evaluation spikes and labels must be aligned and finite")
	if raw_values.size != prototypes.shape[0] or len(np.unique(raw_values)) != raw_values.size:
		raise ValueError("label_values must match prototype classes")
	if not np.isin(values, raw_values).all():
		raise ValueError("evaluation labels contain classes absent from training")
	encoded = torch.as_tensor(np.searchsorted(raw_values, values), dtype=torch.long)
	model.eval()
	with torch.no_grad():
		features = model(eval_x)
		predictions = prototype_logits(features, prototypes, temperature).argmax(dim=1)
	result: dict[str, float | None] = {
		"prototype_accuracy": _accuracy(predictions, encoded),
		"linear_head_accuracy": None,
	}
	if train_spikes is not None or train_labels is not None:
		if train_spikes is None or train_labels is None:
			raise ValueError("train_spikes and train_labels are required together")
		train_x = _as_spike_tensor(train_spikes)
		train_values = np.asarray(train_labels).reshape(-1)
		if train_x.shape[-1] != train_values.size or not np.isin(train_values, raw_values).all():
			raise ValueError("linear-head training data is invalid or contains unknown labels")
		train_y = torch.as_tensor(np.searchsorted(raw_values, train_values), dtype=torch.long)
		initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
		_, linear_eval = _linear_head_comparison(
			initial_state, train_x, train_y, eval_x, encoded,
			in_features=train_x.shape[0], hidden_features=model.fc1.out_features,
			out_features=model.out_features, n_classes=raw_values.size,
			epochs=linear_head_epochs, learning_rate=1e-3,
			batch_size=batch_size, device=eval_x.device,
		)
		result["linear_head_accuracy"] = linear_eval
	return result


def train_subject(
	subject: int = 1,
	*,
	checkpoint_dir: str | Path = "results/checkpoints",
	epochs: int = 50,
	batch_size: int = 32,
	seed: int = 42,
	learning_rate: float = 1e-3,
	temperature: float = 0.1,
	augmentation_probability: float = 0.2,
	hidden: int = 256,
	validation_fraction: float = 0.2,
	device: torch.device | str = "cpu",
) -> PhaseAResult:
	"""Load, encode, and train one MOABB subject end to end.

	Calibration is fitted exclusively on the training-session data and stored
	with the Phase A checkpoint. Final-session evaluation is performed later by
	``evaluate_phase_a`` using the stored contract.
	"""
	from neurostream.data.loader import load_bnci2014_001
	from neurostream.data.spike_encoder import fit_encoder, apply_encoder

	sessions = load_bnci2014_001(subjects=subject)
	train_name = next((name for name in sessions if "train" in name.lower()), None)
	test_name  = next((name for name in sessions if "test"  in name.lower()), None)
	if train_name is None or test_name is None:
		raise ValueError(f"could not identify train/test sessions: {list(sessions)}")

	train_session = sessions[train_name]
	test_session  = sessions[test_name]

	# Use the true sampling frequency from the raw data — never hard-code 250 Hz.
	train_sfreq = train_session.sfreq
	test_sfreq  = test_session.sfreq
	if abs(train_sfreq - test_sfreq) > 1e-3:
		raise ValueError(
			"Train and test sessions have incompatible sampling frequencies: "
			f"{train_sfreq} Hz vs {test_sfreq} Hz. Resample both sessions before training.",
		)

	# Step 1: fit calibration thresholds on training data ONLY.
	calibration = fit_encoder(train_session.X, sfreq=train_sfreq)

	# Step 2: encode training data using the fitted calibration.
	train_encoded = apply_encoder(train_session.X, calibration, sfreq=train_sfreq)
	train_label_names = {
		int(item["event_code"]): str(item["event"])
		for item in train_session.metadata
		if "event_code" in item and "event" in item
	}
	label_names = [
		train_label_names.get(int(value), str(value))
		for value in np.unique(train_session.y)
	]

	_set_seed(seed)
	model = SNNFeatureExtractor(
		in_features=train_encoded.spikes.shape[0], hidden=hidden
	)
	return train_phase_a(
		train_encoded.spikes,
		train_session.y,
		subject=subject,
		epochs=epochs,
		batch_size=batch_size,
		seed=seed,
		learning_rate=learning_rate,
		temperature=temperature,
		augmentation_probability=augmentation_probability,
		model=model,
		validation_fraction=validation_fraction,
		checkpoint_dir=checkpoint_dir,
		device=device,
		encoder_calibration=calibration,
		label_names=label_names,
		preprocessing_contract={
			"sampling_frequency_hz": train_sfreq,
			"filter_settings": {
				"l_freq_hz": train_session.l_freq,
				"h_freq_hz": train_session.h_freq,
				"order": 4,
			},
			"epoch_window_seconds": {
				"tmin": train_session.tmin,
				"tmax": train_session.tmax,
			},
			"channel_order": list(train_session.channel_names),
		},
	)
