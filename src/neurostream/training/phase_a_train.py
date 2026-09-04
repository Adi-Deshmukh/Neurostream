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
	test_accuracy: float
	train_accuracy: float
	validation_accuracy: float
	validation_size: int
	train_losses: list[float]
	label_values: np.ndarray
	checkpoint_path: Path | None
	prototype_path: Path | None
	linear_head_test_accuracy: float | None = None
	linear_head_train_accuracy: float | None = None


def _as_spike_tensor(spikes: np.ndarray | torch.Tensor) -> torch.Tensor:
	tensor = torch.as_tensor(spikes, dtype=torch.float32)
	if tensor.ndim != 3:
		raise ValueError("spikes must have shape (channels, timesteps, trials)")
	return tensor


def _encode_labels(labels: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
	values = np.asarray(labels).reshape(-1)
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
			if getattr(network, "spatial_layer", None) is not None:
				w = network.spatial_layer.spatial_conv.weight.squeeze(-1)
				w_norm = F.normalize(w, dim=-1)
				gram = w_norm @ w_norm.T
				eye = torch.eye(gram.shape[0], device=gram.device)
				loss = loss + 0.05 * F.mse_loss(gram, eye)
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
	test_x: torch.Tensor,
	test_y: torch.Tensor,
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
		test_accuracy = _accuracy(classifier(encoder(test_x)).argmax(dim=1), test_y)
	return train_accuracy, test_accuracy


def train_phase_a(
	train_spikes: np.ndarray | torch.Tensor,
	train_labels: np.ndarray | torch.Tensor,
	test_spikes: np.ndarray | torch.Tensor,
	test_labels: np.ndarray | torch.Tensor,
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
	compare_linear_head: bool = True,
	model: SNNFeatureExtractor | None = None,
	checkpoint_dir: str | Path = "results/checkpoints",
	device: torch.device | str = "cpu",
) -> PhaseAResult:
	"""Train Phase A and save the frozen extractor and initial prototypes.

	Inputs use the spike encoder layout ``(channels, timesteps, trials)``.
	The labels are mapped to consecutive class indices for CrossEntropyLoss;
	the original sorted label values are returned in ``label_values``.
	"""
	if epochs < 1 or n_classes < 2 or batch_size < 1:
		raise ValueError("epochs, n_classes, and batch_size must be positive")
	if not 0 <= augmentation_probability < 1:
		raise ValueError("augmentation_probability must be in [0, 1)")
	_set_seed(seed)
	train_x = _as_spike_tensor(train_spikes).to(device)
	test_x = _as_spike_tensor(test_spikes).to(device)
	train_y, label_values = _encode_labels(train_labels)
	if len(label_values) != n_classes:
		raise ValueError(
			f"expected {n_classes} training classes, found {len(label_values)}"
		)
	test_values = np.asarray(test_labels).reshape(-1)
	if not np.isin(test_values, label_values).all():
		raise ValueError("test labels contain classes absent from training labels")
	test_y = torch.as_tensor(
		np.searchsorted(label_values, test_values), dtype=torch.long, device=device
	)
	train_y = train_y.to(device)
	if train_x.shape[-1] != train_y.numel() or test_x.shape[-1] != test_y.numel():
		raise ValueError("each spike trial must have exactly one label")
	fit_indices, validation_indices = _stratified_split(
		train_y, validation_fraction, seed
	)
	fit_x, fit_y = train_x[:, :, fit_indices], train_y[fit_indices]
	validation_x = train_x[:, :, validation_indices]
	validation_y = train_y[validation_indices]

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
	linear_train_accuracy = linear_test_accuracy = None
	if compare_linear_head:
		linear_train_accuracy, linear_test_accuracy = _linear_head_comparison(
			initial_state, fit_x, fit_y, test_x, test_y,
			in_features=train_x.shape[0], out_features=network.out_features,
			hidden_features=network.fc1.out_features,
			n_classes=n_classes, epochs=epochs, learning_rate=learning_rate,
			batch_size=batch_size, device=device,
		)

	network.eval()
	with torch.no_grad():
		final_prototypes = _class_prototypes(network(fit_x), fit_y, n_classes)
		test_features = network(test_x)
		predictions = prototype_logits(test_features, final_prototypes, temperature).argmax(dim=1)
		test_accuracy = float((predictions == test_y).float().mean().cpu())

	for parameter in network.parameters():
		parameter.requires_grad_(False)
	network.eval()

	output_dir = Path(checkpoint_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = output_dir / f"phase_a_subject_{subject}.pt"
	prototype_path = output_dir / f"prototypes_subject_{subject}.pt"
	torch.save(
		{
			"model_state_dict": network.state_dict(),
			"subject": subject,
			"seed": seed,
			"test_accuracy": test_accuracy,
			"train_accuracy": train_accuracy,
			"validation_accuracy": validation_accuracy,
			"validation_size": validation_indices.numel(),
			"linear_head_test_accuracy": linear_test_accuracy,
			"linear_head_train_accuracy": linear_train_accuracy,
			"label_values": label_values.tolist(),
			"feature_dim": network.out_features,
		},
		checkpoint_path,
	)
	torch.save(final_prototypes.cpu(), prototype_path)
	logger.info(
		"Phase A subject %s fit_accuracy=%.2f%% validation_accuracy=%.2f%% "
		"test_accuracy=%.2f%%; prototypes=%s",
		subject,
		train_accuracy * 100,
		validation_accuracy * 100,
		test_accuracy * 100,
		tuple(final_prototypes.shape),
	)
	return PhaseAResult(
		model=network,
		prototypes=final_prototypes.cpu(),
		test_accuracy=test_accuracy,
		train_accuracy=train_accuracy,
		validation_accuracy=validation_accuracy,
		validation_size=validation_indices.numel(),
		train_losses=losses,
		label_values=label_values,
		checkpoint_path=checkpoint_path,
		prototype_path=prototype_path,
		linear_head_test_accuracy=linear_test_accuracy,
		linear_head_train_accuracy=linear_train_accuracy,
	)


def train(*args, **kwargs) -> PhaseAResult:
	"""Compatibility alias for :func:`train_phase_a`."""
	return train_phase_a(*args, **kwargs)


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
	"""Load, encode, and train one MOABB subject end to end."""
	from neurostream.data.loader import load_bnci2014_001
	from neurostream.data.spike_encoder import encode

	sessions = load_bnci2014_001(subjects=subject)
	train_name = next((name for name in sessions if "train" in name.lower()), None)
	test_name = next((name for name in sessions if "test" in name.lower()), None)
	if train_name is None or test_name is None:
		raise ValueError(f"could not identify train/test sessions: {list(sessions)}")

	train_session = sessions[train_name]
	test_session = sessions[test_name]
	train_encoded = encode(train_session.X)
	test_encoded = encode(test_session.X)
	_set_seed(seed)
	model = SNNFeatureExtractor(
		in_features=train_encoded.spikes.shape[0], hidden=hidden, out_features=512
	)
	return train_phase_a(
		train_encoded.spikes,
		train_session.y,
		test_encoded.spikes,
		test_session.y,
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
	)
