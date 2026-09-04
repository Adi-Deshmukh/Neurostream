"""Evaluate frozen Phase A SNN with target-session adaptation."""

import argparse
import pickle

import torch

from neurostream.data.loader import load_bnci2014_001
from neurostream.data.spike_encoder import encode
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.training.phase_b_adapt import adapt_target_session, calibrate_target_session


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", type=int, default=1)
	parser.add_argument("--checkpoint", default="results/checkpoints/phase_a_subject_1.pt")
	parser.add_argument("--prototypes", default="results/checkpoints/prototypes_subject_1.pt")
	parser.add_argument("--confidence-threshold", type=float, default=0.75)
	parser.add_argument("--momentum", type=float, default=0.70)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--calibration-fraction", type=float, default=0.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--enable-consolidation", action="store_true", default=False)
	parser.add_argument("--no-consolidation", action="store_false", dest="enable_consolidation")
	parser.add_argument("--consolidation-interval", type=int, default=50)
	parser.add_argument("--sleep-steps", type=int, default=5)
	parser.add_argument("--noise-std", type=float, default=0.05)
	parser.add_argument("--pull-rate", type=float, default=0.5)
	args = parser.parse_args()

	try:
		checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
	except pickle.UnpicklingError:
		checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
	sessions = load_bnci2014_001(subjects=args.subject)
	train_name = next(name for name in sessions if "train" in name.lower())
	test_name = next(name for name in sessions if "test" in name.lower())
	train_spikes = encode(sessions[train_name].X).spikes
	test_spikes = encode(sessions[test_name].X).spikes
	model = SNNFeatureExtractor(
		in_features=train_spikes.shape[0],
		out_features=int(checkpoint["feature_dim"]),
	)
	model.load_state_dict(checkpoint["model_state_dict"])
	prototypes = torch.load(args.prototypes, map_location="cpu", weights_only=True)
	label_values = torch.as_tensor(checkpoint["label_values"])
	test_labels = torch.as_tensor(
		[torch.where(label_values == int(label))[0].item() for label in sessions[test_name].y],
		dtype=torch.long,
	)
	train_labels = torch.as_tensor(
		[torch.where(label_values == int(label))[0].item() for label in sessions[train_name].y],
		dtype=torch.long,
	)
	if args.calibration_fraction > 0:
		result = calibrate_target_session(
			model, prototypes, test_spikes, test_labels,
			calibration_fraction=args.calibration_fraction,
			momentum=args.momentum, batch_size=args.batch_size, seed=args.seed,
		)
	else:
		result = adapt_target_session(
			model, prototypes, test_spikes, test_labels,
			momentum=args.momentum, confidence_threshold=args.confidence_threshold,
			batch_size=args.batch_size, source_spikes=train_spikes,
			source_labels=train_labels,
			enable_consolidation=args.enable_consolidation,
			consolidation_interval=args.consolidation_interval,
			sleep_steps=args.sleep_steps,
			noise_std=args.noise_std,
			pull_rate=args.pull_rate,
		)
	print(f"subject={args.subject} session={test_name}")
	print(f"consolidation_enabled={args.enable_consolidation}")
	print(f"before_accuracy={result.before_accuracy:.2%}")
	print(f"after_accuracy={result.after_accuracy:.2%}")
	print(f"session_1_retention={result.retention_accuracy:.2%}")
	print(f"accepted_trials={result.accepted_trials}/{test_spikes.shape[-1]}")
	print(f"acceptance_rate={result.acceptance_rate:.2%}")
	if result.consolidation_phases:
		print(f"consolidation_phases={result.consolidation_phases}")
		print(f"drift_loss={result.drift_before_after:.4f}")
	if result.calibration_trials:
		print(f"calibration_trials={result.calibration_trials} evaluation_trials={result.evaluation_trials}")
	print(f"prototypes_shape={tuple(result.prototypes.shape)}")


if __name__ == "__main__":
	main()
