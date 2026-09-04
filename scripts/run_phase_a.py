"""Run supervised Phase A training for one MOABB subject."""

import argparse
import logging

from neurostream.training.phase_a_train import train_subject


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", type=int, default=1)
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--augmentation-probability", type=float, default=0.0)
	parser.add_argument("--hidden", type=int, default=256)
	parser.add_argument("--out-features", type=int, default=512)
	parser.add_argument("--validation-fraction", type=float, default=0.2)
	parser.add_argument("--checkpoint-dir", default="results/checkpoints")
	args = parser.parse_args()
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	result = train_subject(
		subject=args.subject,
		epochs=args.epochs,
		batch_size=args.batch_size,
		seed=args.seed,
		learning_rate=args.learning_rate,
		temperature=args.temperature,
		augmentation_probability=args.augmentation_probability,
		hidden=args.hidden,
		validation_fraction=args.validation_fraction,
		checkpoint_dir=args.checkpoint_dir,
	)
	print(f"subject={args.subject} train_accuracy={result.train_accuracy:.2%} validation_accuracy={result.validation_accuracy:.2%} test_accuracy={result.test_accuracy:.2%}")
	print(f"linear_head_train_accuracy={result.linear_head_train_accuracy:.2%} linear_head_test_accuracy={result.linear_head_test_accuracy:.2%}")
	print(f"checkpoint={result.checkpoint_path}")
	print(f"prototypes={result.prototype_path} shape={tuple(result.prototypes.shape)}")


if __name__ == "__main__":
	main()
