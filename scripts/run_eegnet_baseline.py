"""Train an EEGNet baseline on subject 1 using the Phase A split."""

import argparse

from neurostream.data.loader import load_bnci2014_001
from neurostream.models.eegnet_baseline import train_eegnet_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--checkpoint", default="results/checkpoints/eegnet_subject_1.pt")
    args = parser.parse_args()

    sessions = load_bnci2014_001(subjects=args.subject)
    train_name = next(name for name in sessions if "train" in name.lower())
    test_name = next(name for name in sessions if "test" in name.lower())
    train_session, test_session = sessions[train_name], sessions[test_name]
    result = train_eegnet_baseline(
        train_session.X,
        train_session.y,
        test_session.X,
        test_session.y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    print(f"subject={args.subject}")
    print(f"train_accuracy={result.train_accuracy:.2%}")
    print(f"validation_accuracy={result.validation_accuracy:.2%}")
    print(f"test_accuracy={result.test_accuracy:.2%}")
    print(f"checkpoint={result.checkpoint_path}")


if __name__ == "__main__":
    main()
