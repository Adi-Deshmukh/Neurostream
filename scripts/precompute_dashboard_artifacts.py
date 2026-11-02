"""Precompute feature-only replay artifacts for fast dashboard startup.

Run from the repository root:
    python scripts/precompute_dashboard_artifacts.py --subject 1 --projection PCA

The resulting cache contains model features and projection metadata only. It does
not persist raw EEG epochs or encoded spike tensors.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurostream.dashboard.simulation import TrialReplayEngine


def _default_checkpoint_dir() -> Path:
    candidates = (
        PROJECT_ROOT / "results" / "checkpoints",
        PROJECT_ROOT / "scripts" / "results" / "checkpoints",
    )
    for path in candidates:
        if (path / "phase_a_subject_1.pt").is_file():
            return path
        if any(path.glob("sub_*/seed_*/phase_a_subject_*.pt")):
            return path
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True, choices=range(1, 10))
    parser.add_argument("--projection", choices=("PCA", "UMAP"), default="PCA")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prototypes", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "dashboard_cache")
    args = parser.parse_args()

    checkpoint_dir = _default_checkpoint_dir()
    checkpoint = (args.checkpoint or checkpoint_dir / f"phase_a_subject_{args.subject}.pt").resolve()
    prototypes = (args.prototypes or checkpoint_dir / f"prototypes_subject_{args.subject}.pt").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not prototypes.is_file():
        raise FileNotFoundError(f"prototypes not found: {prototypes}")

    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    prototype_hash = hashlib.sha256(prototypes.read_bytes()).hexdigest()
    artifact_dir = args.output_root.resolve() / (
        f"subject_{args.subject}_{checkpoint.stem}_"
        f"{checkpoint_hash[:12]}_{prototype_hash[:12]}_{args.projection.lower()}"
    )
    print(f"Preparing subject S{args.subject:02d} ({args.projection})")
    print("This one-time operation loads the dataset, applies the checkpoint encoder, and extracts SNN features.")
    engine = TrialReplayEngine.from_checkpoint(
        checkpoint_path=checkpoint,
        prototypes_path=prototypes,
        subject=args.subject,
        projection_method=args.projection,
    )
    engine.save_prepared_artifact(artifact_dir)
    print(f"Prepared replay cache: {artifact_dir}")
    print("Stored: SNN features, labels, prototypes, projection, measured activity and provenance.")
    print("Not stored: raw EEG epochs or encoded spike tensors.")


if __name__ == "__main__":
    main()
