"""Serialization and validation for held-out evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def checkpoint_hash(path: str | Path) -> str:
    """Return the SHA-256 identity of an exact checkpoint file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_evaluation_artifact(
    path: str | Path,
    *,
    checkpoint_path: str | Path,
    subject: int,
    train_session: str,
    evaluation_session: str,
    metrics: dict[str, Any],
    label_values: list[int],
    label_names: list[str],
    run_config: dict[str, Any],
) -> Path:
    """Write a self-identifying held-out evaluation result."""
    checkpoint_path = Path(checkpoint_path)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_hash(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "subject": subject,
        "train_session": train_session,
        "evaluation_session": evaluation_session,
        "split_identity": "held_out_evaluation",
        "protocol": "static_baseline_prequential_adaptation_retrospective_diagnostic",
        "label_values": [int(value) for value in label_values],
        "label_names": list(label_names),
        "metrics": metrics,
        "run_config": run_config,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return output


def load_evaluation_artifact(
    path: str | Path,
    *,
    checkpoint_path: str | Path,
    subject: int | None = None,
) -> dict[str, Any]:
    """Load and reject artifacts that do not match the selected checkpoint."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "checkpoint_sha256", "subject", "train_session",
        "evaluation_session", "split_identity", "protocol", "label_values",
        "label_names", "metrics", "run_config",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"evaluation artifact is missing fields: {sorted(missing)}")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported evaluation artifact schema")
    if artifact["split_identity"] != "held_out_evaluation":
        raise ValueError("artifact is not a held-out evaluation")
    if artifact["checkpoint_sha256"] != checkpoint_hash(checkpoint_path):
        raise ValueError("evaluation artifact does not match the selected checkpoint")
    if subject is not None and artifact["subject"] != subject:
        raise ValueError("evaluation artifact subject does not match selection")
    if len(artifact["label_values"]) != len(artifact["label_names"]):
        raise ValueError("evaluation artifact label mapping is invalid")
    if not isinstance(artifact["metrics"], dict):
        raise ValueError("evaluation artifact metrics must be an object")
    return artifact
