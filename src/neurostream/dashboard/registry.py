"""Read-only discovery and validation for NeuroStream dashboard artifacts.

This module deliberately performs no model inference, EEG preprocessing, or network
access. It reads local manifests and validates evaluation JSON against checkpoint
SHA-256 identities before exposing metrics to the UI.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from neurostream.evaluation.artifacts import load_evaluation_artifact


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_artifact_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    options = (root / candidate, root / "scripts" / candidate)
    return next((item.resolve() for item in options if item.exists()), options[0].resolve())


def load_preprocessing_contract(
    checkpoint_path: Path,
    expected_sha256: str | None = None,
    expected_subject: int | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Read checkpoint metadata without constructing a model or loading custom objects."""
    import hashlib
    import torch

    checkpoint_path = Path(checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("checkpoint SHA-256 changed since the registry was validated")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a state dictionary")
    if expected_subject is not None and int(payload.get("subject", -1)) != expected_subject:
        raise ValueError("checkpoint subject does not match the selected experiment")
    if expected_seed is not None and int(payload.get("seed", -1)) != expected_seed:
        raise ValueError("checkpoint seed does not match the selected experiment")
    contract = payload.get("preprocessing_contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint does not contain a preprocessing contract")
    return contract


def _manifest_paths(root: Path) -> list[Path]:
    bases = (root / "results", root / "scripts" / "results")
    found = set()
    for base in bases:
        if base.exists():
            found.update(
                path.resolve()
                for path in base.rglob("evaluation_manifest.json")
                if path.is_file()
            )
    return sorted(found, key=lambda path: str(path))


def discover_experiments(root: Path | None = None) -> list[dict[str, Any]]:
    """Discover manifest runs and validate every evaluation/checkpoint pair."""
    root = (root or project_root()).resolve()
    discovered: dict[tuple[int, int, str], dict[str, Any]] = {}
    for manifest_path in _manifest_paths(root):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest root must be a JSON object")
            runs = manifest.get("runs", [])
            if not isinstance(runs, list):
                raise ValueError("manifest 'runs' must be a list")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            discovered[(0, 0, str(manifest_path))] = {
                "subject": None, "seed": None, "manifest_path": manifest_path,
                "checkpoint_path": None, "evaluation_path": None,
                "status": "invalid_manifest", "error": str(exc), "artifact": None,
                "metrics": {},
            }
            continue

        for run in runs:
            try:
                subject, seed = int(run["subject"]), int(run["seed"])
                checkpoint = resolve_artifact_path(root, run["checkpoint"])
                evaluation = resolve_artifact_path(root, run["evaluation_artifact"])
                key = (subject, seed, str(checkpoint))
                row: dict[str, Any] = {
                    "subject": subject, "seed": seed, "manifest_path": manifest_path,
                    "checkpoint_path": checkpoint, "evaluation_path": evaluation,
                    "status": "missing_checkpoint", "error": None, "artifact": None,
                    "metrics": {},
                }
                if not checkpoint.is_file():
                    row["error"] = f"Checkpoint not found: {checkpoint}"
                elif not evaluation.is_file():
                    row["status"] = "missing_evaluation"
                    row["error"] = f"Evaluation artifact not found: {evaluation}"
                else:
                    try:
                        artifact = load_evaluation_artifact(
                            evaluation, checkpoint_path=checkpoint, subject=subject
                        )
                        config = artifact.get("run_config", {})
                        if int(config.get("seed", seed)) != seed:
                            raise ValueError("evaluation seed does not match manifest")
                        row.update(
                            status="valid", artifact=artifact,
                            metrics=dict(artifact.get("metrics", {})),
                            label_names=list(artifact.get("label_names", [])),
                            run_config=config,
                        )
                    except Exception as exc:
                        row["status"] = "invalid_artifact"
                        row["error"] = str(exc)
                discovered[key] = row
            except (KeyError, TypeError, ValueError) as exc:
                marker = (0, 0, f"{manifest_path}:{len(discovered)}")
                discovered[marker] = {
                    "subject": None, "seed": None, "manifest_path": manifest_path,
                    "checkpoint_path": None, "evaluation_path": None,
                    "status": "invalid_manifest_entry", "error": str(exc),
                    "artifact": None, "metrics": {},
                }

    # Surface orphaned/legacy checkpoints too, without treating them as evaluated runs.
    referenced = {
        str(row["checkpoint_path"])
        for row in discovered.values()
        if row.get("checkpoint_path") is not None
    }
    for checkpoint_root in (root / "results" / "checkpoints", root / "scripts" / "results" / "checkpoints"):
        if not checkpoint_root.exists():
            continue
        for checkpoint in checkpoint_root.rglob("phase_a_subject_*.pt"):
            resolved = checkpoint.resolve()
            if str(resolved) in referenced:
                continue
            match = re.search(r"phase_a_subject_(\d+)\.pt$", checkpoint.name)
            if not match:
                continue
            subject = int(match.group(1))
            seed_match = re.search(r"seed_(\d+)", str(checkpoint.parent))
            seed = int(seed_match.group(1)) if seed_match else None
            key = (subject, seed if seed is not None else -1, str(resolved))
            discovered[key] = {
                "subject": subject, "seed": seed, "manifest_path": None,
                "checkpoint_path": resolved, "evaluation_path": None,
                "status": "checkpoint_without_manifest",
                "error": "Checkpoint exists, but no manifest-linked evaluation artifact was found.",
                "artifact": None, "metrics": {},
            }

    return sorted(
        discovered.values(),
        key=lambda row: (
            row["subject"] if row["subject"] is not None else 99,
            row["seed"] if row["seed"] is not None else 999999,
            str(row["checkpoint_path"] or ""),
        ),
    )


def discover_ablation_reports(root: Path | None = None) -> list[dict[str, Any]]:
    """Find paired consolidation reports; ignore per-run evaluation JSON files."""
    root = (root or project_root()).resolve()
    base = root / "results" / "ablations" / "consolidation"
    reports: list[dict[str, Any]] = []
    if not base.exists():
        return reports
    for path in base.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and (
            payload.get("protocol") == "paired_subject_seed_consolidation_ablation"
            and isinstance(payload.get("conditions"), dict)
            and {"enabled", "disabled"}.issubset(payload["conditions"])
        ):
            reports.append({"path": path, "payload": payload})
    return sorted(reports, key=lambda item: str(item["path"]))


def paired_ablation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return paired enabled/disabled rows, keyed by exact subject and seed."""
    conditions = report.get("conditions", {})
    by_condition: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for name in ("enabled", "disabled"):
        rows = conditions.get(name, {}).get("runs", [])
        by_condition[name] = {
            (int(row["subject"]), int(row["seed"])): row
            for row in rows
            if "subject" in row and "seed" in row
        }
    metrics = (
        "target_static_baseline_acc", "target_prequential_acc",
        "target_retrospective_acc", "source_accuracy_after_adaptation",
        "source_retention_delta", "adaptation_gain", "spike_sparsity",
    )
    paired = []
    for key in sorted(set(by_condition["enabled"]) & set(by_condition["disabled"])):
        on, off = by_condition["enabled"][key], by_condition["disabled"][key]
        item: dict[str, Any] = {"subject": key[0], "seed": key[1]}
        for metric in metrics:
            a, b = on.get(metric), off.get(metric)
            item[f"{metric}_enabled"] = a
            item[f"{metric}_disabled"] = b
            item[f"{metric}_difference"] = (
                float(a) - float(b)
                if isinstance(a, (int, float)) and isinstance(b, (int, float))
                else None
            )
        item["consolidation_phases"] = on.get("consolidation_phases")
        paired.append(item)
    return paired
