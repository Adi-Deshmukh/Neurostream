from pathlib import Path
import json

from neurostream.dashboard.registry import (
    discover_experiments, load_preprocessing_contract, paired_ablation_rows,
)
from neurostream.evaluation.artifacts import save_evaluation_artifact


def test_discovery_rejects_missing_checkpoint_without_fabricating_metrics(tmp_path):
    root = tmp_path
    manifest_dir = root / "results" / "checkpoints"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "evaluation_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "runs": [{
            "subject": 1, "seed": 42,
            "checkpoint": "results/checkpoints/missing.pt",
            "evaluation_artifact": "results/evaluations/missing.json",
        }],
    }))
    rows = discover_experiments(root)
    assert len(rows) == 1
    assert rows[0]["status"] == "missing_checkpoint"
    assert rows[0]["metrics"] == {}


def test_paired_ablation_joins_exact_subject_seed_pairs():
    report = {
        "conditions": {
            "enabled": {"runs": [
                {"subject": 1, "seed": 42, "target_prequential_acc": 0.6},
                {"subject": 2, "seed": 42, "target_prequential_acc": 0.7},
            ]},
            "disabled": {"runs": [
                {"subject": 1, "seed": 42, "target_prequential_acc": 0.5},
                {"subject": 2, "seed": 123, "target_prequential_acc": 0.4},
            ]},
        }
    }
    rows = paired_ablation_rows(report)
    assert len(rows) == 1
    assert rows[0]["subject"] == 1
    assert rows[0]["seed"] == 42
    assert abs(rows[0]["target_prequential_acc_difference"] - 0.1) < 1e-9


def test_project_root_is_repository_root():
    from neurostream.dashboard.registry import project_root
    assert (project_root() / "pyproject.toml").is_file()


def test_manifest_discovery_includes_legacy_scripts_results_root(tmp_path):
    root = tmp_path
    manifest_dir = root / "scripts" / "results" / "checkpoints"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "evaluation_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "runs": [{
            "subject": 1, "seed": 7,
            "checkpoint": "scripts/results/checkpoints/missing.pt",
            "evaluation_artifact": "scripts/results/checkpoints/missing.json",
        }],
    }))
    rows = discover_experiments(root)
    assert len(rows) == 1
    assert rows[0]["seed"] == 7
    assert rows[0]["status"] == "missing_checkpoint"


def test_discovery_accepts_checkpoint_hash_matched_evaluation(tmp_path):
    root = tmp_path
    checkpoint = root / "results" / "checkpoints" / "sub_1" / "seed_42" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-fixture")
    evaluation = root / "results" / "evaluations" / "sub_1" / "seed_42" / "run.json"
    save_evaluation_artifact(
        evaluation,
        checkpoint_path=checkpoint,
        subject=1,
        train_session="session_T",
        evaluation_session="session_E",
        metrics={
            "target_static_baseline_acc": 0.51,
            "target_prequential_acc": 0.56,
            "target_retrospective_acc": 0.58,
            "source_accuracy_after_adaptation": 0.74,
        },
        label_values=[1, 2],
        label_names=["Left", "Right"],
        run_config={"seed": 42},
        run_id="fixture-s1-seed42",
    )
    manifest_dir = root / "results" / "checkpoints"
    (manifest_dir / "evaluation_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "runs": [{
            "subject": 1, "seed": 42,
            "checkpoint": str(checkpoint.relative_to(root)),
            "evaluation_artifact": str(evaluation.relative_to(root)),
        }],
    }))
    rows = discover_experiments(root)
    assert len(rows) == 1
    assert rows[0]["status"] == "valid"
    assert rows[0]["artifact"]["run_id"] == "fixture-s1-seed42"
    assert rows[0]["metrics"]["target_prequential_acc"] == 0.56


def test_registry_lists_checkpoint_without_claiming_evaluation(tmp_path):
    root = tmp_path
    checkpoint = root / "scripts" / "results" / "checkpoints" / "phase_a_subject_1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"phase-a-checkpoint")
    rows = discover_experiments(root)
    assert len(rows) == 1
    assert rows[0]["subject"] == 1
    assert rows[0]["seed"] is None
    assert rows[0]["status"] == "checkpoint_without_manifest"
    assert rows[0]["metrics"] == {}


def test_checkpoint_preprocessing_contract_is_read_without_model_construction(tmp_path):
    import hashlib
    import torch

    checkpoint = tmp_path / "phase_a_subject_1.pt"
    expected = {
        "sampling_frequency_hz": 250.0,
        "epoch_window_seconds": {"tmin": 2.0, "tmax": 6.0},
        "normalization": {"method": "per_trial_demean_preserve_channel_scale"},
    }
    torch.save({
        "model_state_dict": {"weight": torch.ones(2)},
        "preprocessing_contract": expected,
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert load_preprocessing_contract(checkpoint, digest) == expected
