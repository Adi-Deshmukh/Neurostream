from pathlib import Path
import json

from neurostream.dashboard.registry import discover_experiments, paired_ablation_rows


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
    assert rows[0]["target_prequential_acc_difference"] == 0.1


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
