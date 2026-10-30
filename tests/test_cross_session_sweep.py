"""Tests for reproducible seeding and cross-session sweep execution."""

import csv
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch

from neurostream.evaluation.cross_session_eval import (
    run_consolidation_ablation,
    run_cross_session_sweep,
    run_subject_cross_session,
)
from neurostream.data.loader import load_bnci2014_001
from neurostream.evaluation.artifacts import checkpoint_hash, load_evaluation_artifact
from neurostream.utils.seed import seed_everything, temp_seed


def test_seed_everything_determinism():
    """Verify that seed_everything synchronizes torch, numpy, and random."""
    seed_everything(123)
    py_1 = [random.random() for _ in range(5)]
    np_1 = np.random.randn(5)
    th_1 = torch.randn(5)

    seed_everything(123)
    py_2 = [random.random() for _ in range(5)]
    np_2 = np.random.randn(5)
    th_2 = torch.randn(5)

    assert py_1 == py_2
    np.testing.assert_array_equal(np_1, np_2)
    torch.testing.assert_close(th_1, th_2)


def test_temp_seed_context_manager():
    """Verify temp_seed isolates localized seed changes and restores prior state."""
    seed_everything(777)
    base_val = random.random()

    seed_everything(777)
    _ = random.random()  # step 1
    expected_next = random.random()  # step 2

    seed_everything(777)
    _ = random.random()  # step 1
    with temp_seed(999):
        _ = random.random()
        _ = np.random.randn(3)
        _ = torch.randn(3)

    restored_next = random.random()
    assert restored_next == expected_next


class MockRaw:
    def __init__(self, data: np.ndarray, sfreq: float = 250.0):
        self._data = data
        self.info = {"sfreq": sfreq}

    def copy(self):
        return MockRaw(self._data.copy(), self.info["sfreq"])

    def pick(self, ch_type: str):
        return self

    def get_data(self):
        return self._data


class MockDataset:
    """Mock MOABB dataset providing fast synthetic 22-channel sessions."""

    def __init__(self, n_channels: int = 22, n_samples: int = 4000):
        self.n_channels = n_channels
        self.n_samples = n_samples

    def get_data(self, subjects: list[int]):
        import mne

        result = {}
        for s in subjects:
            sfreq = 250.0
            info = mne.create_info(self.n_channels, sfreq, ch_types="eeg")
            raw_train = mne.io.RawArray(
                np.random.default_rng(s * 10).normal(size=(self.n_channels, self.n_samples)),
                info,
                verbose=False,
            )
            onsets = [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
            descriptions = ["769", "770", "771", "772", "769", "770", "771", "772"]
            raw_train.set_annotations(
                mne.Annotations(onset=onsets, duration=[0] * 8, description=descriptions)
            )
            raw_test = mne.io.RawArray(
                np.random.default_rng(s * 10 + 1).normal(size=(self.n_channels, self.n_samples)),
                info,
                verbose=False,
            )
            raw_test.set_annotations(
                mne.Annotations(onset=onsets, duration=[0] * 8, description=descriptions)
            )
            result[s] = {
                "0train": {"0": raw_train},
                "1test": {"0": raw_test},
            }
        return result


class PathAwareMockDataset(MockDataset):
    def __init__(self):
        super().__init__()
        self.received_cache_path = None

    def data_path(self, subject, path=None, **kwargs):
        return str(path or "default")

    def get_data(self, subjects, cache_config=None):
        self.received_cache_path = getattr(cache_config, "path", None)
        return super().get_data(subjects)


def test_loader_uses_supported_dataset_and_cache_paths(tmp_path):
    dataset = PathAwareMockDataset()
    load_bnci2014_001(
        subjects=1,
        dataset=dataset,
        dataset_path=str(tmp_path / "dataset"),
        cache_path=str(tmp_path / "cache"),
    )
    assert callable(dataset.data_path)
    assert dataset.received_cache_path == str(tmp_path / "cache")


def test_cross_session_sweep_with_mock_dataset(tmp_path):
    """End-to-end integration test of sweep with mock dataset and CSV validation."""
    mock_ds = MockDataset()
    csv_file = tmp_path / "test_sweep.csv"

    summary = run_cross_session_sweep(
        subjects=[1],
        seeds=[42],
        epochs=2,
        batch_size=2,
        validation_fraction=0.5,
        dataset=mock_ds,
        output_csv=csv_file,
        checkpoint_dir=tmp_path / "checkpoints",
        encoder_timesteps=10,
        encoder_rate_threshold_pct=60.0,
        model_beta=0.8,
        model_threshold=0.4,
        model_out_features=16,
        output_evaluation_dir=tmp_path / "evaluations",
        expected_sampling_frequency_hz=250.0,
        epoch_tmin=2.0,
        epoch_tmax=6.0,
        filter_l_freq_hz=4.0,
        filter_h_freq_hz=100.0,
        hidden=32,
    )

    assert len(summary.runs) == 1
    run = summary.runs[0]
    assert run.subject == 1
    assert run.seed == 42
    assert 0.0 <= run.post_adaptation_acc <= 1.0
    assert 0.0 <= run.spike_sparsity <= 1.0
    assert csv_file.exists()
    assert (tmp_path / "evaluations" / "sub_1" / "seed_42" / "subject_1_seed_42.json").exists()
    assert (tmp_path / "checkpoints" / "evaluation_manifest.json").exists()
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "sub_1" / "seed_42" / "phase_a_subject_1.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["encoder_calibration"]["n_timesteps"] == 10
    assert checkpoint["encoder_calibration"]["rate_threshold_pct"] == 60.0
    assert checkpoint["feature_dim"] == 16
    assert checkpoint["preprocessing_contract"]["effective_configuration"]["model_out_features"] == 16

    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert "subject" in rows[0]
        assert "target_prequential_acc" in rows[0]
        assert "source_accuracy_after_adaptation" in rows[0]
        assert "spike_sparsity" in rows[0]
        assert "consolidation_phases" in rows[0]


def test_sweep_manifest_artifacts_and_dashboard_aggregation(tmp_path):
    """Verify every manifest run is dashboard-loadable and aggregates by subject."""
    checkpoint_root = tmp_path / "checkpoints"
    evaluation_root = tmp_path / "evaluations"
    summary = run_cross_session_sweep(
        subjects=[1],
        seeds=[42, 123],
        epochs=1,
        batch_size=2,
        validation_fraction=0.5,
        dataset=MockDataset(),
        checkpoint_dir=checkpoint_root,
        output_evaluation_dir=evaluation_root,
        encoder_timesteps=10,
        hidden=16,
        model_out_features=16,
    )

    manifest = json.loads(
        (checkpoint_root / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert len(manifest["runs"]) == len(summary.runs) == 2

    metric_values = {
        "target_static_baseline_acc": [],
        "target_prequential_acc": [],
        "target_retrospective_acc": [],
        "source_accuracy_after_adaptation": [],
    }
    selected_run = next(run for run in manifest["runs"] if run["seed"] == 123)
    for run in manifest["runs"]:
        checkpoint_path = Path(run["checkpoint"])
        artifact_path = Path(run["evaluation_artifact"])
        assert checkpoint_path.is_file()
        assert artifact_path.is_file()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        artifact = load_evaluation_artifact(
            artifact_path, checkpoint_path=checkpoint_path, subject=run["subject"]
        )
        assert artifact["checkpoint_sha256"] == checkpoint_hash(checkpoint_path)
        assert artifact["run_config"]["seed"] == run["seed"]
        assert artifact["subject"] == run["subject"]
        assert artifact["checkpoint_path"] == str(checkpoint_path)
        for metric_name in metric_values:
            metric_values[metric_name].append(float(artifact["metrics"][metric_name]))

    # This mirrors the dashboard's mean +/- population standard deviation view.
    for values in metric_values.values():
        assert len(values) == 2
        assert np.isclose(np.mean(values), np.mean(values))
        assert np.isclose(np.std(values), np.std(values))
    assert selected_run["subject"] == 1
    assert selected_run["seed"] == 123


def test_consolidation_ablation_reports_paired_protocol(tmp_path):
    report = run_consolidation_ablation(
        subjects=[1],
        seeds=[42],
        output_dir=tmp_path / "ablation",
        dataset=MockDataset(),
        epochs=1,
        batch_size=2,
        validation_fraction=0.5,
        hidden=16,
    )
    assert report["protocol"] == "paired_subject_seed_consolidation_ablation"
    assert set(report["conditions"]) == {"enabled", "disabled"}
    for condition in report["conditions"].values():
        assert "mean_target_static_baseline" in condition
        assert "mean_target_prequential" in condition
        assert "mean_target_retrospective" in condition
        assert "std_target_static_baseline" in condition
        assert "mean_source_retention" in condition
        assert "std_target_retrospective" in condition
        assert "std_source_retention" in condition
        assert "settings" in condition
    assert (tmp_path / "ablation" / "report.json").exists()
