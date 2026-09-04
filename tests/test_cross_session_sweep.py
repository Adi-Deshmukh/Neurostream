"""Tests for reproducible seeding and cross-session sweep execution."""

import csv
from pathlib import Path
import random
import numpy as np
import pytest
import torch

from neurostream.evaluation.cross_session_eval import run_cross_session_sweep, run_subject_cross_session
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
        hidden=32,
    )

    assert len(summary.runs) == 1
    run = summary.runs[0]
    assert run.subject == 1
    assert run.seed == 42
    assert 0.0 <= run.post_adaptation_acc <= 1.0
    assert 0.0 <= run.spike_sparsity <= 1.0
    assert csv_file.exists()

    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert "subject" in rows[0]
        assert "post_adaptation_acc" in rows[0]
        assert "spike_sparsity" in rows[0]
        assert "consolidation_phases" in rows[0]
