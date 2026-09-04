"""Focused tests for Phase A training artifacts."""

import numpy as np
import torch

from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.training.phase_a_train import train_phase_a


def test_phase_a_trains_freezes_and_saves_artifacts(tmp_path):
    rng = np.random.default_rng(11)
    train_spikes = (rng.random((66, 5, 8)) > 0.75).astype(np.uint8)
    test_spikes = (rng.random((66, 5, 4)) > 0.75).astype(np.uint8)
    train_labels = np.array([10, 20, 30, 40, 10, 20, 30, 40])
    test_labels = np.array([10, 20, 30, 40])
    model = SNNFeatureExtractor(in_features=66, hidden=8, out_features=8)

    result = train_phase_a(
        train_spikes,
        train_labels,
        test_spikes,
        test_labels,
        epochs=2,
        model=model,
        checkpoint_dir=tmp_path,
    )

    assert result.prototypes.shape == (4, 8)
    assert 0.0 <= result.test_accuracy <= 1.0
    assert len(result.train_losses) == 2
    assert result.checkpoint_path.exists()
    assert result.prototype_path.exists()
    assert torch.load(result.prototype_path, weights_only=True).shape == (4, 8)
    assert all(not parameter.requires_grad for parameter in result.model.parameters())


def test_phase_a_seed_reproduces_metrics(tmp_path):
    rng = np.random.default_rng(12)
    train_spikes = (rng.random((12, 5, 8)) > 0.75).astype(np.uint8)
    test_spikes = (rng.random((12, 5, 4)) > 0.75).astype(np.uint8)
    labels = np.array([10, 20, 30, 40, 10, 20, 30, 40])
    test_labels = np.array([10, 20, 30, 40])

    first = train_phase_a(
        train_spikes, labels, test_spikes, test_labels, epochs=1,
        checkpoint_dir=tmp_path / "first",
    )
    second = train_phase_a(
        train_spikes, labels, test_spikes, test_labels, epochs=1,
        checkpoint_dir=tmp_path / "second",
    )

    assert first.train_accuracy == second.train_accuracy
    assert first.test_accuracy == second.test_accuracy
    torch.testing.assert_close(first.prototypes, second.prototypes)
