"""Tests for the conventional EEGNet-style baseline."""

import numpy as np
import torch

from neurostream.models.eegnet_baseline import EEGNetBaseline, train_eegnet_baseline


def test_eegnet_forward_shape():
    model = EEGNetBaseline(channels=6, n_classes=4)
    output = model(torch.randn(5, 6, 200))
    assert output.shape == (5, 4)
    assert torch.isfinite(output).all()


def test_eegnet_training_uses_stratified_split_and_saves(tmp_path):
    rng = np.random.default_rng(4)
    train_x = rng.standard_normal((20, 6, 200)).astype(np.float32)
    test_x = rng.standard_normal((8, 6, 200)).astype(np.float32)
    train_y = np.tile(np.arange(4), 5)
    test_y = np.tile(np.arange(4), 2)
    result = train_eegnet_baseline(
        train_x, train_y, test_x, test_y, epochs=1, batch_size=4,
        checkpoint_path=tmp_path / "baseline.pt",
    )
    assert 0 <= result.train_accuracy <= 1
    assert 0 <= result.validation_accuracy <= 1
    assert 0 <= result.test_accuracy <= 1
    assert result.checkpoint_path.exists()
