"""Tests for frozen-SNN prototype adaptation."""

import numpy as np
import torch

from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.models.prototype_memory import PrototypeMemory
from neurostream.training.phase_b_adapt import adapt_target_session, calibrate_target_session


def test_prototype_update_matches_ema_rule():
    memory = PrototypeMemory(torch.zeros(4, 3), alpha=0.25)

    memory.update(torch.tensor([4.0, 8.0, 12.0]), 2)

    torch.testing.assert_close(
        memory.prototypes[2], torch.tensor([1.0, 2.0, 3.0])
    )
    torch.testing.assert_close(memory.prototypes[0], torch.zeros(3))


def test_prototype_classification_uses_cosine_argmax():
    prototypes = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ])
    memory = PrototypeMemory(prototypes)
    features = torch.tensor([
        [3.0, 0.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, 5.0],
    ])

    torch.testing.assert_close(memory.classify(features), torch.tensor([0, 3, 2]))


def test_adaptation_updates_prototypes_not_model(tmp_path):
    torch.manual_seed(3)
    model = SNNFeatureExtractor(in_features=6, hidden=8, out_features=4)
    spikes = (np.random.default_rng(3).random((6, 5, 8)) > 0.7).astype(np.uint8)
    prototypes = torch.randn(4, 4)
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    result = adapt_target_session(
        model,
        prototypes,
        spikes,
        np.arange(8) % 4,
        confidence_threshold=0.0,
        batch_size=3,
    )

    assert result.accepted_trials == 8
    assert result.prototypes.shape == (4, 4)
    assert result.before_accuracy >= 0
    assert result.after_accuracy >= 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_state[name])


def test_labelled_calibration_keeps_evaluation_trials_separate():
    torch.manual_seed(4)
    model = SNNFeatureExtractor(in_features=6, hidden=8, out_features=4)
    spikes = (np.random.default_rng(4).random((6, 5, 20)) > 0.7).astype(np.uint8)
    result = calibrate_target_session(
        model, torch.randn(4, 4), spikes, np.tile(np.arange(4), 5),
        calibration_fraction=0.25, seed=42,
    )
    assert result.calibration_trials == 4
    assert result.evaluation_trials == 16
    assert result.accepted_trials == result.calibration_trials
