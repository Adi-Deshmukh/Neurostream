"""Tests for evaluation metrics and spike-sparsity tracking."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from neurostream.evaluation.metrics import (
    EvaluationMetrics,
    SpikeSparsityTracker,
    accuracy,
    adaptation_gain,
    forgetting,
    format_metrics_table,
    measure_model_sparsity,
    retention_rate,
)
from neurostream.models.lif_neuron import LIFNeuron
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor


# ── Metric arithmetic tests ──────────────────────────────────────────────────


def test_accuracy_various_inputs():
    # Perfect accuracy
    assert accuracy([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    # Zero accuracy
    assert accuracy([0, 1], [1, 0]) == 0.0
    # 50% accuracy with numpy arrays
    assert accuracy(np.array([0, 1, 2, 3]), np.array([0, 1, 0, 0])) == 0.5
    # PyTorch tensors
    assert accuracy(torch.tensor([1, 2]), torch.tensor([1, 2])) == 1.0

    # Error handling
    with pytest.raises(ValueError, match="matching"):
        accuracy([1, 2], [1])
    with pytest.raises(ValueError, match="empty"):
        accuracy([], [])


def test_adaptation_gain_and_forgetting():
    # Gain: post - pre
    assert pytest.approx(adaptation_gain(0.40, 0.55)) == 0.15
    assert pytest.approx(adaptation_gain(0.60, 0.50)) == -0.10

    # Forgetting: initial - post_adaptation
    assert pytest.approx(forgetting(0.80, 0.65)) == 0.15
    assert pytest.approx(forgetting(0.70, 0.75)) == -0.05

    # Retention rate: post / initial
    assert pytest.approx(retention_rate(0.80, 0.60)) == 0.75
    assert retention_rate(0.0, 0.5) == 0.0


# ── SpikeSparsityTracker hook tests ──────────────────────────────────────────


def test_sparsity_tracker_with_snn_extractor():
    torch.manual_seed(42)
    model = SNNFeatureExtractor(in_features=66, hidden=256, out_features=128)
    spikes_input = (torch.rand(66, 25, 10) > 0.7).float()

    with SpikeSparsityTracker(model) as tracker:
        assert tracker.is_active
        out = model(spikes_input)
        mean_rate = tracker.mean_firing_rate()
        layer_rates = tracker.layer_sparsity()

    assert not tracker.is_active
    assert 0.0 <= mean_rate <= 1.0
    assert "lif1" in layer_rates
    assert "lif2" in layer_rates
    assert 0.0 <= layer_rates["lif1"] <= 1.0
    assert 0.0 <= layer_rates["lif2"] <= 1.0

    # Test measure_model_sparsity helper
    m_rate, l_rates = measure_model_sparsity(model, spikes_input)
    assert pytest.approx(m_rate, rel=1e-5) == mean_rate
    assert pytest.approx(l_rates["lif1"], rel=1e-5) == layer_rates["lif1"]


def test_sparsity_tracker_extremes():
    """Verify 0% sparsity for silent neurons and 100% for saturated neurons."""
    # Custom mock with LIFNeuron that never fires
    class SilentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lif = LIFNeuron(threshold=100.0)

        def forward(self, x):
            spk, _ = self.lif(x)
            return spk

    silent = SilentModel()
    with SpikeSparsityTracker(silent) as tracker:
        _ = silent(torch.zeros(10, 20))
        assert tracker.mean_firing_rate() == 0.0

    # Custom mock with LIFNeuron that always fires
    class SaturatedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lif = LIFNeuron(threshold=0.01)

        def forward(self, x):
            spk, _ = self.lif(x)
            return spk

    sat = SaturatedModel()
    with SpikeSparsityTracker(sat) as tracker:
        _ = sat(torch.full((10, 20), 5.0))
        assert tracker.mean_firing_rate() == 1.0


# ── Table formatting test ────────────────────────────────────────────────────


def test_format_metrics_table():
    record1 = EvaluationMetrics(
        subject=1,
        pre_adaptation_acc=0.3759,
        post_adaptation_acc=0.3865,
        adaptation_gain=0.0106,
        initial_retention_acc=0.6000,
        post_retention_acc=0.5426,
        forgetting=0.0574,
        retention_rate=0.9043,
        spike_sparsity=0.1845,
        consolidation_enabled=False,
    )
    record2 = EvaluationMetrics(
        subject=1,
        pre_adaptation_acc=0.3759,
        post_adaptation_acc=0.3759,
        adaptation_gain=0.0000,
        initial_retention_acc=0.6000,
        post_retention_acc=0.5532,
        forgetting=0.0468,
        retention_rate=0.9220,
        spike_sparsity=0.1830,
        consolidation_enabled=True,
    )

    table = format_metrics_table([record1, record2])
    assert "Section 5.4.1" in table
    assert "Pre-Acc" in table
    assert "Forget" in table
    assert "Sparsity" in table
    assert "37.59%" in table
    assert "54.26%" in table
    assert "55.32%" in table
    assert "OFF" in table
    assert "ON" in table

