"""Tests for dashboard simulation and trial replay engine."""

import numpy as np
import pytest
import torch

from neurostream.dashboard.simulation import TrialReplayEngine
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor


@pytest.fixture
def mock_replay_engine() -> TrialReplayEngine:
    torch.manual_seed(42)
    model = SNNFeatureExtractor(in_features=6, hidden=16, out_features=8)
    initial_prototypes = torch.randn(4, 8)

    n_s1 = 40
    n_s2 = 50
    s1_spikes = (np.random.default_rng(1).random((6, 5, n_s1)) > 0.7).astype(np.uint8)
    s1_labels = np.tile(np.arange(4), n_s1 // 4)

    s2_spikes = (np.random.default_rng(2).random((6, 5, n_s2)) > 0.7).astype(np.uint8)
    s2_labels = np.tile(np.arange(4), (n_s2 // 4) + 1)[:n_s2]

    return TrialReplayEngine(
        model=model,
        initial_prototypes=initial_prototypes,
        session1_spikes=s1_spikes,
        session1_labels=s1_labels,
        session2_spikes=s2_spikes,
        session2_labels=s2_labels,
        class_names=["Left", "Right", "Feet", "Tongue"],
    )


def test_replay_engine_initialization(mock_replay_engine):
    engine = mock_replay_engine
    assert engine.s1_features_2d.shape == (40, 2)
    assert engine.s2_features_2d.shape == (50, 2)
    assert engine.initial_prototypes_2d.shape == (4, 2)
    assert len(engine.class_names) == 4
    assert 0.0 <= engine.mean_sparsity <= 1.0


def test_replay_simulation_execution(mock_replay_engine):
    engine = mock_replay_engine
    sim = engine.run_replay(
        momentum=0.90,
        confidence_threshold=0.5,
        enable_consolidation=True,
        consolidation_interval=20,
        sleep_steps=3,
    )

    assert len(sim.steps) == 50
    assert sim.session1_features_2d.shape == (40, 2)
    assert sim.session2_features_2d.shape == (50, 2)

    # Verify step progression
    for i, step in enumerate(sim.steps):
        assert step.step == i + 1
        assert step.feature_2d.shape == (2,)
        assert step.prototypes_2d.shape == (4, 2)
        assert 0.0 <= step.confidence <= 1.0
        assert 0.0 <= step.cumulative_accuracy <= 1.0

    # Verify prototypes moved over time (trajectory changed)
    first_step_protos = sim.steps[0].prototypes_2d
    last_step_protos = sim.steps[-1].prototypes_2d
    # At least some prototypes should have adapted
    assert not np.allclose(first_step_protos, last_step_protos, atol=1e-6)
