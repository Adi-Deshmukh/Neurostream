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
    assert engine.measurement is not None
    assert engine.measurement.units == "fraction_of_spike_opportunities"
    assert 0.0 <= engine.measurement.aggregate_activity_fraction <= 1.0


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
    assert sim.prototype_history_2d.shape == (50, 4, 2)
    np.testing.assert_allclose(
        sim.prototype_history_2d[0], sim.steps[0].prototypes_2d
    )
    np.testing.assert_allclose(
        sim.prototype_history_2d[-1], sim.steps[-1].prototypes_2d
    )

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


def test_feature_only_prepared_artifact_round_trip(mock_replay_engine, tmp_path):
    engine = mock_replay_engine
    artifact_dir = tmp_path / "prepared_subject_1"
    engine.save_prepared_artifact(artifact_dir)

    loaded = TrialReplayEngine.from_prepared_artifact(
        artifact_dir,
        checkpoint_ref=engine.checkpoint_ref,
        subject=engine.subject,
    )

    assert loaded.s1_spikes is None
    assert loaded.s2_spikes is None
    assert loaded.model is None
    np.testing.assert_allclose(loaded.s1_features, engine.s1_features)
    np.testing.assert_allclose(loaded.s2_features, engine.s2_features)
    np.testing.assert_array_equal(loaded.s1_labels, engine.s1_labels)
    np.testing.assert_array_equal(loaded.s2_labels, engine.s2_labels)
    assert loaded.measurement is not None
    assert loaded.measurement.checkpoint_ref == engine.checkpoint_ref

    original_replay = engine.run_replay(
        momentum=0.9, confidence_threshold=0.5, enable_consolidation=False
    )
    cached_replay = loaded.run_replay(
        momentum=0.9, confidence_threshold=0.5, enable_consolidation=False
    )
    assert cached_replay.adapted_target_accuracy == pytest.approx(
        original_replay.adapted_target_accuracy
    )
    np.testing.assert_allclose(
        cached_replay.prototype_history_2d,
        original_replay.prototype_history_2d,
    )


def test_prepared_artifact_rejects_checkpoint_mismatch(mock_replay_engine, tmp_path):
    artifact_dir = tmp_path / "prepared_subject_1"
    mock_replay_engine.save_prepared_artifact(artifact_dir)

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        TrialReplayEngine.from_prepared_artifact(
            artifact_dir,
            checkpoint_ref="wrong-checkpoint-hash",
            subject=mock_replay_engine.subject,
        )


def test_sleep_consolidation_replay_is_reproducible(mock_replay_engine):
    engine = mock_replay_engine
    kwargs = {
        "momentum": 0.9,
        "confidence_threshold": 0.5,
        "enable_consolidation": True,
        "consolidation_interval": 5,
        "sleep_steps": 2,
        "pull_rate": 0.5,
    }
    first = engine.run_replay(**kwargs)
    second = engine.run_replay(**kwargs)

    assert first.adapted_target_accuracy == pytest.approx(
        second.adapted_target_accuracy
    )
    assert first.session1_retention_accuracy == pytest.approx(
        second.session1_retention_accuracy
    )
    np.testing.assert_allclose(
        first.prototype_history_2d,
        second.prototype_history_2d,
    )
