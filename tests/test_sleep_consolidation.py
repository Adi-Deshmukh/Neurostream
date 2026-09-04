"""Tests for sleep consolidation module and ablation comparisons."""

import numpy as np
import pytest
import torch

from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.training.phase_b_adapt import adapt_target_session
from neurostream.training.sleep_consolidation import SleepConsolidator, SleepConsolidationStepResult


@pytest.fixture
def snapshot_prototypes() -> torch.Tensor:
    """Fixed class prototypes representing Phase A session-1 features."""
    torch.manual_seed(42)
    # 4 classes, 128 features (normalized)
    p = torch.randn(4, 128)
    return torch.nn.functional.normalize(p, dim=-1)


def test_synthetic_features_shape_and_noise(snapshot_prototypes):
    sigma = 0.05
    consolidator = SleepConsolidator(
        snapshot_prototypes,
        noise_std=sigma,
        seed=123,
    )
    synthetic = consolidator.generate_synthetic_features(batch_size_per_class=100)
    assert synthetic.shape == (4, 100, 128)

    diff = synthetic - snapshot_prototypes.unsqueeze(1)
    # Mean difference should be close to 0
    assert torch.abs(diff.mean()).item() < 0.01
    # Standard deviation should be close to sigma=0.05
    assert torch.abs(diff.std() - sigma).item() < 0.01


def test_sleep_loss_computation(snapshot_prototypes):
    consolidator = SleepConsolidator(snapshot_prototypes)

    # Identical prototypes -> loss should be exactly 0
    loss, per_class = consolidator.compute_sleep_loss(snapshot_prototypes)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(per_class, torch.zeros(4), atol=1e-6, rtol=1e-6)

    # Orthogonal or drifted prototypes -> loss should be > 0
    drifted = snapshot_prototypes + torch.randn_like(snapshot_prototypes) * 0.5
    loss_drifted, per_class_drifted = consolidator.compute_sleep_loss(drifted)
    assert loss_drifted.item() > 0.05
    assert (per_class_drifted > 0.0).all()


def test_consolidation_pulls_drifted_prototypes_back(snapshot_prototypes):
    consolidator = SleepConsolidator(
        snapshot_prototypes,
        sleep_steps=5,
        noise_std=0.05,
        pull_rate=0.5,
        seed=42,
    )

    # Simulate drifted prototypes (e.g. from adapting to new session distribution)
    torch.manual_seed(99)
    drifted = snapshot_prototypes + torch.randn_like(snapshot_prototypes) * 0.8

    initial_loss, _ = consolidator.compute_sleep_loss(drifted)
    consolidated, step_res = consolidator.consolidate(drifted)
    final_loss, _ = consolidator.compute_sleep_loss(consolidated)

    assert step_res.triggered
    assert final_loss.item() < initial_loss.item()
    assert step_res.drift_after < step_res.drift_before


def test_consolidation_interval_trigger_schedule(snapshot_prototypes):
    interval = 50
    consolidator = SleepConsolidator(
        snapshot_prototypes,
        consolidation_interval=interval,
        sleep_steps=5,
    )

    current = snapshot_prototypes.clone() + 0.1

    # Step 49 -> Not triggered
    _, res_49 = consolidator.step(current, step_count=49)
    assert not res_49.triggered

    # Step 50 -> Triggered
    new_proto, res_50 = consolidator.step(current, step_count=50)
    assert res_50.triggered
    assert consolidator.consolidation_count == 1

    # Step 51 -> Not triggered
    _, res_51 = consolidator.step(new_proto, step_count=51)
    assert not res_51.triggered

    # Step 100 -> Triggered
    _, res_100 = consolidator.step(new_proto, step_count=100)
    assert res_100.triggered
    assert consolidator.consolidation_count == 2


def test_adapt_target_session_with_consolidation():
    torch.manual_seed(42)
    model = SNNFeatureExtractor(in_features=6, hidden=8, out_features=4)
    model_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 120 trials so multiple consolidation phases trigger (interval=50)
    n_trials = 120
    spikes = (np.random.default_rng(42).random((6, 5, n_trials)) > 0.6).astype(np.uint8)
    prototypes = torch.randn(4, 4)
    labels = np.tile(np.arange(4), n_trials // 4)

    result = adapt_target_session(
        model,
        prototypes,
        spikes,
        labels,
        confidence_threshold=0.0,
        enable_consolidation=True,
        consolidation_interval=50,
        sleep_steps=5,
        noise_std=0.05,
        pull_rate=0.5,
    )

    # Check model weights remained frozen
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, model_before[k])

    assert result.accepted_trials == n_trials
    assert result.consolidation_phases == 2  # At step 50 and 100
    assert not np.isnan(result.drift_before_after)
    assert result.prototypes.shape == (4, 4)


def test_consolidation_ablation_improves_retention():
    """Ablation comparison: consolidation ON vs OFF.

    With high drift pressure toward target data, consolidation ON should yield
    lower drift from session-1 prototypes and measurably higher session-1 retention.
    """
    torch.manual_seed(42)
    model = SNNFeatureExtractor(in_features=6, hidden=8, out_features=4)

    # Source session (session 1)
    n_source = 60
    source_spikes = (np.random.default_rng(1).random((6, 5, n_source)) > 0.6).astype(np.uint8)
    source_labels = np.tile(np.arange(4), n_source // 4)

    # Target session (session 2)
    n_target = 160
    target_spikes = (np.random.default_rng(2).random((6, 5, n_target)) > 0.6).astype(np.uint8)
    target_labels = np.tile(np.arange(4), n_target // 4)

    # Initial session-1 prototypes
    with torch.no_grad():
        source_feats = model(torch.as_tensor(source_spikes, dtype=torch.float32))
        s1_prototypes = torch.stack([
            source_feats[source_labels == c].mean(dim=0) for c in range(4)
        ])

    # Run adaptation without consolidation (OFF)
    res_off = adapt_target_session(
        model,
        s1_prototypes.clone(),
        target_spikes,
        target_labels,
        source_spikes=source_spikes,
        source_labels=source_labels,
        confidence_threshold=0.0,
        momentum=0.8,  # aggressive update to induce drift
        enable_consolidation=False,
    )

    # Run adaptation with consolidation (ON)
    res_on = adapt_target_session(
        model,
        s1_prototypes.clone(),
        target_spikes,
        target_labels,
        source_spikes=source_spikes,
        source_labels=source_labels,
        confidence_threshold=0.0,
        momentum=0.8,
        enable_consolidation=True,
        consolidation_interval=25,  # frequent consolidation to counter drift
        sleep_steps=5,
        noise_std=0.05,
        pull_rate=0.8,
    )

    # Compute drift of resulting prototypes from s1_prototypes
    drift_off = 1.0 - torch.nn.functional.cosine_similarity(res_off.prototypes, s1_prototypes, dim=-1).mean().item()
    drift_on = 1.0 - torch.nn.functional.cosine_similarity(res_on.prototypes, s1_prototypes, dim=-1).mean().item()

    # Consolidation ON must keep prototypes closer to session 1
    assert drift_on < drift_off
    # Session 1 retention should be higher or equal with consolidation
    assert res_on.retention_accuracy >= res_off.retention_accuracy
