"""Tests for the two-layer SNN feature extractor.

Validates output shape, gradient flow, determinism, non-degenerate features,
and edge cases using a dummy (66, 25, 8) spike tensor.
"""

import torch
import pytest

from neurostream.models.snn_feature_extractor import SNNFeatureExtractor


# ── fixtures ──────────────────────────────────────────────────────────────

N_VIRTUAL_CHANNELS = 66
N_TIMESTEPS = 25
BATCH_SIZE = 8


@pytest.fixture
def model() -> SNNFeatureExtractor:
    """Default SNN feature extractor."""
    torch.manual_seed(42)
    return SNNFeatureExtractor()


@pytest.fixture
def dummy_spikes() -> torch.Tensor:
    """Binary spike tensor matching spike encoder output: (66, 25, 8)."""
    torch.manual_seed(42)
    return (torch.rand(N_VIRTUAL_CHANNELS, N_TIMESTEPS, BATCH_SIZE) > 0.65).float()


# ── output shape test ────────────────────────────────────────────────────


class TestOutputShape:
    """Input (66, 25, N) must produce output (N, 512)."""

    def test_output_shape(self, model, dummy_spikes):
        out = model(dummy_spikes)
        assert out.shape == (BATCH_SIZE, 512)


# ── output validity ─────────────────────────────────────────────────────


class TestOutputValidity:
    """Output must be finite (no NaN/Inf) and non-degenerate."""

    def test_output_finite(self, model, dummy_spikes):
        out = model(dummy_spikes)
        assert torch.isfinite(out).all(), f"Non-finite values in output: {out}"

    def test_features_non_degenerate(self, model, dummy_spikes):
        """SNN must produce non-zero features with binary spike input.

        This catches the feature-collapse bug where all neurons are silent
        due to threshold/init mismatch.
        """
        out = model(dummy_spikes)
        mean_firing_rate = out.mean().item()
        assert mean_firing_rate > 0.01, (
            f"Mean firing rate {mean_firing_rate:.4f} is too low — "
            "features are collapsing (neurons not firing)"
        )


# ── gradient flow ────────────────────────────────────────────────────────


class TestGradientFlow:
    """All trainable parameters must receive non-None, non-NaN gradients."""

    def test_gradients_flow(self, model, dummy_spikes):
        out = model(dummy_spikes)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


# ── determinism ──────────────────────────────────────────────────────────


class TestDeterminism:
    """Same input must produce identical output (no stochastic behavior)."""

    def test_deterministic(self, dummy_spikes):
        torch.manual_seed(42)
        model_a = SNNFeatureExtractor()
        out_a = model_a(dummy_spikes)

        torch.manual_seed(42)
        model_b = SNNFeatureExtractor()
        out_b = model_b(dummy_spikes)

        torch.testing.assert_close(out_a, out_b)


# ── edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: single trial, custom dimensions."""

    def test_single_trial(self, model):
        """Batch size 1 should work correctly."""
        spikes = (torch.rand(N_VIRTUAL_CHANNELS, N_TIMESTEPS, 1) > 0.65).float()
        out = model(spikes)
        assert out.shape == (1, 512)
        assert torch.isfinite(out).all()

    def test_custom_dimensions(self):
        """Non-default hidden=64, out_features=32 should produce (N, 32)."""
        torch.manual_seed(42)
        model = SNNFeatureExtractor(in_features=66, hidden=64, out_features=32)
        spikes = (torch.rand(66, 25, BATCH_SIZE) > 0.65).float()
        out = model(spikes)
        assert out.shape == (BATCH_SIZE, 32)

    def test_default_threshold(self):
        """Default threshold should be 0.5 (lowered from 1.0 to fix collapse)."""
        model = SNNFeatureExtractor()
        assert model.lif1.threshold == 0.5
        assert model.lif2.threshold == 0.5
