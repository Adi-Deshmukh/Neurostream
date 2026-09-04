"""Tests for the custom LIF neuron module.

Covers membrane dynamics, threshold-and-reset behavior, membrane decay,
and surrogate gradient flow.
"""

import torch
import torch.nn as nn
import pytest

from neurostream.models.lif_neuron import LIFNeuron, SurrogateSpike


# ── fixtures ──────────────────────────────────────────────────────────────

BATCH = 4
FEATURES = 16


@pytest.fixture
def neuron() -> LIFNeuron:
    return LIFNeuron(beta=0.9, threshold=1.0, slope=25.0)


# ── shape tests ───────────────────────────────────────────────────────────


class TestOutputShapes:
    """forward() must return (spike, mem) with shape matching input."""

    def test_output_shapes_2d(self, neuron):
        x = torch.randn(BATCH, FEATURES)
        spike, mem = neuron(x)
        assert spike.shape == (BATCH, FEATURES)
        assert mem.shape == (BATCH, FEATURES)

    def test_output_shapes_1d(self, neuron):
        x = torch.randn(FEATURES)
        spike, mem = neuron(x)
        assert spike.shape == (FEATURES,)
        assert mem.shape == (FEATURES,)


# ── binary spike tests ───────────────────────────────────────────────────


class TestSpikeBinary:
    """Spikes must be exactly {0.0, 1.0} in the forward pass."""

    def test_spike_binary_forward(self, neuron):
        x = torch.randn(BATCH, FEATURES)
        mem = None
        for _ in range(10):
            spike, mem = neuron(x, mem)
            unique = torch.unique(spike)
            for v in unique:
                assert v.item() in (0.0, 1.0), f"Non-binary spike value: {v.item()}"


# ── threshold and reset tests ────────────────────────────────────────────


class TestThresholdAndReset:
    """Supra-threshold input must fire; sub-threshold must stay silent."""

    def test_threshold_fires(self, neuron):
        """Constant input above threshold should produce spikes eventually."""
        x = torch.full((1, FEATURES), 1.5)  # well above threshold=1.0
        mem = None
        any_spike = False
        for _ in range(10):
            spike, mem = neuron(x, mem)
            if spike.sum().item() > 0:
                any_spike = True
                break
        assert any_spike, "Supra-threshold input never produced a spike"

    def test_subthreshold_silent(self, neuron):
        """Very small constant input should never fire over a few steps."""
        x = torch.full((1, FEATURES), 0.01)  # well below threshold=1.0
        mem = None
        total_spikes = 0.0
        # With beta=0.9, membrane converges to 0.01/(1-0.9) = 0.1 < 1.0
        for _ in range(50):
            spike, mem = neuron(x, mem)
            total_spikes += spike.sum().item()
        assert total_spikes == 0.0, "Sub-threshold input should produce zero spikes"

    def test_reset_on_spike(self, neuron):
        """After a spike fires, membrane should drop below threshold."""
        x = torch.full((1, 1), 2.0)  # guaranteed to fire on first step
        spike, mem = neuron(x)
        assert spike.item() == 1.0, "Should fire on strong input"
        assert mem.item() < neuron.threshold, (
            f"After spike, mem={mem.item():.3f} should be < threshold={neuron.threshold}"
        )


# ── membrane decay test ─────────────────────────────────────────────────


class TestMembraneDynamics:
    """Membrane should decay toward zero with no input."""

    def test_membrane_decay(self, neuron):
        """Zero input → membrane decays by factor beta each step."""
        # Start with sub-threshold membrane (no spike)
        mem = torch.full((1, FEATURES), 0.5)
        x = torch.zeros(1, FEATURES)

        _, mem_next = neuron(x, mem)

        # No spike should occur (0.5 * 0.9 = 0.45 < 1.0)
        expected = 0.5 * neuron.beta
        torch.testing.assert_close(
            mem_next,
            torch.full_like(mem_next, expected),
            atol=1e-6,
            rtol=1e-6,
        )


# ── surrogate gradient test ─────────────────────────────────────────────


class TestSurrogateGradient:
    """Gradients must flow through the surrogate spike function."""

    def test_surrogate_gradient_flows(self):
        """A linear layer → LIF → loss.backward() must produce non-None,
        non-NaN gradients on the linear layer's weight."""
        linear = nn.Linear(FEATURES, FEATURES)
        lif = LIFNeuron(beta=0.9, threshold=1.0, slope=25.0)

        x = torch.randn(BATCH, FEATURES)
        mem = None

        # Run 5 timesteps so there's a chance of spikes
        total_spikes = torch.zeros(BATCH, FEATURES)
        for _ in range(5):
            cur = linear(x)
            spike, mem = lif(cur, mem)
            total_spikes = total_spikes + spike

        loss = total_spikes.sum()
        loss.backward()

        assert linear.weight.grad is not None, "Gradient is None — surrogate not working"
        assert not torch.isnan(linear.weight.grad).any(), "NaN in gradients"
        assert linear.weight.grad.abs().sum() > 0, "All-zero gradients"


# ── init_mem test ────────────────────────────────────────────────────────


class TestInitMem:
    """init_mem() factory must return zeros with correct shape."""

    def test_init_mem(self, neuron):
        mem = neuron.init_mem(BATCH, FEATURES)
        assert mem.shape == (BATCH, FEATURES)
        assert (mem == 0).all()
        assert mem.dtype == torch.float32
