"""Tests for the dual-scheme spike encoder.

Uses a fixed-seed synthetic fixture (seed=42) producing (8, 22, 1000) epochs
to simulate 8 trials of 22-channel EEG at 250 Hz over 4 seconds.
"""

import numpy as np
import pytest

from neurostream.data.spike_encoder import (
    N_BANDS,
    encode,
)

# ── fixtures ──────────────────────────────────────────────────────────────

SFREQ = 250.0
N_TRIALS = 8
N_CHANNELS = 22
T_EPOCH = 1000  # 4 s at 250 Hz
N_TIMESTEPS = 25


@pytest.fixture
def synthetic_epochs() -> np.ndarray:
    """Deterministic synthetic EEG-like data: (8, 22, 1000) float32."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((N_TRIALS, N_CHANNELS, T_EPOCH)).astype(np.float32)


@pytest.fixture
def encoded(synthetic_epochs):
    """Run the encoder once and cache the result for multiple tests."""
    return encode(synthetic_epochs, sfreq=SFREQ, n_timesteps=N_TIMESTEPS)


# ── tests ─────────────────────────────────────────────────────────────────


class TestOutputShape:
    """Output tensor must be (66, 25, N)."""

    def test_output_shape(self, encoded):
        assert encoded.spikes.shape == (N_CHANNELS * N_BANDS, N_TIMESTEPS, N_TRIALS)

    def test_virtual_channel_count(self, encoded):
        """66 = 22 channels × 3 bands."""
        assert encoded.spikes.shape[0] == N_CHANNELS * N_BANDS


class TestBinaryOutput:
    """Every element must be strictly 0 or 1."""

    def test_output_binary(self, encoded):
        unique = np.unique(encoded.spikes)
        assert set(unique).issubset({0, 1}), f"Non-binary values found: {unique}"

    def test_dtype(self, encoded):
        assert encoded.spikes.dtype == np.uint8


class TestFiringRateBounds:
    """Mean firing rate must be non-trivial but well under 50%."""

    def test_mean_firing_rate_under_50_pct(self, encoded):
        assert encoded.mean_firing_rate < 0.50, (
            f"Mean firing rate {encoded.mean_firing_rate:.2%} exceeds 50% ceiling"
        )

    def test_firing_rate_nontrivial(self, encoded):
        assert encoded.mean_firing_rate > 0.0, "All-zero spike tensor is degenerate"


class TestGammaTTFS:
    """Gamma channels (indices 44–65) should have exactly 1 spike per trial."""

    def test_gamma_ttfs_sparsity(self, encoded):
        gamma_start = N_CHANNELS * 2  # channels 44–65
        gamma_spikes = encoded.spikes[gamma_start:, :, :]  # (22, 25, N)
        # Each (channel, trial) pair should have exactly 1 spike
        spikes_per_trial_channel = gamma_spikes.sum(axis=1)  # (22, N)
        assert np.all(spikes_per_trial_channel == 1), (
            "TTFS gamma channels should have exactly 1 spike per trial per channel, "
            f"got min={spikes_per_trial_channel.min()}, max={spikes_per_trial_channel.max()}"
        )


class TestDeterminism:
    """Two calls with the same input must produce identical output."""

    def test_deterministic(self, synthetic_epochs):
        result_a = encode(synthetic_epochs, sfreq=SFREQ, n_timesteps=N_TIMESTEPS)
        result_b = encode(synthetic_epochs, sfreq=SFREQ, n_timesteps=N_TIMESTEPS)
        np.testing.assert_array_equal(result_a.spikes, result_b.spikes)
        assert result_a.mean_firing_rate == result_b.mean_firing_rate


class TestThresholdControl:
    """Higher threshold percentile should produce fewer spikes."""

    def test_rate_threshold_adjusts_sparsity(self, synthetic_epochs):
        result_low = encode(
            synthetic_epochs, sfreq=SFREQ, n_timesteps=N_TIMESTEPS,
            rate_threshold_pct=30.0,
        )
        result_high = encode(
            synthetic_epochs, sfreq=SFREQ, n_timesteps=N_TIMESTEPS,
            rate_threshold_pct=70.0,
        )
        assert result_high.mean_firing_rate < result_low.mean_firing_rate, (
            f"Higher threshold ({result_high.mean_firing_rate:.2%}) should produce "
            f"fewer spikes than lower threshold ({result_low.mean_firing_rate:.2%})"
        )


class TestEdgeCases:
    """Edge cases: single trial, minimal input."""

    def test_channel_and_timestep_dimensions_are_dynamic(self):
        rng = np.random.default_rng(7)
        dynamic_epochs = rng.standard_normal((2, 5, 400)).astype(np.float32)
        result = encode(dynamic_epochs, sfreq=250.0, n_timesteps=10)
        assert result.spikes.shape == (5 * N_BANDS, 10, 2)

    def test_single_trial(self):
        rng = np.random.default_rng(99)
        single = rng.standard_normal((1, N_CHANNELS, T_EPOCH)).astype(np.float32)
        result = encode(single, sfreq=SFREQ, n_timesteps=N_TIMESTEPS)
        assert result.spikes.shape == (N_CHANNELS * N_BANDS, N_TIMESTEPS, 1)
        assert set(np.unique(result.spikes)).issubset({0, 1})


class TestInputValidation:
    """Bad inputs must raise ValueError."""

    def test_input_validation_2d(self):
        with pytest.raises(ValueError, match="ndim"):
            encode(np.zeros((22, 1000)), sfreq=SFREQ)

    def test_input_validation_1d(self):
        with pytest.raises(ValueError, match="ndim"):
            encode(np.zeros((1000,)), sfreq=SFREQ)
