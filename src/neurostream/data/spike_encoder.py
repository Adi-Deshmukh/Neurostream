"""Dual-scheme spike encoder for preprocessed EEG epochs.

Converts ``(N, 22, T_epoch)`` z-scored EEG into a ``(66, 25, N)`` binary spike
tensor using:

* **Rate coding** for alpha (8-13 Hz) and beta (13-30 Hz) bands — preserves
  the amplitude-modulated ERD/ERS motor-imagery signal.
* **Time-to-first-spike (TTFS) coding** for the gamma (30+ Hz) band —
  preserves transient onset timing better than rate coding at high frequencies.

66 = 22 channels × 3 frequency bands, 25 timesteps per trial.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band definitions (lo_hz, hi_hz)
# ---------------------------------------------------------------------------
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
GAMMA_BAND = (30.0, 100.0)

N_CHANNELS = 22
N_BANDS = 3  # alpha, beta, gamma
N_VIRTUAL_CHANNELS = N_CHANNELS * N_BANDS  # 66


class EncoderResult(NamedTuple):
    """Return type of :func:`encode`."""

    spikes: np.ndarray  # (66, n_timesteps, N) binary
    mean_firing_rate: float


# ── internal helpers ──────────────────────────────────────────────────────


def _bandpass(
    data: np.ndarray, sfreq: float, lo: float, hi: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass on the last axis."""
    nyq = sfreq / 2.0
    hi = min(hi, nyq - 2.0)  # clamp to stay safely below Nyquist
    if lo >= hi:
        raise ValueError(
            f"Band ({lo}, {hi}) Hz is invalid at sfreq={sfreq} Hz "
            f"(Nyquist={nyq} Hz)"
        )
    sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32, copy=False)


def _decompose_bands(
    epochs: np.ndarray, sfreq: float
) -> np.ndarray:
    """Extract alpha, beta, gamma sub-bands from each channel.

    Parameters
    ----------
    epochs : ndarray, shape ``(N, 22, T_epoch)``
    sfreq : float

    Returns
    -------
    bands : ndarray, shape ``(3, N, 22, T_epoch)``
        Index 0 = alpha, 1 = beta, 2 = gamma.
    """
    alpha = _bandpass(epochs, sfreq, *ALPHA_BAND)
    beta = _bandpass(epochs, sfreq, *BETA_BAND)
    gamma = _bandpass(epochs, sfreq, *GAMMA_BAND)
    return np.stack([alpha, beta, gamma], axis=0)


def _band_power(band_signal: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute binned power envelope.

    Parameters
    ----------
    band_signal : ndarray, shape ``(N, 22, T_epoch)``
    n_bins : int
        Number of temporal bins (timesteps).

    Returns
    -------
    power : ndarray, shape ``(N, 22, n_bins)``
        Mean squared amplitude per bin.
    """
    n_trials, n_ch, t_total = band_signal.shape
    # Truncate so T divides evenly into n_bins
    usable = (t_total // n_bins) * n_bins
    trimmed = band_signal[:, :, :usable]
    # Reshape → (N, 22, n_bins, samples_per_bin), then mean-square
    reshaped = trimmed.reshape(n_trials, n_ch, n_bins, -1)
    return np.mean(reshaped ** 2, axis=-1).astype(np.float32, copy=False)


def _rate_encode(
    power: np.ndarray, threshold_percentile: float
) -> np.ndarray:
    """Deterministic rate coding: fire if bin power ≥ per-channel threshold.

    Parameters
    ----------
    power : ndarray, shape ``(N, 22, n_bins)``
    threshold_percentile : float
        Percentile (0-100) of the power distribution *per channel across all
        trials* used as the firing threshold.

    Returns
    -------
    spikes : ndarray, shape ``(N, 22, n_bins)``  dtype uint8, values {0, 1}.
    """
    # Compute threshold per channel: percentile across trials and time bins
    # power shape (N, 22, n_bins) → collapse axis 0 and 2
    thresholds = np.percentile(
        power, threshold_percentile, axis=(0, 2), keepdims=True
    )  # shape (1, 22, 1)
    return (power >= thresholds).astype(np.uint8)


def _ttfs_encode(power: np.ndarray) -> np.ndarray:
    """Time-to-first-spike coding: one spike at the peak-power bin.

    For each (trial, channel), the bin with maximum power fires a single spike.
    This inherently yields sparsity of ``1 / n_bins``.

    Parameters
    ----------
    power : ndarray, shape ``(N, 22, n_bins)``

    Returns
    -------
    spikes : ndarray, shape ``(N, 22, n_bins)``  dtype uint8, values {0, 1}.
    """
    n_trials, n_ch, n_bins = power.shape
    spikes = np.zeros_like(power, dtype=np.uint8)
    peak_bins = np.argmax(power, axis=-1)  # (N, 22)
    # Advanced indexing to set the peak bin to 1
    trials_idx = np.arange(n_trials)[:, None]  # (N, 1)
    ch_idx = np.arange(n_ch)[None, :]  # (1, 22)
    spikes[trials_idx, ch_idx, peak_bins] = 1
    return spikes


# ── public API ────────────────────────────────────────────────────────────


def encode(
    epochs: np.ndarray,
    sfreq: float = 250.0,
    *,
    n_timesteps: int = 25,
    rate_threshold_pct: float = 50.0,
) -> EncoderResult:
    """Encode preprocessed EEG epochs into binary spike trains.

    Parameters
    ----------
    epochs : ndarray, shape ``(N, 22, T_epoch)``
        Z-scored EEG trials from the preprocessing pipeline.
    sfreq : float
        Sampling frequency in Hz (default 250).
    n_timesteps : int
        Number of temporal bins per trial (default 25).
    rate_threshold_pct : float
        Percentile threshold for rate coding (default 50).

    Returns
    -------
    result : EncoderResult
        ``result.spikes`` — ``(66, n_timesteps, N)`` uint8 binary tensor.
        ``result.mean_firing_rate`` — scalar in [0, 1].

    Raises
    ------
    ValueError
        If ``epochs`` does not have 3 dimensions.
    """
    if epochs.ndim != 3:
        raise ValueError(
            f"epochs must have shape (N, channels, samples), got ndim={epochs.ndim}"
        )

    n_trials, n_ch, t_epoch = epochs.shape

    # 1. Sub-band decomposition → (3, N, 22, T_epoch)
    bands = _decompose_bands(epochs, sfreq)

    # 2. Compute binned power per band → (3, N, 22, n_timesteps)
    powers = np.stack(
        [_band_power(bands[b], n_timesteps) for b in range(N_BANDS)], axis=0
    )

    # 3. Encode: rate for alpha (0) & beta (1), TTFS for gamma (2)
    alpha_spikes = _rate_encode(powers[0], rate_threshold_pct)  # (N, 22, T)
    beta_spikes = _rate_encode(powers[1], rate_threshold_pct)   # (N, 22, T)
    gamma_spikes = _ttfs_encode(powers[2])                      # (N, 22, T)

    # 4. Stack bands → (N, 66, n_timesteps) then transpose to (66, T, N)
    all_spikes = np.concatenate(
        [alpha_spikes, beta_spikes, gamma_spikes], axis=1
    )  # (N, 66, n_timesteps)
    spike_tensor = np.transpose(all_spikes, (1, 2, 0))  # (66, n_timesteps, N)

    # 5. Validate & log
    assert spike_tensor.shape == (N_VIRTUAL_CHANNELS, n_timesteps, n_trials)
    mean_fr = float(spike_tensor.mean())
    logger.info(
        "Spike encoding complete: shape=%s, mean_firing_rate=%.4f (%.1f%%)",
        spike_tensor.shape,
        mean_fr,
        mean_fr * 100,
    )

    # Per-band breakdown for diagnostics
    alpha_fr = float(alpha_spikes.mean())
    beta_fr = float(beta_spikes.mean())
    gamma_fr = float(gamma_spikes.mean())
    logger.info(
        "  Per-band firing rates — alpha: %.4f (%.1f%%), "
        "beta: %.4f (%.1f%%), gamma: %.4f (%.1f%%)",
        alpha_fr, alpha_fr * 100,
        beta_fr, beta_fr * 100,
        gamma_fr, gamma_fr * 100,
    )

    if mean_fr >= 0.50:
        logger.warning(
            "Mean firing rate %.2f%% exceeds 50%% target — consider raising "
            "rate_threshold_pct or switching to Poisson rate coding.",
            mean_fr * 100,
        )

    return EncoderResult(spikes=spike_tensor, mean_firing_rate=mean_fr)
