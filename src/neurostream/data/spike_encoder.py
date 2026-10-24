"""Dual-scheme spike encoder for preprocessed EEG epochs.

Converts ``(N, channels, T_epoch)`` z-scored EEG into a
``(channels * 3, n_timesteps, N)`` binary spike tensor using:

* **Rate coding** for alpha (8-13 Hz) and beta (13-30 Hz) bands — preserves
  the amplitude-modulated ERD/ERS motor-imagery signal.
* **Time-to-first-spike (TTFS) coding** for the gamma (30+ Hz) band —
  preserves transient onset timing better than rate coding at high frequencies.

For the BNCI2014-001 defaults this is one virtual channel per input channel
and frequency band, with the timestep count selected by ``n_timesteps``.

Calibration contract
--------------------
Rate-encoding thresholds must be fitted **only** on Session 1 training data
and then *applied* without re-fitting to validation, held-out test, and online
Session 2 inference.  Use :func:`fit_encoder` once on the training split, then
call :func:`apply_encoder` for every subsequent split.  The convenience
function :func:`encode` preserves the old single-pass API (fits and applies on
the same batch) but is **only correct** when the caller passes a pure training
split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band definitions (lo_hz, hi_hz)
# ---------------------------------------------------------------------------
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
GAMMA_BAND = (30.0, 100.0)

BANDS = (ALPHA_BAND, BETA_BAND, GAMMA_BAND)
N_BANDS = len(BANDS)


@dataclass(frozen=True)
class EncoderResult:
    """Return type of :func:`encode` / :func:`apply_encoder`."""

    spikes: np.ndarray  # (channels * n_bands, n_timesteps, N) binary
    mean_firing_rate: float


@dataclass(frozen=True)
class EncoderCalibration:
    """Per-channel rate-encoding thresholds fitted on training data.

    Attributes
    ----------
    alpha_thresholds : ndarray, shape ``(1, channels, 1)``
        Per-channel power thresholds for the alpha band.
    beta_thresholds : ndarray, shape ``(1, channels, 1)``
        Per-channel power thresholds for the beta band.
    sfreq : float
        Sampling frequency used when the calibration was fitted.
    n_timesteps : int
        Number of temporal bins (timesteps) used when the calibration was fitted.
    rate_threshold_pct : float
    channel_count : int
    filter_order : int = 4
        Percentile used when the calibration was fitted.
    """

    alpha_thresholds: np.ndarray  # (1, channels, 1)
    beta_thresholds: np.ndarray   # (1, channels, 1)
    sfreq: float
    n_timesteps: int
    rate_threshold_pct: float
    channel_count: int
    filter_order: int = 4


# ── internal helpers ──────────────────────────────────────────────────────


def _validate_bands(sfreq: float) -> None:
    """Warn or raise if any required band exceeds the Nyquist frequency.

    For BNCI2014-001 at 250 Hz the Nyquist is 125 Hz so all three bands are
    valid after the 2 Hz safety margin applied in :func:`_bandpass`.  At lower
    sampling rates (e.g. 128 Hz) the gamma upper edge would be clamped, and a
    warning is emitted because the clamp still produces a valid filter.
    """
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be a finite positive number")
    nyq = sfreq / 2.0
    for lo, hi in BANDS:
        effective_hi = min(hi, nyq - 2.0)
        if lo >= effective_hi:
            raise ValueError(
                f"Band ({lo}, {hi}) Hz is entirely above Nyquist-2 Hz "
                f"({effective_hi:.1f} Hz) for sfreq={sfreq} Hz. "
                "Lower the band or use a higher sampling rate."
            )
        if hi > nyq - 2.0:
            logger.warning(
                "Band upper edge %.1f Hz clamped to %.1f Hz (Nyquist-2) "
                "for sfreq=%.1f Hz. Consider a higher sampling rate to "
                "preserve the full gamma band.",
                hi, nyq - 2.0, sfreq,
            )


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
    epochs : ndarray, shape ``(N, channels, T_epoch)``
    sfreq : float

    Returns
    -------
    bands : ndarray, shape ``(n_bands, N, channels, T_epoch)``
    Band order follows ``BANDS``.
    """
    return np.stack(
        [_bandpass(epochs, sfreq, *band) for band in BANDS],
        axis=0,
    )


def _band_power(band_signal: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute binned power envelope.

    Parameters
    ----------
    band_signal : ndarray, shape ``(N, channels, T_epoch)``
    n_bins : int
        Number of temporal bins (timesteps).

    Returns
    -------
    power : ndarray, shape ``(N, channels, n_bins)``
        Mean squared amplitude per bin.
    """
    n_trials, n_ch, t_total = band_signal.shape
    if t_total < n_bins:
        raise ValueError("n_timesteps cannot exceed the number of epoch samples")
    # Truncate so T divides evenly into n_bins
    usable = (t_total // n_bins) * n_bins
    trimmed = band_signal[:, :, :usable]
    # Reshape into temporal bins, then compute mean-square power.
    reshaped = trimmed.reshape(n_trials, n_ch, n_bins, -1)
    return np.mean(reshaped ** 2, axis=-1).astype(np.float32, copy=False)


def _fit_thresholds(
    power: np.ndarray, threshold_percentile: float
) -> np.ndarray:
    """Compute per-channel firing thresholds from **training** power.

    Parameters
    ----------
    power : ndarray, shape ``(N, channels, n_bins)``
        Binned band power computed from the training split only.
    threshold_percentile : float
        Percentile (0–100) of the power distribution per channel across all
        training trials and time bins.

    Returns
    -------
    thresholds : ndarray, shape ``(1, channels, 1)``
    """
    if not 0 <= threshold_percentile <= 100:
        raise ValueError("rate_threshold_pct must be between 0 and 100")
    if power.size == 0 or not np.isfinite(power).all():
        raise ValueError("training power must be non-empty and finite")
    return np.percentile(
        power, threshold_percentile, axis=(0, 2), keepdims=True
    ).astype(np.float32)


def _rate_encode_with_thresholds(
    power: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Apply pre-fitted thresholds to produce binary spikes.

    Parameters
    ----------
    power : ndarray, shape ``(N, channels, n_bins)``
    thresholds : ndarray, shape ``(1, channels, 1)``

    Returns
    -------
    spikes : ndarray, shape ``(N, channels, n_bins)`` dtype uint8, values {0, 1}.
    """
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
    spikes : ndarray, shape ``(N, channels, n_bins)`` dtype uint8, values {0, 1}.
    """
    n_trials, n_ch, n_bins = power.shape
    spikes = np.zeros_like(power, dtype=np.uint8)
    peak_bins = np.argmax(power, axis=-1)  # (N, 22)
    # Advanced indexing to set the peak bin to 1
    trials_idx = np.arange(n_trials)[:, None]  # (N, 1)
    ch_idx = np.arange(n_ch)[None, :]          # (1, 22)
    spikes[trials_idx, ch_idx, peak_bins] = 1
    return spikes


def _validate_epochs(
    epochs: np.ndarray, n_timesteps: int
) -> tuple[int, int, int]:
    if epochs.ndim != 3:
        raise ValueError(
            f"epochs must have shape (N, channels, samples), got ndim={epochs.ndim}"
        )
    if n_timesteps < 1:
        raise ValueError("n_timesteps must be at least 1")
    if not np.issubdtype(epochs.dtype, np.number):
        raise ValueError("epochs must contain numeric values")
    if not np.isfinite(epochs).all():
        raise ValueError("epochs must contain only finite values")
    n_trials, n_ch, t_epoch = epochs.shape
    if n_trials < 1 or n_ch < 1:
        raise ValueError("epochs must contain at least one trial and channel")
    if t_epoch < n_timesteps:
        raise ValueError("n_timesteps cannot exceed the number of epoch samples")
    return n_trials, n_ch, t_epoch


def _build_spike_tensor(
    alpha_spikes: np.ndarray,
    beta_spikes: np.ndarray,
    gamma_spikes: np.ndarray,
    n_ch: int,
    n_timesteps: int,
    n_trials: int,
) -> np.ndarray:
    """Concatenate band spikes and transpose to ``(channels*bands, T, N)``."""
    all_spikes = np.concatenate(
        [alpha_spikes, beta_spikes, gamma_spikes], axis=1
    )  # (N, channels * n_bands, n_timesteps)
    spike_tensor = np.transpose(all_spikes, (1, 2, 0))
    assert spike_tensor.shape == (n_ch * N_BANDS, n_timesteps, n_trials)
    return spike_tensor


def _log_result(
    spike_tensor: np.ndarray,
    alpha_spikes: np.ndarray,
    beta_spikes: np.ndarray,
    gamma_spikes: np.ndarray,
) -> float:
    mean_fr = float(spike_tensor.mean())
    logger.info(
        "Spike encoding complete: shape=%s, mean_firing_rate=%.4f (%.1f%%)",
        spike_tensor.shape,
        mean_fr,
        mean_fr * 100,
    )
    alpha_fr = float(alpha_spikes.mean())
    beta_fr  = float(beta_spikes.mean())
    gamma_fr = float(gamma_spikes.mean())
    logger.info(
        "  Per-band firing rates — alpha: %.4f (%.1f%%), "
        "beta: %.4f (%.1f%%), gamma: %.4f (%.1f%%)",
        alpha_fr, alpha_fr * 100,
        beta_fr,  beta_fr  * 100,
        gamma_fr, gamma_fr * 100,
    )
    if mean_fr >= 0.50:
        logger.warning(
            "Mean firing rate %.2f%% exceeds 50%% target — consider raising "
            "rate_threshold_pct or switching to Poisson rate coding.",
            mean_fr * 100,
        )
    return mean_fr


# ── public API ────────────────────────────────────────────────────────────


def fit_encoder(
    train_epochs: np.ndarray,
    sfreq: float,
    *,
    n_timesteps: int = 25,
    rate_threshold_pct: float = 50.0,
) -> EncoderCalibration:
    """Fit rate-encoding thresholds on **training data only**.

    Call this exactly once on the Session 1 training split.  Persist the
    returned :class:`EncoderCalibration` inside the model checkpoint so that
    identical thresholds are applied to validation, held-out test, and online
    Session 2 inference via :func:`apply_encoder`.

    Parameters
    ----------
    train_epochs : ndarray, shape ``(N_train, channels, T_epoch)``
        Z-scored EEG trials from the **training split only**.
    sfreq : float
        True sampling frequency read from ``raw.info["sfreq"]``.
        Do *not* hard-code 250 Hz.
    n_timesteps : int
        Number of temporal bins per trial (default 25).
    rate_threshold_pct : float
        Percentile of the per-channel power distribution used as the firing
        threshold (default 50).

    Returns
    -------
    calibration : EncoderCalibration
        Thresholds to serialise inside the model checkpoint.
    """
    if not isinstance(n_timesteps, int):
        raise ValueError("n_timesteps must be an integer")
    _validate_epochs(train_epochs, n_timesteps)
    _validate_bands(sfreq)

    bands = _decompose_bands(train_epochs, sfreq)
    alpha_power = _band_power(bands[0], n_timesteps)
    beta_power  = _band_power(bands[1], n_timesteps)

    alpha_thresholds = _fit_thresholds(alpha_power, rate_threshold_pct)
    beta_thresholds  = _fit_thresholds(beta_power,  rate_threshold_pct)

    logger.info(
        "Encoder calibrated on %d training trials at sfreq=%.1f Hz "
        "(n_timesteps=%d, threshold_pct=%.1f). "
        "alpha_thresh mean=%.4f, beta_thresh mean=%.4f.",
        train_epochs.shape[0], sfreq, n_timesteps, rate_threshold_pct,
        float(alpha_thresholds.mean()), float(beta_thresholds.mean()),
    )
    return EncoderCalibration(
        alpha_thresholds=alpha_thresholds,
        beta_thresholds=beta_thresholds,
        sfreq=sfreq,
        n_timesteps=n_timesteps,
        rate_threshold_pct=rate_threshold_pct,
        channel_count=train_epochs.shape[1],
    )


def apply_encoder(
    epochs: np.ndarray,
    calibration: EncoderCalibration,
    *,
    sfreq: float | None = None,
) -> EncoderResult:
    """Encode epochs using **pre-fitted** thresholds from :func:`fit_encoder`.

    Use this for every split that is *not* the training set: validation,
    held-out test, and online Session 2 inference.

    Parameters
    ----------
    epochs : ndarray, shape ``(N, channels, T_epoch)``
        Z-scored EEG trials.
    calibration : EncoderCalibration
        Thresholds produced by :func:`fit_encoder` on the training split.

    Returns
    -------
    result : EncoderResult
    """
    if not isinstance(calibration, EncoderCalibration):
        raise TypeError("calibration must be an EncoderCalibration")
    if sfreq is not None and not np.isclose(float(sfreq), calibration.sfreq, atol=1e-3):
        raise ValueError(
            f"sampling frequency mismatch: calibration uses {calibration.sfreq} Hz, "
            f"but input uses {float(sfreq)} Hz"
        )
    _validate_bands(calibration.sfreq)
    if calibration.channel_count < 1 or calibration.n_timesteps < 1:
        raise ValueError("calibration contains invalid channel or timestep metadata")
    if not 0 <= calibration.rate_threshold_pct <= 100:
        raise ValueError("calibration threshold percentile must be between 0 and 100")
    if calibration.alpha_thresholds.shape != (1, calibration.channel_count, 1):
        raise ValueError("alpha calibration thresholds have an invalid shape")
    if calibration.beta_thresholds.shape != (1, calibration.channel_count, 1):
        raise ValueError("beta calibration thresholds have an invalid shape")
    if not np.isfinite(calibration.alpha_thresholds).all() or not np.isfinite(calibration.beta_thresholds).all():
        raise ValueError("calibration thresholds must be finite")
    sfreq = calibration.sfreq
    n_timesteps = calibration.n_timesteps
    n_trials, n_ch, _ = _validate_epochs(epochs, n_timesteps)
    if n_ch != calibration.channel_count:
        raise ValueError(
            f"channel count mismatch: calibration expects {calibration.channel_count}, "
            f"but input has {n_ch}"
        )

    bands  = _decompose_bands(epochs, sfreq)
    powers = np.stack(
        [_band_power(bands[b], n_timesteps) for b in range(N_BANDS)], axis=0
    )

    alpha_spikes = _rate_encode_with_thresholds(powers[0], calibration.alpha_thresholds)
    beta_spikes  = _rate_encode_with_thresholds(powers[1], calibration.beta_thresholds)
    gamma_spikes = _ttfs_encode(powers[2])

    spike_tensor = _build_spike_tensor(
        alpha_spikes, beta_spikes, gamma_spikes, n_ch, n_timesteps, n_trials
    )
    mean_fr = _log_result(spike_tensor, alpha_spikes, beta_spikes, gamma_spikes)
    return EncoderResult(spikes=spike_tensor, mean_firing_rate=mean_fr)


def encode(
    epochs: np.ndarray,
    sfreq: float = 250.0,
    *,
    n_timesteps: int = 25,
    rate_threshold_pct: float = 50.0,
) -> EncoderResult:
    """Encode preprocessed EEG epochs into binary spike trains.

    .. warning::
        This convenience function fits **and** applies thresholds from the same
        batch.  It is correct only when ``epochs`` is a pure training split.
        For validation, test, or online inference, use :func:`fit_encoder` once
        on training data, then :func:`apply_encoder` on every other split to
        avoid threshold leakage.

    Parameters
    ----------
    epochs : ndarray, shape ``(N, channels, T_epoch)``
        Z-scored EEG trials from the preprocessing pipeline.
    sfreq : float
        Sampling frequency in Hz.  Pass ``raw.info["sfreq"]`` rather than
        relying on the default of 250 Hz.
    n_timesteps : int
        Number of temporal bins per trial (default 25).
    rate_threshold_pct : float
        Percentile threshold for rate coding (default 50).

    Returns
    -------
    result : EncoderResult
        ``result.spikes`` — ``(channels * n_bands, n_timesteps, N)`` uint8 tensor.
        ``result.mean_firing_rate`` — scalar in [0, 1].

    Raises
    ------
    ValueError
        If ``epochs`` does not have 3 dimensions, or if any band is invalid for
        the supplied ``sfreq``.
    """
    if not isinstance(n_timesteps, int):
        raise ValueError("n_timesteps must be an integer")
    n_trials, n_ch, _ = _validate_epochs(epochs, n_timesteps)
    _validate_bands(sfreq)

    # 1. Sub-band decomposition → (n_bands, N, channels, T_epoch)
    bands = _decompose_bands(epochs, sfreq)

    # 2. Compute binned power per band → (n_bands, N, channels, n_timesteps)
    powers = np.stack(
        [_band_power(bands[b], n_timesteps) for b in range(N_BANDS)], axis=0
    )

    # 3. Fit thresholds from this batch and encode.
    #    (Only correct when epochs is the training split.)
    alpha_thresholds = _fit_thresholds(powers[0], rate_threshold_pct)
    beta_thresholds  = _fit_thresholds(powers[1], rate_threshold_pct)

    alpha_spikes = _rate_encode_with_thresholds(powers[0], alpha_thresholds)
    beta_spikes  = _rate_encode_with_thresholds(powers[1], beta_thresholds)
    gamma_spikes = _ttfs_encode(powers[2])

    spike_tensor = _build_spike_tensor(
        alpha_spikes, beta_spikes, gamma_spikes, n_ch, n_timesteps, n_trials
    )
    mean_fr = _log_result(spike_tensor, alpha_spikes, beta_spikes, gamma_spikes)
    return EncoderResult(spikes=spike_tensor, mean_firing_rate=mean_fr)
