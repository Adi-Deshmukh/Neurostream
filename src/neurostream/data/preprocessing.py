"""Filtering, epoch extraction, baseline correction, and normalization."""

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

MOTOR_IMAGERY_EVENT_ALIASES = {
	"769": ("769", "left_hand"),
	"770": ("770", "right_hand"),
	"771": ("771", "feet"),
	"772": ("772", "tongue"),
}


class EEGChannelNormalizer:
	"""Training-fitted channel scaling shared by every evaluation session."""

	def __init__(self, epsilon: float = 1e-6) -> None:
		if not np.isfinite(epsilon) or epsilon <= 0:
			raise ValueError("epsilon must be finite and positive")
		self.epsilon = float(epsilon)
		self.mean_: np.ndarray | None = None
		self.std_: np.ndarray | None = None

	def fit(self, epochs: np.ndarray) -> "EEGChannelNormalizer":
		epochs = _validate_epochs(epochs)
		self.mean_ = epochs.mean(axis=(0, 2), keepdims=True).astype(np.float32)
		self.std_ = np.maximum(
			epochs.std(axis=(0, 2), keepdims=True), self.epsilon
		).astype(np.float32)
		return self

	def transform(self, epochs: np.ndarray) -> np.ndarray:
		epochs = _validate_epochs(epochs)
		if self.mean_ is None or self.std_ is None:
			raise RuntimeError("normalizer must be fitted before transform")
		if epochs.shape[1] != self.mean_.shape[1]:
			raise ValueError("EEG channel count mismatch")
		return ((epochs - self.mean_) / self.std_).astype(np.float32, copy=False)

	def payload(self) -> dict[str, Any]:
		if self.mean_ is None or self.std_ is None:
			raise RuntimeError("normalizer must be fitted before serialization")
		return {
			"method": "training_fitted_channel_zscore",
			"epsilon": self.epsilon,
			"mean": self.mean_.reshape(-1).tolist(),
			"std": self.std_.reshape(-1).tolist(),
		}


def _validate_epochs(epochs: np.ndarray) -> np.ndarray:
	epochs = np.asarray(epochs, dtype=np.float32)
	if epochs.ndim != 3:
		raise ValueError("epochs must have shape (trials, channels, samples)")
	if min(epochs.shape) < 1:
		raise ValueError("epochs must contain at least one trial, channel, and sample")
	if not np.isfinite(epochs).all():
		raise ValueError("epochs must contain only finite values")
	return epochs


def bandpass_filter(
	data: np.ndarray, sfreq: float, l_freq: float = 4.0, h_freq: float = 100.0
) -> np.ndarray:
	"""Apply an offline, zero-phase Butterworth bandpass.

	This pipeline is epoch-based, not causal streaming: ``sosfiltfilt`` uses
	samples on both sides of every point. A causal filter with carried state is
	required before using this preprocessing contract for live sample-by-sample
	inference.
	"""
	if data.ndim != 2:
		raise ValueError("data must have shape (channels, samples)")
	if data.shape[0] < 1 or data.shape[1] < 1:
		raise ValueError("data must contain at least one channel and sample")
	if not np.isfinite(data).all():
		raise ValueError("data must contain only finite values")
	if not np.isfinite(sfreq) or sfreq <= 0:
		raise ValueError("sfreq must be finite and positive")
	if not 0 < l_freq < h_freq < sfreq / 2:
		raise ValueError("cutoffs must satisfy 0 < l_freq < h_freq < Nyquist")
	sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
	return sosfiltfilt(sos, data, axis=-1).astype(np.float32, copy=False)


def zscore_channels(epochs: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
	"""Z-score each channel independently within each trial."""
	epochs = _validate_epochs(epochs)
	mean = epochs.mean(axis=-1, keepdims=True)
	std = epochs.std(axis=-1, keepdims=True)
	return ((epochs - mean) / np.maximum(std, epsilon)).astype(np.float32, copy=False)


def preprocess_session(
	raw: Any,
	*,
	tmin: float = 2.0,
	tmax: float = 6.0,
	l_freq: float = 4.0,
	h_freq: float = 100.0,
	normalization: str = "demean",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
	"""Convert one MNE Raw run into demeaned EEG trials and metadata.

	For BNCI2014-001, MOABB's trigger events represent trial start. The
	experiment presents the class cue two seconds later and ends imagery at
	 six seconds, so the configured ``[2, 6]`` window is cue-relative ``[0, 4]``.
	"""
	import mne

	sfreq = float(raw.info["sfreq"])
	if not np.isfinite(sfreq) or sfreq <= 0:
		raise ValueError("raw data must provide a finite positive sampling frequency")
	if tmin >= tmax or l_freq <= 0 or h_freq <= l_freq:
		raise ValueError("invalid epoch window or filter settings")
	filtered = raw.copy().pick("eeg")
	filtered._data = bandpass_filter(filtered.get_data(), sfreq, l_freq, h_freq)
	events, event_id = mne.events_from_annotations(filtered, verbose=False)
	if not event_id:
		raise ValueError("raw run contains no annotated events")
	selected_events = {
		canonical: next(
			(event_id[alias] for alias in aliases if alias in event_id),
			None,
		)
		for canonical, aliases in MOTOR_IMAGERY_EVENT_ALIASES.items()
	}
	missing_labels = [label for label, code in selected_events.items() if code is None]
	if missing_labels:
		raise ValueError(
			"raw run is missing expected motor-imagery event labels: "
			+ ", ".join(missing_labels)
		)
	imagery_event_id = {
		label: int(code) for label, code in selected_events.items()
		if code is not None
	}

	epoch_end = tmax - 1.0 / sfreq
	epochs = mne.Epochs(
		filtered,
		events,
		event_id=imagery_event_id,
		tmin=tmin,
		tmax=epoch_end,
		baseline=None,
		preload=True,
		reject_by_annotation=True,
		verbose=False,
	)
	data = epochs.get_data()
	if data.ndim != 3 or data.shape[0] == 0:
		raise ValueError("preprocessing produced no valid epochs")
	data -= data.mean(axis=-1, keepdims=True)
	data = data.astype(np.float32, copy=False)
	if normalization == "per_trial_zscore":
		data = zscore_channels(data)
	elif normalization != "demean":
		raise ValueError(
			"normalization must be 'demean' or 'per_trial_zscore'"
		)
	inverse_event_id = {code: label for label, code in epochs.event_id.items()}
	class_counts = {
		label: int(np.sum(epochs.events[:, 2] == code))
		for label, code in epochs.event_id.items()
	}
	metadata = [
		{
			"event": inverse_event_id[int(code)],
			"event_code": int(code),
			"trial": index,
			"class_trial_counts": class_counts,
		}
		for index, code in enumerate(epochs.events[:, 2])
	]
	return data, epochs.events[:, 2].astype(np.int64), metadata
