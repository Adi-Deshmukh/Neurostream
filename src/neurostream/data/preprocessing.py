"""Filtering, epoch extraction, baseline correction, and normalization."""

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
	data: np.ndarray, sfreq: float, l_freq: float = 4.0, h_freq: float = 40.0
) -> np.ndarray:
	"""Apply a zero-phase Butterworth bandpass to ``(channels, samples)`` data."""
	if data.ndim != 2:
		raise ValueError("data must have shape (channels, samples)")
	if not 0 < l_freq < h_freq < sfreq / 2:
		raise ValueError("cutoffs must satisfy 0 < l_freq < h_freq < Nyquist")
	sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
	return sosfiltfilt(sos, data, axis=-1).astype(np.float32, copy=False)


def zscore_channels(epochs: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
	"""Z-score each channel independently within each trial."""
	if epochs.ndim != 3:
		raise ValueError("epochs must have shape (trials, channels, samples)")
	mean = epochs.mean(axis=-1, keepdims=True)
	std = epochs.std(axis=-1, keepdims=True)
	return ((epochs - mean) / np.maximum(std, epsilon)).astype(np.float32, copy=False)


def preprocess_session(
	raw: Any,
	*,
	tmin: float = 2.0,
	tmax: float = 6.0,
	l_freq: float = 4.0,
	h_freq: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
	"""Convert one MNE Raw run into normalized EEG trials and metadata."""
	import mne

	sfreq = float(raw.info["sfreq"])
	filtered = raw.copy().pick("eeg")
	filtered._data = bandpass_filter(filtered.get_data(), sfreq, l_freq, h_freq)
	events, event_id = mne.events_from_annotations(filtered, verbose=False)
	if not event_id:
		raise ValueError("raw run contains no annotated events")

	epoch_end = tmax - 1.0 / sfreq
	epochs = mne.Epochs(
		filtered,
		events,
		event_id=event_id,
		tmin=tmin,
		tmax=epoch_end,
		baseline=None,
		preload=True,
		reject_by_annotation=True,
		verbose=False,
	)
	data = epochs.get_data()
	data -= data.mean(axis=-1, keepdims=True)
	data = zscore_channels(data)
	inverse_event_id = {code: label for label, code in epochs.event_id.items()}
	metadata = [
		{"event": inverse_event_id[int(code)], "event_code": int(code), "trial": index}
		for index, code in enumerate(epochs.events[:, 2])
	]
	return data, epochs.events[:, 2].astype(np.int64), metadata
