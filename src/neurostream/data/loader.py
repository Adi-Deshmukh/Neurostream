"""MOABB wrapper for the BNCI 2014-001 motor-imagery dataset."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class SessionData:
	"""Preprocessed examples and trial metadata for one session."""

	X: np.ndarray
	y: np.ndarray
	metadata: list[dict[str, Any]]
	sfreq: float = 250.0
	"""True sampling frequency in Hz, read from ``raw.info["sfreq"]``."""
	channel_names: tuple[str, ...] = ()
	tmin: float = 2.0
	tmax: float = 6.0
	l_freq: float = 4.0
	h_freq: float = 100.0
	epoch_reference: str = "trial_start"
	trial_counts: dict[str, int] = field(default_factory=dict)
	normalization: dict[str, Any] | None = None


def load_bnci2014_001(
	subjects: int | Iterable[int] = 1,
	*,
	dataset: Any | None = None,
	sessions: Iterable[str] | None = None,
	tmin: float = 2.0,
	tmax: float = 6.0,
	l_freq: float = 4.0,
	h_freq: float = 100.0,
	dataset_path: str | None = None,
	cache_path: str | None = None,
	epoch_reference: str = "trial_start",
	normalization: str = "training_fitted_channel_zscore",
) -> dict[str, SessionData]:
	"""Load BNCI2014_001 and return trials grouped by MOABB session.

	``dataset`` may be a MOABB-compatible object, which keeps fixture tests
	independent of network downloads.

	Each returned :class:`SessionData` carries the true ``sfreq`` extracted
	from the first run's MNE ``Raw`` object.  All runs in one session must
	share the same sampling frequency; a :exc:`ValueError` is raised if they
	differ.
	"""
	if dataset_path is not None:
		import mne
		Path(dataset_path).mkdir(parents=True, exist_ok=True)
		mne.set_config("MNE_DATA", str(dataset_path), set_env=True)
	if dataset is None:
		from moabb.datasets import BNCI2014_001

		dataset = BNCI2014_001()

	subject_ids = [subjects] if isinstance(subjects, int) else list(subjects)
	requested_sessions = set(sessions) if sessions is not None else None
	cache_config = None
	if cache_path is not None:
		from moabb.datasets.base import CacheConfig
		Path(cache_path).mkdir(parents=True, exist_ok=True)
		cache_config = CacheConfig(path=str(cache_path), use=True)
		get_data_params = __import__("inspect").signature(dataset.get_data).parameters
		if "cache_config" in get_data_params:
			raw_data = dataset.get_data(subjects=subject_ids, cache_config=cache_config)
		else:
			raw_data = dataset.get_data(subjects=subject_ids)
	else:
		raw_data = dataset.get_data(subjects=subject_ids)
	from .preprocessing import preprocess_session

	grouped: dict[str, list[tuple[Any, int, str]]] = {}
	for subject_id in subject_ids:
		for session_name, runs in raw_data[subject_id].items():
			if requested_sessions is not None and session_name not in requested_sessions:
				continue
			for run_name, raw in runs.items():
				grouped.setdefault(session_name, []).append((raw, subject_id, run_name))

	result: dict[str, SessionData] = {}
	for session_name, raw_runs in grouped.items():
		arrays, labels, metadata = [], [], []
		trial_counts: dict[str, int] = {}
		session_sfreq: float | None = None
		channel_names: tuple[str, ...] | None = None
		for raw, subject_id, run_name in raw_runs:
			# Extract the true sampling frequency from every run and confirm
			# they are identical within this session.
			run_sfreq = float(raw.info["sfreq"])
			run_channel_names = tuple(raw.copy().pick("eeg").ch_names)
			if channel_names is None:
				channel_names = run_channel_names
			elif run_channel_names != channel_names:
				raise ValueError(
					f"Session '{session_name}' contains runs with mismatched EEG channel order."
				)
			if session_sfreq is None:
				session_sfreq = run_sfreq
			elif abs(run_sfreq - session_sfreq) > 1e-3:
				raise ValueError(
					f"Session '{session_name}' contains runs with mismatched "
					f"sampling frequencies: {session_sfreq} Hz vs {run_sfreq} Hz."
				)

			X, y, run_metadata = preprocess_session(
				raw, tmin=tmin, tmax=tmax, l_freq=l_freq, h_freq=h_freq,
				normalization="per_trial_zscore" if normalization == "per_trial_zscore" else "demean",
			)
			arrays.append(X)
			labels.append(y)
			metadata.extend(
				{**item, "subject": subject_id, "session": session_name, "run": run_name}
				for item in run_metadata
			)
			for item in run_metadata:
				label = str(item["event"])
				trial_counts[label] = trial_counts.get(label, 0) + 1
		if not arrays:
			raise ValueError(f"Session '{session_name}' contains no runs")
		result[session_name] = SessionData(
			X=np.concatenate(arrays, axis=0).astype(np.float32, copy=False),
			y=np.concatenate(labels, axis=0),
			metadata=metadata,
			sfreq=session_sfreq if session_sfreq is not None else 250.0,
			channel_names=channel_names or (),
			tmin=tmin,
			tmax=tmax,
			l_freq=l_freq,
			h_freq=h_freq,
			epoch_reference=epoch_reference,
			trial_counts=trial_counts,
		)
	from .preprocessing import EEGChannelNormalizer
	train_name = next(
		(name for name in result if "train" in name.lower()), next(iter(result), None)
	)
	if train_name is None:
		raise ValueError("no sessions were available for normalization")
	if normalization == "training_fitted_channel_zscore":
		normalizer = EEGChannelNormalizer().fit(result[train_name].X)
		for session in result.values():
			session.X = normalizer.transform(session.X)
		normalization_payload = normalizer.payload()
	elif normalization == "per_trial_zscore":
		normalization_payload = {"method": "per_trial_channel_zscore"}
	elif normalization == "demean":
		normalization_payload = {"method": "per_trial_demean_preserve_channel_scale"}
	else:
		raise ValueError(
			"normalization must be 'training_fitted_channel_zscore', "
			"'per_trial_zscore', or 'demean'"
		)
	for session in result.values():
		session.normalization = normalization_payload
	return result


def load_data(*args: Any, **kwargs: Any) -> dict[str, SessionData]:
	"""Short alias for :func:`load_bnci2014_001`."""
	return load_bnci2014_001(*args, **kwargs)
