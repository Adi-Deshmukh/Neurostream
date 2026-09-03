"""MOABB wrapper for the BNCI 2014-001 motor-imagery dataset."""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class SessionData:
	"""Preprocessed examples and trial metadata for one session."""

	X: np.ndarray
	y: np.ndarray
	metadata: list[dict[str, Any]]


def load_bnci2014_001(
	subjects: int | Iterable[int] = 1,
	*,
	dataset: Any | None = None,
	sessions: Iterable[str] | None = None,
	tmin: float = 2.0,
	tmax: float = 6.0,
	l_freq: float = 4.0,
	h_freq: float = 40.0,
) -> dict[str, SessionData]:
	"""Load BNCI2014_001 and return trials grouped by MOABB session.

	``dataset`` may be a MOABB-compatible object, which keeps fixture tests
	independent of network downloads.
	"""
	if dataset is None:
		from moabb.datasets import BNCI2014_001

		dataset = BNCI2014_001()

	subject_ids = [subjects] if isinstance(subjects, int) else list(subjects)
	requested_sessions = set(sessions) if sessions is not None else None
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
		for raw, subject_id, run_name in raw_runs:
			X, y, run_metadata = preprocess_session(
				raw, tmin=tmin, tmax=tmax, l_freq=l_freq, h_freq=h_freq
			)
			arrays.append(X)
			labels.append(y)
			metadata.extend(
				{**item, "subject": subject_id, "session": session_name, "run": run_name}
				for item in run_metadata
			)
		result[session_name] = SessionData(
			X=np.concatenate(arrays, axis=0).astype(np.float32, copy=False),
			y=np.concatenate(labels, axis=0),
			metadata=metadata,
		)
	return result


def load_data(*args: Any, **kwargs: Any) -> dict[str, SessionData]:
	"""Short alias for :func:`load_bnci2014_001`."""
	return load_bnci2014_001(*args, **kwargs)
