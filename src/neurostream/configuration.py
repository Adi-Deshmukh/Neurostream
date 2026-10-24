"""Validated project configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    dataset_name: str
    dataset_path: Path | None
    cache_path: Path
    sampling_frequency_hz: float
    epoch_tmin: float
    epoch_tmax: float
    filter_l_freq_hz: float
    filter_h_freq_hz: float
    encoder_timesteps: int
    encoder_channel_count: int
    encoder_rate_threshold_pct: float
    encoder_filter_order: int
    model_in_features: int
    model_hidden: int
    model_out_features: int
    model_beta: float
    model_threshold: float
    training_epochs: int
    training_batch_size: int
    training_learning_rate: float
    validation_fraction: float
    random_seed: int
    evaluation_confidence_threshold: float
    adaptation_momentum: float
    output_checkpoint_dir: Path
    output_evaluation_dir: Path
    output_log_dir: Path


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive(value: Any, name: str, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value) if integer else float(value)


def _fraction(value: Any, name: str) -> float:
    value = _positive(value, name)
    if value > 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return float(value)


def _unit_interval(value: Any, name: str, *, exclusive_upper: bool = False) -> float:
    value = _positive(value, name)
    if value >= 1 if exclusive_upper else value > 1:
        bound = "less than 1" if exclusive_upper else "at most 1"
        raise ValueError(f"{name} must be in (0, {bound})")
    return float(value)


def load_config(path: str | Path = "configs/config.yaml") -> ProjectConfig:
    """Load and validate a complete project configuration.

    Invalid or missing settings fail before any dataset or model work starts.
    Relative paths are resolved against the configuration file's directory.
    """
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration {config_path}: {exc}") from exc
    root = _mapping(raw, "configuration")
    data = _mapping(root.get("data"), "data")
    preprocessing = _mapping(root.get("preprocessing"), "preprocessing")
    encoder = _mapping(root.get("encoder"), "encoder")
    model = _mapping(root.get("model"), "model")
    training = _mapping(root.get("training"), "training")
    evaluation = _mapping(root.get("evaluation"), "evaluation")
    output = _mapping(root.get("output"), "output")

    sfreq = _positive(data.get("sampling_frequency_hz"), "data.sampling_frequency_hz")
    tmin = data.get("epoch_tmin")
    tmax = data.get("epoch_tmax")
    if not isinstance(tmin, (int, float)) or not isinstance(tmax, (int, float)) or tmin >= tmax:
        raise ValueError("data epoch window must contain numeric epoch_tmin < epoch_tmax")
    l_freq = _positive(preprocessing.get("filter_l_freq_hz"), "preprocessing.filter_l_freq_hz")
    h_freq = _positive(preprocessing.get("filter_h_freq_hz"), "preprocessing.filter_h_freq_hz")
    if not l_freq < h_freq < sfreq / 2:
        raise ValueError("preprocessing filter cutoffs must satisfy 0 < low < high < Nyquist")

    dataset_name = data.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("data.dataset_name must be a non-empty string")

    base = config_path.parent
    def relative_path(value: Any, name: str, allow_none: bool = False) -> Path | None:
        if value is None and allow_none:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty path")
        result = Path(value)
        return result if result.is_absolute() else (base / result).resolve()

    encoder_channels = int(_positive(encoder.get("channel_count"), "encoder.channel_count", True))
    model_in_features = int(_positive(model.get("in_features"), "model.in_features", True))
    if model_in_features != encoder_channels * 3:
        raise ValueError("model.in_features must equal encoder.channel_count * 3")
    percentile = _positive(
        encoder.get("rate_threshold_pct"), "encoder.rate_threshold_pct"
    )
    if percentile >= 100:
        raise ValueError("encoder.rate_threshold_pct must be less than 100")
    return ProjectConfig(
        dataset_name=dataset_name,
        dataset_path=relative_path(data.get("dataset_path"), "data.dataset_path", True),
        cache_path=relative_path(data.get("cache_path"), "data.cache_path"),
        sampling_frequency_hz=float(sfreq),
        epoch_tmin=float(tmin),
        epoch_tmax=float(tmax),
        filter_l_freq_hz=float(l_freq),
        filter_h_freq_hz=float(h_freq),
        encoder_timesteps=int(_positive(encoder.get("timesteps"), "encoder.timesteps", True)),
        encoder_channel_count=encoder_channels,
        encoder_rate_threshold_pct=float(percentile),
        encoder_filter_order=int(_positive(encoder.get("filter_order"), "encoder.filter_order", True)),
        model_in_features=model_in_features,
        model_hidden=int(_positive(model.get("hidden"), "model.hidden", True)),
        model_out_features=int(_positive(model.get("out_features"), "model.out_features", True)),
        model_beta=_unit_interval(model.get("beta"), "model.beta", exclusive_upper=True),
        model_threshold=float(_positive(model.get("threshold"), "model.threshold")),
        training_epochs=int(_positive(training.get("epochs"), "training.epochs", True)),
        training_batch_size=int(_positive(training.get("batch_size"), "training.batch_size", True)),
        training_learning_rate=float(_positive(training.get("learning_rate"), "training.learning_rate")),
        validation_fraction=_unit_interval(
            training.get("validation_fraction"),
            "training.validation_fraction",
            exclusive_upper=True,
        ),
        random_seed=int(_positive(training.get("seed"), "training.seed", True)),
        evaluation_confidence_threshold=_fraction(
            evaluation.get("confidence_threshold"), "evaluation.confidence_threshold"
        ),
        adaptation_momentum=_fraction(evaluation.get("adaptation_momentum"), "evaluation.adaptation_momentum"),
        output_checkpoint_dir=relative_path(output.get("checkpoint_dir"), "output.checkpoint_dir"),
        output_evaluation_dir=relative_path(output.get("evaluation_dir"), "output.evaluation_dir"),
        output_log_dir=relative_path(output.get("log_dir"), "output.log_dir"),
    )
