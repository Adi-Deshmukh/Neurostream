"""Reproducible multi-subject cross-session evaluation running Phase A + Phase B.

Runs the complete lifecycle across subjects and seeds:
1. Supervised Phase A SNN training on Session 1 -> freezes model weights.
2. Extracts initial class prototypes from Session 1.
3. Online Phase B adaptation on Session 2 using frozen momentum/alpha with sleep consolidation.
4. Measures static target baseline, prequential target adaptation, retrospective
   target diagnostics, explicit source retention, and LIF spike activity.
5. Aggregates results into a reproducible table and CSV.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import logging
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from neurostream.data.loader import load_bnci2014_001
from neurostream.data.spike_encoder import apply_encoder, fit_encoder
from neurostream.evaluation.metrics import (
    EvaluationMetrics,
    format_metrics_table,
    measure_model_sparsity,
)
from neurostream.evaluation.artifacts import save_evaluation_artifact
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.training.phase_a_train import evaluate_phase_a, train_phase_a
from neurostream.training.phase_b_adapt import adapt_target_session, score_adaptation
from neurostream.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@dataclass
class SubjectRunResult:
    """Full evaluation metrics for one subject and one random seed."""

    subject: int
    seed: int
    phase_a_train_acc: float
    phase_a_val_acc: float
    target_static_baseline_acc: float
    target_prequential_acc: float
    target_retrospective_acc: float
    adaptation_gain: float
    source_accuracy_before_adaptation: float
    source_accuracy_after_adaptation: float
    source_retention_delta: float
    spike_sparsity: float
    accepted_trials: int
    total_target_trials: int
    consolidation_phases: int
    consolidation_enabled: bool

    @property
    def pre_adaptation_acc(self) -> float:
        return self.target_static_baseline_acc

    @property
    def post_adaptation_acc(self) -> float:
        return self.target_prequential_acc


@dataclass
class SweepSummary:
    """Aggregate statistics across multiple subject-seed runs."""

    runs: list[SubjectRunResult]
    mean_pre_acc: float
    mean_post_acc: float
    mean_gain: float
    mean_retention: float
    mean_forgetting: float
    mean_sparsity: float
    std_post_acc: float

    def to_evaluation_metrics(self) -> list[EvaluationMetrics]:
        """Convert all runs into a list of EvaluationMetrics for tabular display."""
        metrics_list = []
        for r in self.runs:
            metrics_list.append(
                EvaluationMetrics(
                    subject=f"S{r.subject}-s{r.seed}",
                    session="cross-session",
                    pre_adaptation_acc=r.target_static_baseline_acc,
                    post_adaptation_acc=r.target_prequential_acc,
                    adaptation_gain=r.adaptation_gain,
                    initial_retention_acc=r.source_accuracy_before_adaptation,
                    post_retention_acc=r.source_accuracy_after_adaptation,
                    forgetting=-r.source_retention_delta,
                    retention_rate=(
                        r.source_accuracy_after_adaptation / r.source_accuracy_before_adaptation
                        if r.source_accuracy_before_adaptation > 0 else 0.0
                    ),
                    spike_sparsity=r.spike_sparsity,
                    consolidation_enabled=r.consolidation_enabled,
                    acceptance_rate=r.accepted_trials / r.total_target_trials if r.total_target_trials > 0 else 0.0,
                )
            )
        return metrics_list


def run_subject_cross_session(
    subject: int,
    seed: int = 42,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    augmentation_probability: float = 0.2,
    hidden: int = 256,
    model_beta: float = 0.9,
    model_threshold: float = 0.5,
    model_out_features: int = 128,
    encoder_timesteps: int = 25,
    encoder_rate_threshold_pct: float = 50.0,
    expected_sampling_frequency_hz: float = 250.0,
    epoch_tmin: float = 2.0,
    epoch_tmax: float = 6.0,
    epoch_reference: str = "trial_start",
    cue_onset_offset_seconds: float = 2.0,
    filter_l_freq_hz: float = 4.0,
    filter_h_freq_hz: float = 100.0,
    output_evaluation_dir: str | Path | None = None,
    dataset_name: str = "BNCI2014_001",
    dataset_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    momentum: float = 0.95,
    confidence_threshold: float = 0.8,
    enable_consolidation: bool = True,
    consolidation_interval: int = 50,
    sleep_steps: int = 5,
    noise_std: float = 0.05,
    pull_rate: float = 0.5,
    checkpoint_dir: str | Path | None = None,
    dataset: Any | None = None,
) -> SubjectRunResult:
    """Execute complete Phase A + Phase B lifecycle for one subject and seed."""
    seed_everything(seed)

    # 1. Load Sessions
    if dataset_name != "BNCI2014_001":
        raise ValueError(f"unsupported dataset_name: {dataset_name}")
    sessions = load_bnci2014_001(
        subjects=subject,
        dataset=dataset,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        cache_path=str(cache_path) if cache_path is not None else None,
        tmin=epoch_tmin,
        tmax=epoch_tmax,
        l_freq=filter_l_freq_hz,
        h_freq=filter_h_freq_hz,
        epoch_reference=epoch_reference,
    )
    train_name = next(name for name in sessions if "train" in name.lower())
    test_name = next(name for name in sessions if "test" in name.lower())

    train_session = sessions[train_name]
    test_session = sessions[test_name]

    if not np.isclose(train_session.sfreq, expected_sampling_frequency_hz, atol=1e-3):
        raise ValueError(
            "training session sampling frequency does not match configuration: "
            f"{train_session.sfreq} Hz vs {expected_sampling_frequency_hz} Hz"
        )
    if not np.isclose(train_session.sfreq, test_session.sfreq, atol=1e-3):
        raise ValueError(
            "cross-session evaluation requires matching sampling frequencies; "
            f"got {train_session.sfreq} Hz and {test_session.sfreq} Hz"
        )
    calibration = fit_encoder(
        train_session.X,
        sfreq=train_session.sfreq,
        n_timesteps=encoder_timesteps,
        rate_threshold_pct=encoder_rate_threshold_pct,
    )
    train_spikes = apply_encoder(
        train_session.X, calibration, sfreq=train_session.sfreq
    ).spikes
    test_spikes = apply_encoder(
        test_session.X, calibration, sfreq=test_session.sfreq
    ).spikes

    # 3. Phase A: Supervised training & freezing
    in_features = train_spikes.shape[0]
    model = SNNFeatureExtractor(
        in_features=in_features,
        hidden=hidden,
        out_features=model_out_features,
        beta=model_beta,
        threshold=model_threshold,
    )

    save_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path("results/checkpoints/sweep") / f"sub_{subject}_seed_{seed}"

    event_names = {
        int(item["event_code"]): str(item["event"])
        for item in train_session.metadata
        if "event_code" in item and "event" in item
    }
    phase_a = train_phase_a(
        train_spikes,
        train_session.y,
        subject=subject,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        augmentation_probability=augmentation_probability,
        model=model,
        checkpoint_dir=save_dir,
        encoder_calibration=calibration,
        label_names=[
            event_names.get(int(value), str(value))
            for value in np.unique(train_session.y)
        ],
        preprocessing_contract={
            "sampling_frequency_hz": expected_sampling_frequency_hz,
            "filter_settings": {
                "l_freq_hz": filter_l_freq_hz,
                "h_freq_hz": filter_h_freq_hz,
                "order": 4,
            },
            "epoch_window_seconds": {
                "tmin": epoch_tmin,
                "tmax": epoch_tmax,
                "reference": epoch_reference,
                "cue_onset_offset_seconds": cue_onset_offset_seconds,
            },
            "channel_order": list(train_session.channel_names),
            "effective_configuration": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "validation_fraction": validation_fraction,
                "augmentation_probability": augmentation_probability,
                "encoder_timesteps": encoder_timesteps,
                "encoder_rate_threshold_pct": encoder_rate_threshold_pct,
                "model_hidden": hidden,
                "model_beta": model_beta,
                "model_threshold": model_threshold,
                "model_out_features": model_out_features,
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path) if dataset_path else None,
                "cache_path": str(cache_path) if cache_path else None,
                "sampling_frequency_hz": expected_sampling_frequency_hz,
                "epoch_tmin": epoch_tmin,
                "epoch_tmax": epoch_tmax,
                "filter_l_freq_hz": filter_l_freq_hz,
                "filter_h_freq_hz": filter_h_freq_hz,
                "epoch_reference": epoch_reference,
                "cue_onset_offset_seconds": cue_onset_offset_seconds,
            },
        },
    )
    phase_a_eval = evaluate_phase_a(
        phase_a.model,
        phase_a.prototypes,
        test_spikes,
        test_session.y,
        phase_a.label_values,
        temperature=0.1,
        train_spikes=train_spikes,
        train_labels=train_session.y,
        linear_head_epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    # 4. Measure Neuromorphic Spike Sparsity via Forward Hooks
    target_spikes_tensor = torch.as_tensor(test_spikes, dtype=torch.float32)
    mean_sparsity, _ = measure_model_sparsity(phase_a.model, target_spikes_tensor)

    # 5. Phase B: Streaming adaptation on target session
    # Map raw labels (e.g. 1, 2, 3, 4) to 0-indexed consecutive class indices
    label_tensor = torch.as_tensor(phase_a.label_values)
    target_labels = torch.as_tensor(
        [torch.where(label_tensor == int(lbl))[0].item() for lbl in test_session.y],
        dtype=torch.long,
    )
    source_labels = torch.as_tensor(
        [torch.where(label_tensor == int(lbl))[0].item() for lbl in train_session.y],
        dtype=torch.long,
    )

    phase_b = adapt_target_session(
        phase_a.model,
        phase_a.prototypes,
        test_spikes,
        momentum=momentum,
        confidence_threshold=confidence_threshold,
        batch_size=batch_size,
        source_spikes=train_spikes,
        enable_consolidation=enable_consolidation,
        consolidation_interval=consolidation_interval,
        sleep_steps=sleep_steps,
        noise_std=noise_std,
        pull_rate=pull_rate,
    )
    phase_b = score_adaptation(phase_b, target_labels, source_labels)

    gain = phase_b.prequential_accuracy - phase_b.static_baseline_accuracy

    result = SubjectRunResult(
        subject=subject,
        seed=seed,
        phase_a_train_acc=phase_a.train_accuracy,
        phase_a_val_acc=phase_a.validation_accuracy,
        target_static_baseline_acc=phase_b.static_baseline_accuracy,
        target_prequential_acc=phase_b.prequential_accuracy,
        target_retrospective_acc=phase_b.retrospective_accuracy,
        adaptation_gain=gain,
        source_accuracy_before_adaptation=phase_b.source_accuracy_before_adaptation,
        source_accuracy_after_adaptation=phase_b.source_accuracy_after_adaptation,
        source_retention_delta=phase_b.source_retention_delta,
        spike_sparsity=mean_sparsity,
        accepted_trials=phase_b.accepted_trials,
        total_target_trials=test_spikes.shape[-1],
        consolidation_phases=phase_b.consolidation_phases,
        consolidation_enabled=enable_consolidation,
    )
    evaluation_dir = (
        Path(output_evaluation_dir)
        if output_evaluation_dir is not None
        else save_dir
    )
    save_evaluation_artifact(
        evaluation_dir / f"subject_{subject}_seed_{seed}.json",
        checkpoint_path=phase_a.checkpoint_path,
        subject=subject,
        train_session=train_name,
        evaluation_session=test_name,
        metrics=asdict(result),
        label_values=phase_a.label_values.tolist(),
        label_names=[
            event_names.get(int(value), str(value))
            for value in phase_a.label_values
        ],
        run_config={
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "momentum": momentum,
            "confidence_threshold": confidence_threshold,
            "encoder_timesteps": encoder_timesteps,
            "encoder_rate_threshold_pct": encoder_rate_threshold_pct,
            "model_beta": model_beta,
            "model_threshold": model_threshold,
            "model_out_features": model_out_features,
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path) if dataset_path else None,
            "cache_path": str(cache_path) if cache_path else None,
            "sampling_frequency_hz": expected_sampling_frequency_hz,
            "epoch_tmin": epoch_tmin,
            "epoch_tmax": epoch_tmax,
            "filter_l_freq_hz": filter_l_freq_hz,
            "filter_h_freq_hz": filter_h_freq_hz,
            "epoch_reference": epoch_reference,
            "cue_onset_offset_seconds": cue_onset_offset_seconds,
        },
        run_id=f"subject-{subject}-seed-{seed}",
    )
    return result


def run_cross_session_sweep(
    subjects: Iterable[int] = range(1, 10),
    seeds: Iterable[int] = (42, 123, 456, 789, 999),
    *,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    augmentation_probability: float = 0.2,
    hidden: int = 256,
    model_beta: float = 0.9,
    model_threshold: float = 0.5,
    encoder_timesteps: int = 25,
    encoder_rate_threshold_pct: float = 50.0,
    expected_sampling_frequency_hz: float = 250.0,
    epoch_tmin: float = 2.0,
    epoch_tmax: float = 6.0,
    epoch_reference: str = "trial_start",
    cue_onset_offset_seconds: float = 2.0,
    filter_l_freq_hz: float = 4.0,
    filter_h_freq_hz: float = 100.0,
    model_out_features: int = 128,
    output_evaluation_dir: str | Path | None = None,
    dataset_name: str = "BNCI2014_001",
    dataset_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    momentum: float = 0.95,
    confidence_threshold: float = 0.8,
    enable_consolidation: bool = True,
    consolidation_interval: int = 50,
    sleep_steps: int = 5,
    noise_std: float = 0.05,
    pull_rate: float = 0.5,
    checkpoint_dir: str | Path = "results/checkpoints/sweep",
    output_csv: str | Path | None = "results/logs/cross_session_sweep.csv",
    dataset: Any | None = None,
) -> SweepSummary:
    """Run cross-session evaluation across subjects and seeds, then save CSV."""
    runs: list[SubjectRunResult] = []
    for subj in subjects:
        for seed in seeds:
            logger.info("Executing sweep: Subject %d, Seed %d", subj, seed)
            res = run_subject_cross_session(
                subject=subj,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                augmentation_probability=augmentation_probability,
                hidden=hidden,
                model_beta=model_beta,
                model_threshold=model_threshold,
                encoder_timesteps=encoder_timesteps,
                encoder_rate_threshold_pct=encoder_rate_threshold_pct,
                expected_sampling_frequency_hz=expected_sampling_frequency_hz,
                epoch_tmin=epoch_tmin,
                epoch_tmax=epoch_tmax,
                epoch_reference=epoch_reference,
                cue_onset_offset_seconds=cue_onset_offset_seconds,
                filter_l_freq_hz=filter_l_freq_hz,
                filter_h_freq_hz=filter_h_freq_hz,
                model_out_features=model_out_features,
                output_evaluation_dir=(
                    Path(output_evaluation_dir) / f"sub_{subj}" / f"seed_{seed}"
                    if output_evaluation_dir is not None else None
                ),
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                cache_path=cache_path,
                momentum=momentum,
                confidence_threshold=confidence_threshold,
                enable_consolidation=enable_consolidation,
                consolidation_interval=consolidation_interval,
                sleep_steps=sleep_steps,
                noise_std=noise_std,
                pull_rate=pull_rate,
                checkpoint_dir=Path(checkpoint_dir) / f"sub_{subj}" / f"seed_{seed}",
                dataset=dataset,
            )
            runs.append(res)

    if not runs:
        raise ValueError("subjects and seeds must not be empty")

    pre_scores = [r.pre_adaptation_acc for r in runs]
    post_scores = [r.post_adaptation_acc for r in runs]
    gain_scores = [r.adaptation_gain for r in runs]
    ret_scores = [r.source_accuracy_after_adaptation for r in runs]
    forget_scores = [-r.source_retention_delta for r in runs]
    sparse_scores = [r.spike_sparsity for r in runs]

    mean_pre = float(np.mean(pre_scores))
    mean_post = float(np.mean(post_scores))
    mean_gain = float(np.mean(gain_scores))
    mean_ret = float(np.mean(ret_scores))
    mean_forget = float(np.mean(forget_scores))
    mean_sparse = float(np.mean(sparse_scores))
    std_post = float(np.std(post_scores))

    summary = SweepSummary(
        runs=runs,
        mean_pre_acc=mean_pre,
        mean_post_acc=mean_post,
        mean_gain=mean_gain,
        mean_retention=mean_ret,
        mean_forgetting=mean_forget,
        mean_sparsity=mean_sparse,
        std_post_acc=std_post,
    )

    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(runs[0]).keys())
            writer.writeheader()
            for r in runs:
                writer.writerow(asdict(r))

    manifest_root = Path(checkpoint_dir)
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "selection_required": True,
        "runs": [
            {
                "subject": run.subject,
                "seed": run.seed,
                "checkpoint": str(
                    manifest_root / f"sub_{run.subject}" / f"seed_{run.seed}"
                    / f"phase_a_subject_{run.subject}.pt"
                ),
                "evaluation_artifact": str(
                    (
                        Path(output_evaluation_dir)
                        if output_evaluation_dir is not None
                        else manifest_root
                    )
                    / f"sub_{run.subject}" / f"seed_{run.seed}"
                    / f"subject_{run.subject}_seed_{run.seed}.json"
                ),
            }
            for run in runs
        ],
    }
    (manifest_root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    return summary


def run_consolidation_ablation(
    subjects: Iterable[int],
    seeds: Iterable[int],
    *,
    output_dir: str | Path = "results/ablations/consolidation",
    **sweep_kwargs: Any,
) -> dict[str, Any]:
    """Run paired consolidation-on/off evaluations with identical subjects/seeds.

    The two conditions use separate output directories, but otherwise share the
    supplied protocol and configuration. Reported standard deviations describe
    variability across the requested subject-seed runs; this is not a causal
    hardware or clinical claim.
    """
    subjects = list(subjects)
    seeds = list(seeds)
    if not subjects or not seeds:
        raise ValueError("subjects and seeds must not be empty")
    root = Path(output_dir)
    conditions: dict[str, SweepSummary] = {}
    for enabled in (False, True):
        condition = "enabled" if enabled else "disabled"
        kwargs = dict(sweep_kwargs)
        kwargs["checkpoint_dir"] = root / condition / "checkpoints"
        kwargs["output_csv"] = root / condition / "results.csv"
        kwargs["enable_consolidation"] = enabled
        conditions[condition] = run_cross_session_sweep(
            subjects=subjects, seeds=seeds, **kwargs
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "paired_subject_seed_consolidation_ablation",
        "subjects": subjects,
        "seeds": seeds,
        "conditions": {},
    }
    for condition, summary in conditions.items():
        runs = [asdict(run) for run in summary.runs]
        report["conditions"][condition] = {
            "consolidation_enabled": condition == "enabled",
            "settings": {
                "consolidation_interval": sweep_kwargs.get("consolidation_interval", 50),
                "sleep_steps": sweep_kwargs.get("sleep_steps", 5),
                "noise_std": sweep_kwargs.get("noise_std", 0.05),
                "pull_rate": sweep_kwargs.get("pull_rate", 0.5),
            },
            "runs": runs,
            "mean_target_static_baseline": statistics.mean(
                run["target_static_baseline_acc"] for run in runs
            ),
            "mean_target_prequential": statistics.mean(
                run["target_prequential_acc"] for run in runs
            ),
            "mean_target_retrospective": statistics.mean(
                run["target_retrospective_acc"] for run in runs
            ),
            "mean_source_retention": statistics.mean(
                run["source_accuracy_after_adaptation"] for run in runs
            ),
            "std_target_prequential": statistics.pstdev(
                run["target_prequential_acc"] for run in runs
            ),
            "std_target_static_baseline": statistics.pstdev(
                run["target_static_baseline_acc"] for run in runs
            ),
            "std_target_retrospective": statistics.pstdev(
                run["target_retrospective_acc"] for run in runs
            ),
            "std_source_retention": statistics.pstdev(
                run["source_accuracy_after_adaptation"] for run in runs
            ),
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
