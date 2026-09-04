"""Reproducible multi-subject cross-session evaluation running Phase A + Phase B.

Runs the complete lifecycle across subjects and seeds:
1. Supervised Phase A SNN training on Session 1 -> freezes model weights.
2. Extracts initial class prototypes from Session 1.
3. Online Phase B adaptation on Session 2 using frozen momentum/alpha with sleep consolidation.
4. Measures target accuracy gain, session-1 retention, forgetting, and LIF spike sparsity.
5. Aggregates results into a reproducible table and CSV.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from neurostream.data.loader import load_bnci2014_001
from neurostream.data.spike_encoder import encode
from neurostream.evaluation.metrics import (
    EvaluationMetrics,
    format_metrics_table,
    measure_model_sparsity,
)
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.training.phase_a_train import train_phase_a
from neurostream.training.phase_b_adapt import adapt_target_session
from neurostream.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@dataclass
class SubjectRunResult:
    """Full evaluation metrics for one subject and one random seed."""

    subject: int
    seed: int
    phase_a_train_acc: float
    phase_a_val_acc: float
    phase_a_test_acc: float
    pre_adaptation_acc: float
    post_adaptation_acc: float
    adaptation_gain: float
    initial_retention_acc: float
    post_retention_acc: float
    forgetting: float
    spike_sparsity: float
    accepted_trials: int
    total_target_trials: int
    consolidation_phases: int
    consolidation_enabled: bool


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
                    pre_adaptation_acc=r.pre_adaptation_acc,
                    post_adaptation_acc=r.post_adaptation_acc,
                    adaptation_gain=r.adaptation_gain,
                    initial_retention_acc=r.initial_retention_acc,
                    post_retention_acc=r.post_retention_acc,
                    forgetting=r.forgetting,
                    retention_rate=r.post_retention_acc / r.initial_retention_acc if r.initial_retention_acc > 0 else 0.0,
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
    sessions = load_bnci2014_001(subjects=subject, dataset=dataset)
    train_name = next(name for name in sessions if "train" in name.lower())
    test_name = next(name for name in sessions if "test" in name.lower())

    train_session = sessions[train_name]
    test_session = sessions[test_name]

    # 2. Encode to spikes
    train_spikes = encode(train_session.X).spikes
    test_spikes = encode(test_session.X).spikes

    # 3. Phase A: Supervised training & freezing
    in_features = train_spikes.shape[0]
    model = SNNFeatureExtractor(in_features=in_features, hidden=hidden)

    save_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path("results/checkpoints/sweep") / f"sub_{subject}_seed_{seed}"

    phase_a = train_phase_a(
        train_spikes,
        train_session.y,
        test_spikes,
        test_session.y,
        subject=subject,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        augmentation_probability=augmentation_probability,
        model=model,
        checkpoint_dir=save_dir,
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
        target_labels,
        momentum=momentum,
        confidence_threshold=confidence_threshold,
        batch_size=batch_size,
        source_spikes=train_spikes,
        source_labels=source_labels,
        enable_consolidation=enable_consolidation,
        consolidation_interval=consolidation_interval,
        sleep_steps=sleep_steps,
        noise_std=noise_std,
        pull_rate=pull_rate,
    )

    gain = phase_b.after_accuracy - phase_b.before_accuracy
    forget = phase_a.test_accuracy - phase_b.retention_accuracy

    return SubjectRunResult(
        subject=subject,
        seed=seed,
        phase_a_train_acc=phase_a.train_accuracy,
        phase_a_val_acc=phase_a.validation_accuracy,
        phase_a_test_acc=phase_a.test_accuracy,
        pre_adaptation_acc=phase_b.before_accuracy,
        post_adaptation_acc=phase_b.after_accuracy,
        adaptation_gain=gain,
        initial_retention_acc=phase_a.test_accuracy,
        post_retention_acc=phase_b.retention_accuracy,
        forgetting=forget,
        spike_sparsity=mean_sparsity,
        accepted_trials=phase_b.accepted_trials,
        total_target_trials=test_spikes.shape[-1],
        consolidation_phases=phase_b.consolidation_phases,
        consolidation_enabled=enable_consolidation,
    )


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
    ret_scores = [r.post_retention_acc for r in runs]
    forget_scores = [r.forgetting for r in runs]
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

    return summary
