"""Evaluation metrics and spike-sparsity tracking for NeuroStream.

This module provides standard performance and neuromorphic efficiency metrics:
- Accuracy: classification correctness
- Adaptation gain: post-adaptation accuracy minus pre-adaptation accuracy
- Retention: source (session-1) accuracy after continuous target adaptation
- Forgetting: drop in session-1 accuracy from initial baseline
- Spike sparsity: mean firing rate across all LIF layers measured via forward hooks,
  serving as empirical validation for neuromorphic energy efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import numpy as np
import torch
import torch.nn as nn

from neurostream.models.lif_neuron import LIFNeuron


def accuracy(
    y_true: torch.Tensor | np.ndarray | Sequence[int],
    y_pred: torch.Tensor | np.ndarray | Sequence[int],
) -> float:
    """Compute top-1 classification accuracy.

    Parameters
    ----------
    y_true : Tensor or ndarray, shape ``(N,)``
    y_pred : Tensor or ndarray, shape ``(N,)``

    Returns
    -------
    float in [0.0, 1.0]
    """
    t_true = torch.as_tensor(y_true).reshape(-1)
    t_pred = torch.as_tensor(y_pred).reshape(-1)
    if t_true.numel() != t_pred.numel():
        raise ValueError("y_true and y_pred must contain matching element counts")
    if t_true.numel() == 0:
        raise ValueError("inputs must not be empty")
    return float((t_true == t_pred).float().mean().item())


def adaptation_gain(pre_adaptation_acc: float, post_adaptation_acc: float) -> float:
    """Compute gain in target accuracy after adaptation: post - pre."""
    return float(post_adaptation_acc - pre_adaptation_acc)


def forgetting(initial_session1_acc: float, post_adaptation_session1_acc: float) -> float:
    """Compute forgetting on session 1: initial - post_adaptation."""
    return float(initial_session1_acc - post_adaptation_session1_acc)


def retention_rate(initial_session1_acc: float, post_adaptation_session1_acc: float) -> float:
    """Compute proportion of session-1 capability retained: post / initial."""
    if initial_session1_acc <= 0:
        return 0.0
    return float(post_adaptation_session1_acc / initial_session1_acc)


class SpikeSparsityTracker:
    """Hook-based accumulator measuring mean firing rate across all LIF layers.

    Attaches forward hooks to all instances of ``LIFNeuron`` in a given model,
    recording the number of spikes fired versus total potential neuron-timesteps.
    Does not require altering the forward method or signature of any model.

    Example
    -------
    >>> model = SNNFeatureExtractor()
    >>> with SpikeSparsityTracker(model) as tracker:
    ...     _ = model(spikes)
    >>> print("Mean firing rate:", tracker.mean_firing_rate())
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.layer_spikes: dict[str, int] = {}
        self.layer_opportunities: dict[str, int] = {}
        self.is_active = False
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, LIFNeuron):
                self.layer_spikes[name] = 0
                self.layer_opportunities[name] = 0
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
        self.is_active = True

    def _make_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            # LIFNeuron returns (spike, mem)
            if isinstance(output, tuple) and len(output) >= 1:
                spike = output[0]
            elif isinstance(output, torch.Tensor):
                spike = output
            else:
                return

            if isinstance(spike, torch.Tensor):
                with torch.no_grad():
                    n_spikes = int(spike.sum().item())
                    n_total = spike.numel()
                    self.layer_spikes[name] += n_spikes
                    self.layer_opportunities[name] += n_total

        return hook

    def reset(self) -> None:
        """Clear all accumulated spike counts."""
        for name in self.layer_spikes:
            self.layer_spikes[name] = 0
            self.layer_opportunities[name] = 0

    def layer_sparsity(self) -> dict[str, float]:
        """Return mean firing rate per LIF layer."""
        result: dict[str, float] = {}
        for name in self.layer_spikes:
            denom = self.layer_opportunities[name]
            result[name] = (self.layer_spikes[name] / denom) if denom > 0 else 0.0
        return result

    def mean_firing_rate(self) -> float:
        """Return aggregate mean firing rate across all hooked LIF layers."""
        total_spikes = sum(self.layer_spikes.values())
        total_opps = sum(self.layer_opportunities.values())
        if total_opps == 0:
            return 0.0
        return float(total_spikes / total_opps)

    def close(self) -> None:
        """Remove all forward hooks from the model."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.is_active = False

    def __enter__(self) -> SpikeSparsityTracker:
        self.reset()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def measure_model_sparsity(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    """Convenience helper to run a forward pass and measure spike sparsity.

    Parameters
    ----------
    model : nn.Module
    input_tensor : Tensor, e.g. ``(channels, timesteps, trials)``

    Returns
    -------
    mean_rate : float
        Overall mean firing rate across all LIF layers.
    layer_rates : dict[str, float]
        Per-layer mean firing rate.
    """
    model.eval()
    with SpikeSparsityTracker(model) as tracker:
        with torch.no_grad():
            _ = model(input_tensor)
        mean_rate = tracker.mean_firing_rate()
        layer_rates = tracker.layer_sparsity()
    return mean_rate, layer_rates


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation record matching Section 5.4.1."""

    subject: int | str = 1
    session: str = "cross-session"
    pre_adaptation_acc: float = 0.0
    post_adaptation_acc: float = 0.0
    adaptation_gain: float = 0.0
    initial_retention_acc: float = 0.0
    post_retention_acc: float = 0.0
    forgetting: float = 0.0
    retention_rate: float = 0.0
    spike_sparsity: float = 0.0
    layer_sparsities: dict[str, float] = field(default_factory=dict)
    consolidation_enabled: bool = False
    acceptance_rate: float = float("nan")

    @classmethod
    def from_adaptation(
        cls,
        subject: int | str,
        phase_a_test_acc: float,
        phase_b_result: Any,
        session_name: str = "session_2",
        spike_sparsity: float = 0.0,
        layer_sparsities: dict[str, float] | None = None,
        consolidation_enabled: bool = False,
    ) -> EvaluationMetrics:
        """Create structured metrics from Phase A and Phase B results."""
        pre_acc = getattr(phase_b_result, "before_accuracy", 0.0)
        post_acc = getattr(phase_b_result, "after_accuracy", 0.0)
        post_ret = getattr(phase_b_result, "retention_accuracy", float("nan"))
        acc_rate = getattr(phase_b_result, "acceptance_rate", float("nan"))

        gain = adaptation_gain(pre_acc, post_acc)
        forget_val = forgetting(phase_a_test_acc, post_ret) if not np.isnan(post_ret) else float("nan")
        ret_rate_val = retention_rate(phase_a_test_acc, post_ret) if not np.isnan(post_ret) else float("nan")

        return cls(
            subject=subject,
            session=session_name,
            pre_adaptation_acc=pre_acc,
            post_adaptation_acc=post_acc,
            adaptation_gain=gain,
            initial_retention_acc=phase_a_test_acc,
            post_retention_acc=post_ret,
            forgetting=forget_val,
            retention_rate=ret_rate_val,
            spike_sparsity=spike_sparsity,
            layer_sparsities=layer_sparsities or {},
            consolidation_enabled=consolidation_enabled,
            acceptance_rate=acc_rate,
        )


def format_metrics_table(
    records: EvaluationMetrics | Sequence[EvaluationMetrics],
    title: str = "Section 5.4.1 - NeuroStream Evaluation Metrics",
) -> str:
    """Format evaluation metrics into a clean printable table.

    Parameters
    ----------
    records : EvaluationMetrics or list of EvaluationMetrics
    title : str

    Returns
    -------
    table_str : str
    """
    if isinstance(records, EvaluationMetrics):
        items = [records]
    else:
        items = list(records)

    lines = []
    sep = "=" * 105
    subsep = "-" * 105

    lines.append(sep)
    lines.append(f"  {title}")
    lines.append(sep)

    header = (
        f"{'Subj':<6} | {'Pre-Acc':<9} | {'Post-Acc':<9} | {'Gain':<8} | "
        f"{'Init-Ret':<9} | {'Post-Ret':<9} | {'Forget':<8} | {'Sparsity':<9} | {'Consolid':<8}"
    )
    lines.append(header)
    lines.append(subsep)

    for r in items:
        pre_str = f"{r.pre_adaptation_acc * 100:.2f}%" if not np.isnan(r.pre_adaptation_acc) else "N/A"
        post_str = f"{r.post_adaptation_acc * 100:.2f}%" if not np.isnan(r.post_adaptation_acc) else "N/A"
        gain_str = f"{r.adaptation_gain * 100:+.2f}%" if not np.isnan(r.adaptation_gain) else "N/A"
        init_ret_str = f"{r.initial_retention_acc * 100:.2f}%" if not np.isnan(r.initial_retention_acc) else "N/A"
        post_ret_str = f"{r.post_retention_acc * 100:.2f}%" if not np.isnan(r.post_retention_acc) else "N/A"
        forget_str = f"{r.forgetting * 100:+.2f}%" if not np.isnan(r.forgetting) else "N/A"
        sparse_str = f"{r.spike_sparsity * 100:.2f}%" if not np.isnan(r.spike_sparsity) else "N/A"
        cons_str = "ON" if r.consolidation_enabled else "OFF"

        row = (
            f"{str(r.subject):<6} | {pre_str:<9} | {post_str:<9} | {gain_str:<8} | "
            f"{init_ret_str:<9} | {post_ret_str:<9} | {forget_str:<8} | {sparse_str:<9} | {cons_str:<8}"
        )
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)
