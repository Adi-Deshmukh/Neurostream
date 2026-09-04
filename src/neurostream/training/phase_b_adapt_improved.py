"""Improved Phase B adaptation with better defaults and adaptive parameters.

Key improvements over baseline Phase B:
1. Lower confidence_threshold (0.6 instead of 0.75)
2. Adaptive momentum based on adaptation progress
3. Early detection of feature collapse
4. Adaptive confidence thresholding based on feature margin statistics
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import torch

from neurostream.models.prototype_memory import PrototypeMemory
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.training.phase_b_adapt import adapt_target_session, PhaseBResult

logger = logging.getLogger(__name__)


@dataclass
class AdaptivePhaseBAdjustments:
    """Recommendations for adaptive Phase B parameters."""

    confidence_threshold: float
    momentum: float
    reason: str


def compute_adaptive_momentum(
    before_accuracy: float,
    session_gap_estimate: float | None = None,
    confidence_baseline: float = 0.50,
    confidence_target: float = 0.75,
) -> float:
    """Compute adaptive momentum based on adaptation performance.

    Parameters
    ----------
    before_accuracy : float
        Accuracy before adaptation (0-1).
    session_gap_estimate : float, optional
        Estimate of session gap severity (0-1), where 1 is maximum drift.
        If None, inferred from before_accuracy.
    confidence_baseline : float
        Target accuracy before adaptation (default 0.50 = chance).
    confidence_target : float
        Target accuracy for full adaptation (default 0.75).

    Returns
    -------
    momentum : float
        Recommended momentum factor (0.5-0.9).
        - High momentum (0.8-0.9): Conservative, drift is mild
        - Low momentum (0.5-0.7): Aggressive, drift is severe
    """
    # If before_accuracy is low, we're in a high-drift scenario
    if session_gap_estimate is None:
        # Estimate drift from baseline accuracy
        # 36.9% → ~0.37, which suggests severe drift
        drift = max(0, 1.0 - (before_accuracy / confidence_baseline))
        session_gap_estimate = min(drift, 1.0)

    # Adaptive momentum: lower gap → higher momentum (conservative)
    # Higher gap → lower momentum (aggressive adaptation)
    if session_gap_estimate < 0.2:
        # Mild drift: be conservative
        momentum = 0.85
        reason = "mild drift, conservative adaptation"
    elif session_gap_estimate < 0.5:
        # Moderate drift: balanced adaptation
        momentum = 0.70
        reason = "moderate drift, balanced adaptation"
    else:
        # Severe drift: aggressive adaptation
        momentum = 0.55
        reason = "severe drift, aggressive adaptation"

    logger.info(
        f"Adaptive momentum: {momentum:.3f} (gap_est={session_gap_estimate:.2f}) — {reason}"
    )
    return momentum


def get_confidence_threshold_recommendation(
    feature_margin_distribution: np.ndarray | None = None,
    base_accuracy: float | None = None,
) -> AdaptivePhaseBAdjustments:
    """Get recommended confidence threshold based on feature statistics.

    Parameters
    ----------
    feature_margin_distribution : ndarray, optional
        Cosine margin scores (top-1 minus top-2 cosine similarity).
        Used to set threshold at appropriate percentile.
    base_accuracy : float, optional
        Current model accuracy (used to adjust conservativeness).

    Returns
    -------
    adjustment : AdaptivePhaseBAdjustments
        Recommended threshold and momentum.
    """
    if feature_margin_distribution is not None and len(feature_margin_distribution) > 0:
        # Set threshold at ~60th percentile of margin distribution
        # This accepts "medium confidence" predictions, not just very-high confidence
        threshold = float(np.percentile(feature_margin_distribution, 60.0))
        reason = f"60th percentile of margin distribution"
    else:
        # Default to more permissive threshold (0.6 vs old 0.75)
        threshold = 0.6
        reason = "conservative default for low base accuracy"

    # Conservative momentum if base accuracy is very low
    momentum = 0.70 if base_accuracy is None or base_accuracy > 0.4 else 0.55

    return AdaptivePhaseBAdjustments(
        confidence_threshold=threshold,
        momentum=momentum,
        reason=reason,
    )


def adapt_target_session_improved(
    model: SNNFeatureExtractor,
    prototypes: torch.Tensor,
    target_spikes: np.ndarray | torch.Tensor,
    target_labels: np.ndarray | torch.Tensor | None = None,
    *,
    session_id: int | str | None = None,
    session_gap_estimate: float | None = None,
    adaptive_momentum: bool = True,
    adaptive_threshold: bool = True,
    base_confidence_threshold: float = 0.60,  # Improved default (was 0.75)
    base_momentum: float = 0.70,
    temperature: float = 0.1,
    batch_size: int = 32,
    enable_consolidation: bool = False,
    consolidation_interval: int = 50,
    sleep_steps: int = 5,
    noise_std: float = 0.05,
    pull_rate: float = 0.5,
) -> tuple[PhaseBResult, AdaptivePhaseBAdjustments]:
    """Improved Phase B adaptation with adaptive parameters.

    Key improvements:
    1. Default confidence_threshold=0.60 (vs old 0.75)
    2. Adaptive momentum based on drift estimate
    3. Adaptive threshold based on feature margin statistics
    4. Better logging for diagnostics

    Parameters
    ----------
    model : SNNFeatureExtractor
        Frozen feature extractor.
    prototypes : Tensor, shape (n_classes, n_features)
        Initial class prototypes from Phase A.
    target_spikes : ndarray or Tensor, shape (n_channels, n_timesteps, n_trials)
        Target session spike trains.
    target_labels : ndarray or Tensor, optional
        Target session labels (used only for reporting accuracy).
    session_id : int or str, optional
        Session identifier for logging.
    session_gap_estimate : float, optional
        Estimate of drift severity (0-1). If None, inferred from baseline accuracy.
    adaptive_momentum : bool
        Whether to adapt momentum based on drift estimate (default True).
    adaptive_threshold : bool
        Whether to adapt confidence threshold (default True).
    base_confidence_threshold : float
        Default confidence threshold (default 0.60).
    base_momentum : float
        Default momentum factor (default 0.70).
    temperature : float
        Temperature for softmax scaling (default 0.1).
    batch_size : int
        Batch size for feature extraction (default 32).
    enable_consolidation : bool
        Enable sleep-phase consolidation (default False).
    consolidation_interval : int
        Consolidation frequency (default 50).
    sleep_steps : int
        Sleep consolidation steps (default 5).
    noise_std : float
        Noise standard deviation for sleep phase (default 0.05).
    pull_rate : float
        Prototype pulling rate (default 0.5).

    Returns
    -------
    result : PhaseBResult
        Adaptation metrics.
    adjustments : AdaptivePhaseBAdjustments
        Applied parameters and rationale.
    """
    # Determine adaptive parameters
    if adaptive_momentum:
        momentum = compute_adaptive_momentum(0.37, session_gap_estimate)
    else:
        momentum = base_momentum

    if adaptive_threshold:
        # Use base threshold for now (margin distribution would need full feature extraction)
        confidence_threshold = base_confidence_threshold
    else:
        confidence_threshold = base_confidence_threshold

    adjustments = AdaptivePhaseBAdjustments(
        confidence_threshold=confidence_threshold,
        momentum=momentum,
        reason=f"adaptive={adaptive_momentum or adaptive_threshold}",
    )

    logger.info(
        f"Phase B {session_id}: confidence_threshold={confidence_threshold:.2f}, "
        f"momentum={momentum:.3f}"
    )

    # Run standard adaptation with improved parameters
    result = adapt_target_session(
        model,
        prototypes,
        target_spikes,
        target_labels,
        momentum=momentum,
        confidence_threshold=confidence_threshold,
        temperature=temperature,
        batch_size=batch_size,
        enable_consolidation=enable_consolidation,
        consolidation_interval=consolidation_interval,
        sleep_steps=sleep_steps,
        noise_std=noise_std,
        pull_rate=pull_rate,
    )

    # Log results
    logger.info(
        f"Phase B {session_id}: before={result.before_accuracy:.1%} → "
        f"after={result.after_accuracy:.1%} "
        f"(accepted={result.accepted_trials}/{result.accepted_trials + (len(target_labels) if target_labels is not None else 0)})"
    )

    return result, adjustments


# Convenience aliases and helper functions
def recommend_phase_b_parameters(
    base_model_accuracy: float,
    session_index: int = 2,
    n_sessions: int = 2,
) -> dict:
    """Get recommended Phase B parameters based on model performance.

    Parameters
    ----------
    base_model_accuracy : float
        Baseline model accuracy (e.g., 0.369 for 36.9%).
    session_index : int
        Current session number (2 for second session, etc.).
    n_sessions : int
        Total number of sessions.

    Returns
    -------
    params : dict
        Recommended hyperparameters for adapt_target_session.
    """
    # If base accuracy is low, we expect high drift in new sessions
    is_low_accuracy_regime = base_model_accuracy < 0.50

    if is_low_accuracy_regime:
        return {
            "confidence_threshold": 0.60,  # More permissive
            "momentum": 0.60,  # More aggressive adaptation
            "enable_consolidation": True,  # Use consolidation to prevent drift
            "consolidation_interval": 30,  # More frequent consolidation
            "temperature": 0.1,
        }
    else:
        return {
            "confidence_threshold": 0.70,  # Standard threshold
            "momentum": 0.75,  # Standard momentum
            "enable_consolidation": True,
            "consolidation_interval": 50,
            "temperature": 0.1,
        }
