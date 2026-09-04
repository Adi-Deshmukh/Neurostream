"""Sleep-phase consolidation mechanism for prototype memory stabilization.

During continuous online adaptation, class prototypes drift toward target-session
distributions, causing catastrophic forgetting of source-session representations.

Sleep-phase consolidation prevents drift by periodically performing a gradient-free
correction using synthetic features generated from an anchored prototype snapshot:

    tilde{P} = P_snapshot + N(0, sigma^2 * I)   (sigma = 0.05)
    L_sleep  = 1 - cos(P_current, P_snapshot)

This maintains the zero-raw-data-storage privacy guarantee by avoiding replay
of real past session EEG trials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SleepConsolidationStepResult:
    """Output diagnostics for a single consolidation phase."""

    triggered: bool
    step_count: int
    sleep_loss: float
    per_class_loss: list[float] = field(default_factory=list)
    drift_before: float = 0.0
    drift_after: float = 0.0


class SleepConsolidator:
    """Consolidation manager applying periodic anchor pulls to prototypes.

    Parameters
    ----------
    snapshot_prototypes : Tensor, shape ``(classes, features)``
        Fixed snapshot of prototypes saved at the end of Phase A (or session-1 calibration).
    consolidation_interval : int
        Number of online adaptation steps between consolidation phases (default 50).
    sleep_steps : int
        Number of synthetic correction iterations per consolidation phase (default 5).
    noise_std : float
        Standard deviation of Gaussian noise added to snapshot prototypes (default 0.05).
    pull_rate : float
        Pull factor lambda for gradient-free EMA correction toward snapshot (default 0.5).
    seed : int | None
        Random seed for reproducible synthetic noise generation.
    """

    def __init__(
        self,
        snapshot_prototypes: torch.Tensor,
        *,
        consolidation_interval: int = 50,
        sleep_steps: int = 5,
        noise_std: float = 0.05,
        pull_rate: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if snapshot_prototypes.ndim != 2 or snapshot_prototypes.shape[0] < 2:
            raise ValueError("snapshot_prototypes must have shape (classes, features)")
        if consolidation_interval < 1:
            raise ValueError("consolidation_interval must be at least 1")
        if sleep_steps < 1:
            raise ValueError("sleep_steps must be at least 1")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if not 0 < pull_rate <= 1:
            raise ValueError("pull_rate must be in (0, 1]")

        self.snapshot_prototypes = snapshot_prototypes.detach().clone().float()
        self.consolidation_interval = consolidation_interval
        self.sleep_steps = sleep_steps
        self.noise_std = noise_std
        self.pull_rate = pull_rate
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        self.consolidation_count = 0
        self.loss_history: list[float] = []

    def compute_sleep_loss(
        self,
        current_prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine anchor loss L_sleep = 1 - cos(P_current, P_snapshot).

        Returns
        -------
        mean_loss : Tensor (scalar)
            Average cosine distance across all classes.
        per_class_loss : Tensor, shape ``(classes,)``
            Cosine distance per class.
        """
        # Cosine similarity along feature dimension
        cos_sim = F.cosine_similarity(current_prototypes, self.snapshot_prototypes, dim=-1)
        per_class_loss = 1.0 - cos_sim
        mean_loss = per_class_loss.mean()
        return mean_loss, per_class_loss

    def generate_synthetic_features(
        self,
        batch_size_per_class: int = 1,
    ) -> torch.Tensor:
        """Generate synthetic feature samples centered around snapshot prototypes.

        Parameters
        ----------
        batch_size_per_class : int
            Number of synthetic noisy vectors generated per class.

        Returns
        -------
        synthetic : Tensor, shape ``(classes, batch_size_per_class, features)``
            if batch_size_per_class > 1, else ``(classes, features)``.
        """
        n_classes, n_features = self.snapshot_prototypes.shape
        noise = torch.randn(
            (n_classes, batch_size_per_class, n_features),
            generator=self.rng,
            dtype=self.snapshot_prototypes.dtype,
            device=self.snapshot_prototypes.device,
        ) * self.noise_std

        expanded_snapshot = self.snapshot_prototypes.unsqueeze(1)  # (classes, 1, features)
        synthetic = expanded_snapshot + noise
        if batch_size_per_class == 1:
            return synthetic.squeeze(1)
        return synthetic

    def consolidate(
        self,
        current_prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, SleepConsolidationStepResult]:
        """Perform one sleep consolidation phase (running ``sleep_steps`` iterations).

        Parameters
        ----------
        current_prototypes : Tensor, shape ``(classes, features)``
            Live prototypes to be consolidated.

        Returns
        -------
        consolidated_prototypes : Tensor, shape ``(classes, features)``
        result : SleepConsolidationStepResult
        """
        P = current_prototypes.detach().clone().float()

        # Initial drift
        init_loss, _ = self.compute_sleep_loss(P)
        drift_before = float(init_loss.item())

        latest_per_class: list[float] = []
        latest_mean_loss: float = drift_before

        for step in range(self.sleep_steps):
            # 1. Generate synthetic features: P_tilde = P_snapshot + N(0, sigma)
            synthetic = self.generate_synthetic_features(batch_size_per_class=1)

            # 2. Compute L_sleep = 1 - cos(P_current, P_snapshot)
            mean_loss, per_class_loss = self.compute_sleep_loss(P)
            latest_mean_loss = float(mean_loss.item())
            latest_per_class = per_class_loss.cpu().tolist()

            # 3. Gradient-free correction:
            # Direct EMA-style pull toward snapshot weighted by loss and synthetic anchor.
            # alpha_pull = clamp(pull_rate * per_class_loss, max=1.0)
            alpha_pull = (self.pull_rate * per_class_loss).unsqueeze(-1).clamp(min=0.0, max=1.0)

            # Pull toward synthetic anchor (which is centered at P_snapshot)
            P = (1.0 - alpha_pull) * P + alpha_pull * synthetic

        final_loss, _ = self.compute_sleep_loss(P)
        drift_after = float(final_loss.item())

        self.consolidation_count += 1
        self.loss_history.append(drift_after)

        logger.debug(
            "Consolidation phase %d: drift %.4f -> %.4f (loss=%.4f)",
            self.consolidation_count,
            drift_before,
            drift_after,
            latest_mean_loss,
        )

        return P, SleepConsolidationStepResult(
            triggered=True,
            step_count=self.consolidation_interval * self.consolidation_count,
            sleep_loss=latest_mean_loss,
            per_class_loss=latest_per_class,
            drift_before=drift_before,
            drift_after=drift_after,
        )

    def step(
        self,
        current_prototypes: torch.Tensor,
        step_count: int,
    ) -> tuple[torch.Tensor, SleepConsolidationStepResult]:
        """Check interval and execute consolidation if scheduled.

        Parameters
        ----------
        current_prototypes : Tensor, shape ``(classes, features)``
        step_count : int
            Current 1-indexed adaptation step counter.

        Returns
        -------
        prototypes : Tensor
        result : SleepConsolidationStepResult
        """
        if step_count > 0 and (step_count % self.consolidation_interval == 0):
            return self.consolidate(current_prototypes)

        mean_loss, per_class = self.compute_sleep_loss(current_prototypes)
        return current_prototypes, SleepConsolidationStepResult(
            triggered=False,
            step_count=step_count,
            sleep_loss=float(mean_loss.item()),
            per_class_loss=per_class.cpu().tolist(),
            drift_before=float(mean_loss.item()),
            drift_after=float(mean_loss.item()),
        )
