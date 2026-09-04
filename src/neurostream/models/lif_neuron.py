"""Custom Leaky Integrate-and-Fire neuron with surrogate gradient.

Hand-written membrane dynamics implementing:

    mem[t] = beta * mem[t-1] + I[t]
    spike[t] = Heaviside(mem[t] - threshold)
    mem[t] = mem[t] * (1 - spike[t])        # reset-by-subtraction

The non-differentiable Heaviside is replaced in the backward pass by a
fast-sigmoid surrogate gradient (slope=25 default):

    d_spike/d_mem ≈ 1 / (slope * |mem - threshold| + 1)²

Includes support for learnable decay factor beta and firing threshold.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Surrogate gradient: fast-sigmoid via torch.autograd.Function
# ---------------------------------------------------------------------------


class SurrogateSpike(torch.autograd.Function):
    """Heaviside step in forward, fast-sigmoid surrogate in backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        mem_minus_thresh: torch.Tensor,
        slope: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(mem_minus_thresh)
        ctx.slope = slope
        return (mem_minus_thresh > 0).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None]:
        (mem_minus_thresh,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (slope * torch.abs(mem_minus_thresh) + 1.0) ** 2
        return grad, None


# ---------------------------------------------------------------------------
# LIF Neuron Module
# ---------------------------------------------------------------------------


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron layer with optional learnable parameters.

    Parameters
    ----------
    beta : float
        Initial membrane potential decay factor, in (0, 1). Default 0.9.
    threshold : float
        Initial firing threshold voltage. Default 1.0.
    slope : float
        Fast-sigmoid surrogate gradient slope. Default 25.
    learnable : bool
        If True, makes beta and threshold trainable parameters via backpropagation.
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.learnable = learnable
        self.slope = slope

        if learnable:
            # Map beta into logit space so sigmoid maps to (0.5, 0.99)
            norm_b = (beta - 0.50) / 0.49
            norm_b = min(max(norm_b, 1e-4), 1.0 - 1e-4)
            init_logit = math.log(norm_b / (1.0 - norm_b))
            self.raw_beta = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
            self.raw_threshold = nn.Parameter(torch.tensor(math.log(max(math.exp(threshold) - 1.0, 1e-4)), dtype=torch.float32))
        else:
            self._beta = float(beta)
            self._threshold = float(threshold)

    @property
    def beta(self) -> float | torch.Tensor:
        """Effective membrane decay factor."""
        if self.learnable:
            return torch.sigmoid(self.raw_beta) * 0.49 + 0.50
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if not self.learnable:
            self._beta = float(value)

    @property
    def threshold(self) -> float | torch.Tensor:
        """Effective firing threshold voltage."""
        if self.learnable:
            return F.softplus(self.raw_threshold) + 0.05
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not self.learnable:
            self._threshold = float(value)

    def forward(
        self,
        input_current: torch.Tensor,
        mem: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one timestep of LIF dynamics."""
        if mem is None:
            mem = torch.zeros_like(input_current)

        eff_beta = self.beta
        eff_thresh = self.threshold

        # 1. Leak + integrate
        mem = eff_beta * mem + input_current

        # 2. Spike (Heaviside in forward, surrogate in backward)
        diff = mem - eff_thresh
        spike = SurrogateSpike.apply(diff, self.slope)

        # 3. Reset-by-subtraction
        mem = mem * (1.0 - spike)

        return spike, mem

    def init_mem(
        self,
        batch_size: int,
        features: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, features, device=device, dtype=dtype)

    def extra_repr(self) -> str:
        b_val = float(self.beta.item()) if isinstance(self.beta, torch.Tensor) else self.beta
        t_val = float(self.threshold.item()) if isinstance(self.threshold, torch.Tensor) else self.threshold
        return f"beta={b_val:.4f}, threshold={t_val:.4f}, slope={self.slope}, learnable={self.learnable}"
