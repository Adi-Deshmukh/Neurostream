"""Custom Leaky Integrate-and-Fire neuron with surrogate gradient.

Hand-written membrane dynamics implementing:

    mem[t] = beta * mem[t-1] + I[t]
    spike[t] = Heaviside(mem[t] - threshold)
    mem[t] = mem[t] * (1 - spike[t])        # reset-by-subtraction

The non-differentiable Heaviside is replaced in the backward pass by a
fast-sigmoid surrogate gradient (slope=25 default):

    d_spike/d_mem ≈ 1 / (slope * |mem - threshold| + 1)²

This module is fully standalone — no snnTorch imports.  The surrogate
gradient concept follows Neftci et al. (2019) and Eshraghian et al. (2023).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Surrogate gradient: fast-sigmoid via torch.autograd.Function
# ---------------------------------------------------------------------------


class SurrogateSpike(torch.autograd.Function):
    """Heaviside step in forward, fast-sigmoid surrogate in backward.

    The forward pass returns a hard binary spike (0 or 1).  The backward pass
    substitutes the gradient of the fast-sigmoid function::

        σ'(x) = 1 / (slope * |x| + 1)²

    where *x = mem − threshold* (the "over-threshold" amount).

    Parameters passed through ``ctx``:
        slope (float): Steepness of the surrogate.  Higher values make the
            surrogate closer to the true Heaviside but can cause vanishing
            gradients.  25 is a well-tested default.
    """

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
        return grad, None  # no gradient for slope


# ---------------------------------------------------------------------------
# LIF Neuron Module
# ---------------------------------------------------------------------------


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron layer.

    Implements first-order membrane dynamics per timestep::

        mem  = beta * mem_prev + input_current
        spike = Heaviside(mem - threshold)   # with surrogate grad
        mem  = mem * (1 - spike)             # reset on spike

    Parameters
    ----------
    beta : float
        Membrane potential decay factor, ``exp(-dt / tau)``.  Values in
        (0, 1); higher → slower leak → longer memory.  Default 0.9.
    threshold : float
        Firing threshold voltage.  Default 1.0.
    slope : float
        Fast-sigmoid surrogate gradient slope.  Default 25.
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.slope = slope

    # -- public API ---------------------------------------------------------

    def forward(
        self,
        input_current: torch.Tensor,
        mem: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one timestep of LIF dynamics.

        Parameters
        ----------
        input_current : Tensor, shape ``(batch, features)`` or ``(features,)``
            Synaptic input current for this timestep.
        mem : Tensor or None
            Membrane potential carried from the previous timestep.  If
            ``None``, initialised to zeros matching *input_current*.

        Returns
        -------
        spike : Tensor
            Binary spike output, same shape as *input_current*.
        mem : Tensor
            Updated membrane potential after leak, integration, and reset.
        """
        if mem is None:
            mem = torch.zeros_like(input_current)

        # 1. Leak + integrate
        mem = self.beta * mem + input_current

        # 2. Spike (Heaviside in forward, surrogate in backward)
        spike = SurrogateSpike.apply(mem - self.threshold, self.slope)

        # 3. Reset-by-subtraction: membrane zeroed where spike occurred
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
        """Create a zero-initialised membrane state tensor.

        Parameters
        ----------
        batch_size : int
        features : int
            Number of neurons in this layer.
        device, dtype : optional
            Forwarded to ``torch.zeros``.

        Returns
        -------
        Tensor, shape ``(batch_size, features)``
        """
        return torch.zeros(batch_size, features, device=device, dtype=dtype)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}, slope={self.slope}"
