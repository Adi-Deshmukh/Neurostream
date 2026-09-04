"""Two-layer LIF-based SNN feature extractor.

Architecture::

    Input (C, T, N)  →  transpose to (N, T, C)
        ↓
    FC-256  →  LIF  →  binary spikes (N, 256)   [per timestep]
        ↓
    FC-128  →  LIF  →  binary spikes (N, 128)   [per timestep]
        ↓
    Mean over T timesteps  →  (N, 128) continuous feature vector

The internal spikes between layers are binary.  The output is rate-decoded
(mean firing rate per neuron), yielding a continuous feature vector for
downstream classification (prototype memory, etc.).

Weight initialisation is tuned for binary spike inputs: Xavier uniform with
a gain that produces post-synaptic currents on the order of the LIF threshold.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from neurostream.models.lif_neuron import LIFNeuron


def _init_snn_weights(module: nn.Module, threshold: float = 0.5) -> None:
    """Initialise linear layers for spiking networks.

    Standard PyTorch Kaiming init produces currents ~sqrt(1/fan_in) which
    is far below typical LIF thresholds, causing neurons to rarely fire.
    We use Xavier uniform scaled by *threshold* so the expected current
    magnitude matches the firing threshold.
    """
    if isinstance(module, nn.Linear):
        gain = threshold * math.sqrt(3.0)  # scale Xavier to threshold order
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SNNFeatureExtractor(nn.Module):
    """Two-layer spiking neural network feature extractor.

    Parameters
    ----------
    in_features : int
        Number of input channels (virtual channels from spike encoder).
        Default 66 = 22 EEG channels × 3 frequency bands.
    hidden : int
        Width of the first hidden layer.  Default 256.
    out_features : int
        Width of the second (output) layer.  Default 128.
    beta : float
        LIF membrane decay factor.  Default 0.9.
    threshold : float
        LIF firing threshold.  Default 0.5 (lowered from 1.0 to prevent
        feature collapse with binary spike inputs).
    """

    def __init__(
        self,
        in_features: int = 66,
        hidden: int = 256,
        out_features: int = 128,
        beta: float = 0.9,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        # Layer 1: FC-256 → LIF
        self.fc1 = nn.Linear(in_features, hidden)
        self.lif1 = LIFNeuron(beta=beta, threshold=threshold)

        # Layer 2: FC-128 → LIF
        self.fc2 = nn.Linear(hidden, out_features)
        self.lif2 = LIFNeuron(beta=beta, threshold=threshold)

        self._out_features = out_features

        # Initialise weights for spiking regime
        self.apply(lambda m: _init_snn_weights(m, threshold))

    @property
    def out_features(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._out_features

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Run the SNN over all timesteps and return rate-decoded features.

        Parameters
        ----------
        spikes : Tensor, shape ``(C, T, N)``
            Binary spike tensor from the spike encoder.
            C = virtual channels, T = timesteps, N = batch size.

        Returns
        -------
        features : Tensor, shape ``(N, out_features)``
            Mean firing-rate feature vector (continuous, not binary).
        """
        # Transpose to batch-first: (C, T, N) → (N, T, C)
        x = spikes.permute(2, 1, 0).float()
        batch_size, n_timesteps, _ = x.shape

        # Initialise membrane potentials
        mem1: torch.Tensor | None = None
        mem2: torch.Tensor | None = None

        # Accumulate output spikes across timesteps
        spike_accumulator = torch.zeros(
            batch_size, self._out_features, device=x.device, dtype=x.dtype
        )

        for t in range(n_timesteps):
            # Slice out this timestep: (N, C)
            x_t = x[:, t, :]

            # Layer 1: FC → LIF
            cur1 = self.fc1(x_t)                    # (N, hidden)
            spike1, mem1 = self.lif1(cur1, mem1)     # (N, hidden) binary

            # Layer 2: FC → LIF
            cur2 = self.fc2(spike1)                  # (N, out_features)
            spike2, mem2 = self.lif2(cur2, mem2)     # (N, out_features) binary

            spike_accumulator = spike_accumulator + spike2

        # Rate decoding: mean spike count over timesteps → continuous features
        features = spike_accumulator / n_timesteps
        return features

    def extra_repr(self) -> str:
        return (
            f"fc1={self.fc1.in_features}→{self.fc1.out_features}, "
            f"fc2={self.fc2.in_features}→{self.fc2.out_features}, "
            f"beta={self.lif1.beta}, threshold={self.lif1.threshold}"
        )
