"""Enhanced Deep Spiking Neural Network with 3-layer architecture and skip connections.

This is an improved version of SNNFeatureExtractor that adds:
- 3rd LIF layer for deeper processing
- More sophisticated residual connections
- Better hyperparameter defaults based on Tier 2 improvements
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurostream.models.lif_neuron import LIFNeuron
from neurostream.models.spatial_snn_layer import SpatialSNNLayer
from neurostream.models.temporal_attention import TemporalSpikeAttention


def _init_snn_weights(module: nn.Module, threshold: float = 0.5) -> None:
    """Initialize linear layers for spiking networks."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        gain = threshold * math.sqrt(3.0)
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DeepSNNFeatureExtractor(nn.Module):
    """Deep Spiking Neural Network with 3-layer architecture and residual connections.

    Tier 2 improvements:
    - 3-layer FC-LIF stack (input → 512 → 256 → 128 → output)
    - Sophisticated residual skip connections between layers
    - Learnable LIF parameters (beta decay, threshold) per layer
    - Temporal attention for intelligent spike pooling
    - Spatial processing for multi-channel EEG

    Parameters
    ----------
    in_features : int
        Number of input channels (default 66 for 22 EEG channels × 3 bands).
    hidden1 : int
        First hidden layer width (default 512).
    hidden2 : int
        Second hidden layer width (default 256).
    out_features : int
        Output feature dimensionality (default 512).
    beta : float
        Initial LIF membrane decay factor (default 0.9).
    threshold : float
        Initial LIF firing threshold (default 0.5).
    enable_spatial : bool
        Enable spatial SNN layer for mixing channels (default True).
    spatial_filters_per_band : int
        Number of learned spatial filters per band (default 8).
    use_attention : bool
        Use temporal attention for spike pooling (default True).
    learnable_lif : bool
        Make LIF parameters trainable (default True).
    skip_scale : float
        Scaling factor for skip connections (default 0.1).
    """

    def __init__(
        self,
        in_features: int = 66,
        hidden1: int = 512,
        hidden2: int = 256,
        out_features: int = 512,
        beta: float = 0.9,
        threshold: float = 0.5,
        enable_spatial: bool = True,
        spatial_filters_per_band: int = 8,
        use_attention: bool = True,
        learnable_lif: bool = True,
        skip_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self._out_features = out_features
        self.use_attention = use_attention
        self.skip_scale = skip_scale

        # Spatial SNN layer for spike inputs
        if enable_spatial and in_features % 3 == 0 and in_features >= 6:
            in_ch = in_features // 3
            self.spatial_layer = SpatialSNNLayer(
                in_channels=in_ch,
                n_bands=3,
                spatial_filters_per_band=spatial_filters_per_band,
            )
            fc1_in = self.spatial_layer.out_channels
        else:
            self.spatial_layer = None
            fc1_in = in_features

        # Learnable Front-End for Raw EEG (already supports direct EEG input)
        self.temporal_conv = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.temporal_bn = nn.BatchNorm2d(16, affine=True, track_running_stats=False)
        self.spatial_depthwise = nn.Conv2d(16, 32, (22, 1), groups=16, bias=False)
        self.spatial_bn = nn.BatchNorm2d(32, affine=True, track_running_stats=False)
        self.front_lif = LIFNeuron(beta=beta, threshold=threshold, learnable=learnable_lif)
        self.time_pool = nn.AvgPool2d((1, 4))
        self.raw_fc_in = nn.Linear(32, fc1_in)

        # Layer 1: FC → LIF (input → hidden1)
        self.fc1 = nn.Linear(fc1_in, hidden1)
        self.lif1 = LIFNeuron(beta=beta, threshold=threshold, learnable=learnable_lif)
        self.skip1_proj = nn.Linear(fc1_in, hidden1) if fc1_in != hidden1 else nn.Identity()

        # Layer 2: FC → LIF (hidden1 → hidden2) with skip connection
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = LIFNeuron(beta=beta, threshold=threshold, learnable=learnable_lif)
        self.skip2_proj = nn.Linear(hidden1, hidden2) if hidden1 != hidden2 else nn.Identity()

        # Layer 3: FC → LIF (hidden2 → out_features) with skip connection
        # This is the NEW deep layer!
        self.fc3 = nn.Linear(hidden2, out_features)
        self.lif3 = LIFNeuron(beta=beta, threshold=threshold, learnable=learnable_lif)
        self.skip3_proj = nn.Linear(hidden2, out_features) if hidden2 != out_features else nn.Identity()

        # Temporal attention for intelligent rate decoding
        if use_attention:
            self.temporal_attention = TemporalSpikeAttention(
                feature_dim=out_features, hidden_dim=128
            )
        else:
            self.temporal_attention = None

        self.last_attention_weights: torch.Tensor | None = None

        # Initialize weights for spiking regime
        self.apply(lambda m: _init_snn_weights(m, threshold))

    @property
    def out_features(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._out_features

    def get_spatial_weights(self) -> dict[str, torch.Tensor] | None:
        """Retrieve learned spatial filter weights if spatial layer is enabled."""
        if self.spatial_layer is not None:
            return self.spatial_layer.get_spatial_weights()
        return None

    def _forward_raw_eeg(self, x: torch.Tensor) -> torch.Tensor:
        """Process raw preprocessed EEG (N, channels, samples) into spike representations."""
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (N, 1, 22, T_samples)

        # Temporal convolution: learn bandpass filters
        feat = self.temporal_conv(x)
        feat = self.temporal_bn(feat)

        # Spatial depthwise convolution: learn electrode combinations
        feat = self.spatial_depthwise(feat)
        feat = self.spatial_bn(feat)

        # Temporal pooling: reduce sample count
        feat = self.time_pool(feat)  # (N, 32, 1, T_pooled)
        feat = feat.squeeze(2)  # (N, 32, T_pooled)

        # Convert to spikes via front LIF
        n_trials, n_channels, n_timesteps = feat.shape
        mem_front = None
        spikes_list = []
        for t in range(n_timesteps):
            spk, mem_front = self.front_lif(feat[:, :, t], mem_front)
            spikes_list.append(spk)

        spikes_front = torch.stack(spikes_list, dim=1)  # (N, T_pooled, 32)
        x_seq = self.raw_fc_in(spikes_front)  # (N, T_pooled, fc1_in)
        return x_seq

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the deep SNN over all timesteps and return attended features.

        Parameters
        ----------
        inputs : Tensor
            Either spike tensor ``(C, T, N)``, or raw EEG ``(N, C, T)`` / ``(C, T, N)``.

        Returns
        -------
        features : Tensor, shape ``(N, out_features)``
        """
        # Determine input type
        if inputs.ndim == 3 and inputs.shape[1] == 22 and inputs.shape[2] > 200:
            # (N, 22, 1000) raw EEG
            x_seq = self._forward_raw_eeg(inputs)
            batch_size = x_seq.shape[0]
            n_timesteps = x_seq.shape[1]
        elif inputs.ndim == 3 and inputs.shape[0] == 22 and inputs.shape[1] > 200:
            # (22, 1000, N) raw EEG
            x_raw = inputs.permute(2, 0, 1)
            x_seq = self._forward_raw_eeg(x_raw)
            batch_size = x_seq.shape[0]
            n_timesteps = x_seq.shape[1]
        else:
            # Spike tensor: (C, T, N) → (N, C, T)
            x = inputs.permute(2, 0, 1).float()
            batch_size = x.shape[0]

            if self.spatial_layer is not None:
                x = self.spatial_layer(x)

            # (N, T, C)
            x_seq = x.permute(0, 2, 1)
            n_timesteps = x_seq.shape[1]

        # Initialize membrane potentials for all 3 layers
        mem1: torch.Tensor | None = None
        mem2: torch.Tensor | None = None
        mem3: torch.Tensor | None = None

        spike_records = []

        # Forward pass over time
        for t in range(n_timesteps):
            x_t = x_seq[:, t, :]

            # Layer 1: FC → LIF
            cur1 = self.fc1(x_t)
            skip1 = self.skip1_proj(x_t)
            spike1, mem1 = self.lif1(cur1 + self.skip_scale * skip1, mem1)

            # Layer 2: FC → LIF with residual connection
            cur2 = self.fc2(spike1)
            skip2 = self.skip2_proj(spike1)
            spike2, mem2 = self.lif2(cur2 + self.skip_scale * skip2, mem2)

            # Layer 3: FC → LIF with residual connection (NEW!)
            cur3 = self.fc3(spike2)
            skip3 = self.skip3_proj(spike2)
            spike3, mem3 = self.lif3(cur3 + self.skip_scale * skip3, mem3)

            spike_records.append(spike3)

        # Stack over time: (N, T, out_features)
        spike_sequence = torch.stack(spike_records, dim=1)

        # Decoding: Temporal Attention or Mean Rate
        if self.temporal_attention is not None:
            features, weights = self.temporal_attention(spike_sequence, return_weights=True)
            self.last_attention_weights = weights.detach()
        else:
            features = spike_sequence.mean(dim=1)
            self.last_attention_weights = None

        return features

    def extra_repr(self) -> str:
        spatial_info = f"spatial_layer={self.spatial_layer is not None}, "
        attn_info = f"attention={self.use_attention}, "
        return (
            f"{spatial_info}{attn_info}"
            f"fc1={self.fc1.in_features}→{self.fc1.out_features}, "
            f"fc2={self.fc2.in_features}→{self.fc2.out_features}, "
            f"fc3={self.fc3.in_features}→{self.fc3.out_features}, "
            f"beta={self.lif1.beta}, threshold={self.lif1.threshold}"
        )
