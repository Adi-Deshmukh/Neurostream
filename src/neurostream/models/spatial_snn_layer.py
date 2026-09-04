"""Spatial SNN Layer for learning EEG electrode weighting and lateralization.

In motor-imagery decoding, the spatial distribution of scalp potentials conveys
critical discriminative information:
- Left-hand movement imagery induces contralateral Event-Related Desynchronization (ERD)
  over the right sensorimotor cortex (electrode C4).
- Right-hand imagery induces contralateral ERD over the left sensorimotor cortex (electrode C3).
- Foot imagery induces mid-sagittal ERD over the vertex (electrode Cz).
- Tongue imagery induces bilateral premotor/inferior rolandic activation.

This layer learns a bank of spatial filters (analogous to Common Spatial Patterns, CSP)
that linearly combines the 22 EEG electrodes across frequency bands before temporal
spiking dynamics.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# Standard 22 EEG channels from BCI Competition IV Dataset 2a (BNCI 2014-001)
BNCI2014_CHANNELS: list[str] = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "Oz",
]


class SpatialSNNLayer(nn.Module):
    """Learnable spatial filter bank combining 22 EEG channels into spatial components.

    Parameters
    ----------
    in_channels : int
        Number of physical EEG channels (default 22 for BCI IV 2a).
    n_bands : int
        Number of frequency sub-bands (default 3: Alpha, Beta, Gamma).
    spatial_filters_per_band : int
        Number of spatial filters learned per frequency band (default 8).
        Total spatial output features = ``n_bands * spatial_filters_per_band``.
    channel_names : Sequence[str] | None
        Names of the EEG channels for interpretability. Defaults to BNCI2014_CHANNELS.
    """

    def __init__(
        self,
        in_channels: int = 22,
        n_bands: int = 3,
        spatial_filters_per_band: int = 8,
        channel_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_bands = n_bands
        self.spatial_filters_per_band = spatial_filters_per_band
        self.channel_names = list(channel_names) if channel_names is not None else list(BNCI2014_CHANNELS)

        out_spatial = n_bands * spatial_filters_per_band
        self.out_channels = out_spatial

        # Spatial filter weights: maps (n_bands, in_channels) -> (n_bands, spatial_filters)
        # We implement this as grouped 1D convolutions across the virtual channels
        self.spatial_conv = nn.Conv1d(
            in_channels=in_channels * n_bands,
            out_channels=out_spatial,
            kernel_size=1,
            groups=n_bands,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm1d(out_spatial, affine=True, track_running_stats=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise spatial weights with CSP-inspired spatial patterns."""
        nn.init.kaiming_uniform_(self.spatial_conv.weight, a=math.sqrt(5))
        with torch.no_grad():
            # Scale to prevent initial saturation
            self.spatial_conv.weight.data *= 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial filtering to incoming virtual channels.

        Parameters
        ----------
        x : Tensor, shape ``(batch_size, n_virtual_channels, n_timesteps)``
            or ``(batch_size, n_timesteps, n_virtual_channels)``
            where ``n_virtual_channels = in_channels * n_bands``.

        Returns
        -------
        filtered : Tensor, shape ``(batch_size, out_channels, n_timesteps)``
        """
        # Ensure shape is (N, C_virtual, T)
        if x.shape[1] != self.in_channels * self.n_bands and x.shape[2] == self.in_channels * self.n_bands:
            x = x.transpose(1, 2)

        out = self.spatial_conv(x)
        out = self.batch_norm(out)
        return out

    def get_spatial_weights(self) -> dict[str, torch.Tensor]:
        """Return spatial filter weights mapped to channel names for visualization.

        Returns
        -------
        dict with band names mapping to tensor of shape ``(spatial_filters, in_channels)``.
        """
        w = self.spatial_conv.weight.detach().cpu()  # (out_spatial, in_channels, 1)
        w = w.squeeze(-1)  # (out_spatial, in_channels)
        n_sp = self.spatial_filters_per_band

        band_names = ["alpha", "beta", "gamma"][: self.n_bands]
        result = {}
        for b_idx, b_name in enumerate(band_names):
            start = b_idx * n_sp
            end = start + n_sp
            result[b_name] = w[start:end, :]

        return result
