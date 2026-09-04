"""Tests for SpatialSNNLayer (learnable spatial channel weighting and lateralization)."""

import pytest
import torch

from neurostream.models.spatial_snn_layer import BNCI2014_CHANNELS, SpatialSNNLayer


def test_spatial_snn_layer_initialization():
    layer = SpatialSNNLayer(in_channels=22, n_bands=3, spatial_filters_per_band=8)
    assert layer.in_channels == 22
    assert layer.n_bands == 3
    assert layer.out_channels == 24
    assert len(layer.channel_names) == 22
    assert "C3" in layer.channel_names
    assert "C4" in layer.channel_names
    assert "Cz" in layer.channel_names


def test_spatial_snn_layer_forward():
    layer = SpatialSNNLayer(in_channels=22, n_bands=3, spatial_filters_per_band=8)
    # Input shape: (batch_size, 66, n_timesteps)
    batch_size = 4
    n_timesteps = 25
    x = torch.randn(batch_size, 66, n_timesteps)
    out = layer(x)

    assert out.shape == (batch_size, 24, n_timesteps)
    assert torch.isfinite(out).all()


def test_spatial_snn_layer_gradient_flow():
    layer = SpatialSNNLayer(in_channels=22, n_bands=3, spatial_filters_per_band=8)
    x = torch.randn(2, 66, 10, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.spatial_conv.weight.grad is not None
    assert not torch.isnan(layer.spatial_conv.weight.grad).any()


def test_get_spatial_weights():
    layer = SpatialSNNLayer(in_channels=22, n_bands=3, spatial_filters_per_band=8)
    weights = layer.get_spatial_weights()

    assert "alpha" in weights
    assert "beta" in weights
    assert "gamma" in weights
    assert weights["alpha"].shape == (8, 22)
    assert weights["beta"].shape == (8, 22)
    assert weights["gamma"].shape == (8, 22)

