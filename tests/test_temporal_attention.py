"""Tests for TemporalSpikeAttention module."""

import pytest
import torch

from neurostream.models.temporal_attention import TemporalSpikeAttention


def test_temporal_attention_shape():
    attn = TemporalSpikeAttention(feature_dim=512, hidden_dim=128)
    # (batch_size=4, n_timesteps=25, feature_dim=512)
    seq = torch.randn(4, 25, 512)
    out, weights = attn(seq, return_weights=True)

    assert out.shape == (4, 512)
    assert weights.shape == (4, 25)
    # Weights should sum to 1 over timesteps
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)


def test_temporal_attention_gradients():
    attn = TemporalSpikeAttention(feature_dim=64, hidden_dim=32)
    seq = torch.randn(2, 10, 64, requires_grad=True)
    out = attn(seq)
    loss = out.sum()
    loss.backward()

    assert seq.grad is not None
    assert not torch.isnan(seq.grad).any()
