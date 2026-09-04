"""Tests for EEG data augmentation routines."""

import numpy as np
import pytest
import torch

from neurostream.data.augmentation import (
    augment_eeg_batch,
    channel_dropout_eeg,
    gaussian_noise_eeg,
    spike_dropout,
    time_shift_eeg,
)


def test_time_shift_preserves_shape():
    x = torch.randn(8, 22, 1000)
    out = time_shift_eeg(x, max_shift=20)
    assert out.shape == x.shape
    assert isinstance(out, torch.Tensor)


def test_channel_dropout_zeros_channels():
    x = torch.ones(4, 22, 100)
    out = channel_dropout_eeg(x, drop_prob=1.0, max_channels=2)
    assert out.shape == x.shape
    # At least one channel should be zeroed
    zero_channels = (out == 0.0).all(dim=-1).any()
    assert zero_channels


def test_gaussian_noise():
    x = torch.zeros(4, 22, 100)
    out = gaussian_noise_eeg(x, std=0.05)
    assert out.shape == x.shape
    assert out.abs().mean() > 0.01


def test_spike_dropout():
    spikes = (torch.rand(66, 25, 8) > 0.5).float()
    out = spike_dropout(spikes, drop_prob=0.2)
    assert out.shape == spikes.shape
    assert out.sum() <= spikes.sum()
