"""Tests for 512-dim prototype memory head with momentum update alpha=0.3."""

import pytest
import torch
import torch.nn.functional as F

from neurostream.models.prototype_memory import PrototypeMemory


@pytest.fixture
def initial_prototypes() -> torch.Tensor:
    torch.manual_seed(42)
    # 4 classes, 512 features
    p = torch.randn(4, 512)
    return F.normalize(p, dim=-1)


def test_prototype_memory_initialization(initial_prototypes):
    mem = PrototypeMemory(initial_prototypes)
    assert mem.alpha == 0.3
    assert mem.momentum == 0.7
    assert mem.feature_dim == 512
    assert mem.n_classes == 4
    torch.testing.assert_close(mem.prototypes, initial_prototypes)


def test_prototype_ema_update():
    # 2 classes, 3 features
    p = torch.zeros(2, 3)
    mem = PrototypeMemory(p, alpha=0.3)
    feat = torch.tensor([10.0, 20.0, 30.0])
    mem.update(feat, pseudo_label=1)

    expected = 0.3 * feat
    torch.testing.assert_close(mem.prototypes[1], expected)
    torch.testing.assert_close(mem.prototypes[0], torch.zeros(3))


def test_cosine_predict_and_logits(initial_prototypes):
    mem = PrototypeMemory(initial_prototypes)
    # Feature identical to prototype 2
    feat = initial_prototypes[2:3].clone() * 2.5
    pred, conf = mem.predict(feat, temperature=0.1)

    assert pred.item() == 2
    assert conf.item() > 0.90


def test_cosine_margin(initial_prototypes):
    mem = PrototypeMemory(initial_prototypes)
    feat = initial_prototypes[0:1].clone()
    margin = mem.cosine_margin(feat)
    assert margin.shape == (1,)
    assert margin.item() > 0.0


def test_pairwise_class_distances(initial_prototypes):
    mem = PrototypeMemory(initial_prototypes)
    dist = mem.pairwise_class_distances()
    assert dist.shape == (4, 4)
    # Diagonal should be ~0 (distance to self)
    for c in range(4):
        assert abs(dist[c, c].item()) < 1e-5


def test_adapt_confidence_threshold_gating(initial_prototypes):
    mem = PrototypeMemory(initial_prototypes, alpha=0.3)
    p_before = mem.prototypes.clone()

    # Low-confidence ambiguous noise feature
    torch.manual_seed(99)
    noise_feat = torch.randn(1, 512)
    accepted = mem.adapt(noise_feat, confidence_threshold=0.999)

    assert accepted == 0
    torch.testing.assert_close(mem.prototypes, p_before)
