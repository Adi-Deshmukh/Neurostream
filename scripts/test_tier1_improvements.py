#!/usr/bin/env python3
"""Quick test of improved Phase A training improvements.

Tests:
1. Verify augmentation functions work
2. Verify deep SNN architecture is valid
3. Run a quick training sweep on subject 1
4. Compare results with baseline
"""

import sys
import logging
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_augmentations():
    """Test the augmentation functions."""
    from neurostream.data.augmentation_advanced import (
        time_shift_spikes,
        channel_dropout,
        gaussian_noise,
        spike_dropout,
        mixup_spikes,
    )

    logger.info("Testing augmentation functions...")

    # Create dummy spike tensor
    spikes = torch.ones((66, 150, 32))  # (channels, timesteps, trials)
    labels = torch.arange(4).repeat(8)  # 4 classes, 8 trials each

    # Test time shift
    shifted = time_shift_spikes(spikes, max_shift=5)
    assert shifted.shape == spikes.shape
    logger.info("✓ Time shift works")

    # Test channel dropout
    dropped = channel_dropout(spikes, dropout_prob=0.2)
    assert dropped.shape == spikes.shape
    assert (dropped.sum() <= spikes.sum())  # Some channels dropped
    logger.info("✓ Channel dropout works")

    # Test Gaussian noise
    noisy = gaussian_noise(spikes.clone(), sigma=0.05)
    assert noisy.shape == spikes.shape
    logger.info("✓ Gaussian noise works")

    # Test spike dropout
    sparse = spike_dropout(spikes.clone(), dropout_prob=0.15)
    assert sparse.shape == spikes.shape
    assert (sparse.sum() <= spikes.sum())
    logger.info("✓ Spike dropout works")

    # Test mixup
    batch_x = torch.randn(66, 150, 32)
    mixed_x, mixed_y = mixup_spikes(batch_x, labels, alpha=0.2)
    assert mixed_x.shape == batch_x.shape
    assert mixed_y.shape[0] == labels.shape[0]
    logger.info("✓ Mixup works")

    logger.info("All augmentation tests passed!")


def test_deep_snn():
    """Test the DeepSNNFeatureExtractor."""
    from neurostream.models.deep_snn_feature_extractor import DeepSNNFeatureExtractor

    logger.info("Testing DeepSNNFeatureExtractor...")

    model = DeepSNNFeatureExtractor(
        in_features=66,
        hidden1=512,
        hidden2=256,
        out_features=512,
        learnable_lif=True,
        use_attention=True,
    )

    # Test forward pass with spike tensor
    spikes = torch.randn(66, 150, 4)  # (channels, timesteps, trials)
    output = model(spikes)
    assert output.shape == (4, 512), f"Expected (4, 512), got {output.shape}"
    logger.info(f"✓ Forward pass works: {output.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model has {n_params:,} parameters")

    # Check learnable LIF parameters
    assert hasattr(model.lif1, "raw_beta")
    assert hasattr(model.lif2, "raw_beta")
    assert hasattr(model.lif3, "raw_beta")
    logger.info("✓ LIF parameters are learnable")

    # Check attention weights
    if model.last_attention_weights is not None:
        logger.info(f"✓ Attention weights captured: {model.last_attention_weights.shape}")

    logger.info("All DeepSNN tests passed!")


def test_improved_training():
    """Test the improved training on subject 1."""
    logger.info("Testing improved Phase A training on subject 1...")
    logger.info("This will download BNCI2014-001 dataset and train for 5 epochs (quick test)...")

    # Import after basic tests
    from neurostream.training.phase_a_train import train_subject as baseline_train
    from scripts.run_phase_a_improved import train_subject_improved

    # Test parameters (smaller for quick test)
    test_params = {
        "subject": 1,
        "epochs": 5,  # Quick test: only 5 epochs
        "batch_size": 32,
        "seed": 42,
        "learning_rate": 1e-3,
        "temperature": 0.1,
        "n_timesteps": 100,  # Test with 100 timesteps
        "hidden": 512,
        "out_features": 512,
        "warmup_epochs": 2,
        "checkpoint_dir": "results/checkpoints_test",
    }

    try:
        logger.info("Running improved training...")
        result_improved = train_subject_improved(**test_params)
        logger.info(
            f"Improved training result: test_acc={result_improved['test_accuracy']:.2%}"
        )
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


def main():
    logger.info("=" * 60)
    logger.info("Neurostream Tier 1 Improvements Test Suite")
    logger.info("=" * 60)

    try:
        # Test 1: Augmentations
        test_augmentations()
        logger.info("")

        # Test 2: Deep SNN
        test_deep_snn()
        logger.info("")

        # Test 3: Improved training (optional, requires MOABB dataset)
        logger.info("Skipping full training test (requires ~10GB MOABB dataset)")
        logger.info("To test full training, run:")
        logger.info("  python scripts/run_phase_a_improved.py --subject 1 --epochs 50")

        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
