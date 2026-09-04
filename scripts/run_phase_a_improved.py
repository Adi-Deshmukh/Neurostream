"""Run improved Phase A training with Tier 1 enhancements for MOABB subject."""

import argparse
import logging
from pathlib import Path

from neurostream.training.phase_a_train import train_subject as base_train_subject
from neurostream.training.phase_a_train import train_phase_a, _as_spike_tensor
from neurostream.data.spike_encoder import encode
from neurostream.data.loader import load_bnci2014_001
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.models.deep_snn_feature_extractor import DeepSNNFeatureExtractor
from neurostream.data.augmentation_advanced import compose_augmentations
import torch
import torch.nn as nn
import numpy as np


def train_subject_improved(
    subject: int = 1,
    *,
    checkpoint_dir: str | Path = "results/checkpoints",
    epochs: int = 100,
    batch_size: int = 32,
    seed: int = 42,
    learning_rate: float = 1e-3,
    temperature: float = 0.1,
    n_timesteps: int = 150,  # Increased from 25 to 150 (6× more temporal resolution)
    hidden: int = 512,  # Increased from 256
    out_features: int = 512,
    validation_fraction: float = 0.2,
    device: str = "cpu",
    # Advanced augmentation parameters
    time_shift_prob: float = 0.3,
    max_shift: int = 10,
    channel_dropout_prob: float = 0.15,
    gaussian_noise_sigma: float = 0.05,
    spike_dropout_prob: float = 0.15,
    mixup_alpha: float = 0.2,
    # Training recipe improvements
    label_smoothing: float = 0.1,
    warmup_epochs: int = 5,
    weight_decay: float = 1e-4,
) -> dict:
    """Train Phase A with Tier 1 improvements.

    Key enhancements:
    - 150 timesteps (6× more temporal resolution than baseline)
    - Advanced augmentation (mixup, time shift, channel dropout, noise)
    - Deeper architecture (256→512 hidden layer)
    - Better training recipe (warmup, label smoothing)
    - Higher learning rate with warmup schedule
    """
    # Load dataset using proper loader with fallback to synthetic data
    sessions = load_bnci2014_001(subjects=subject)
    
    train_name, test_name = "0train", "1test"

    if train_name not in sessions or test_name not in sessions:
        raise ValueError(f"could not identify train/test sessions: {list(sessions)}")

    train_session = sessions[train_name]
    test_session = sessions[test_name]

    # Encode with higher temporal resolution
    train_encoded = encode(train_session.X, n_timesteps=n_timesteps)
    test_encoded = encode(test_session.X, n_timesteps=n_timesteps)

    print(f"Train spikes shape: {train_encoded.spikes.shape}")
    print(f"Test spikes shape: {test_encoded.spikes.shape}")

    model = DeepSNNFeatureExtractor(
        in_features=train_encoded.spikes.shape[0],
        hidden1=512,
        hidden2=256,
        out_features=out_features,
        learnable_lif=True,
        use_attention=True,
    )

    return train_phase_a_improved(
        train_encoded.spikes,
        train_session.y,
        test_encoded.spikes,
        test_session.y,
        subject=subject,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        learning_rate=learning_rate,
        temperature=temperature,
        hidden=hidden,
        out_features=out_features,
        validation_fraction=validation_fraction,
        checkpoint_dir=checkpoint_dir,
        device=device,
        model=model,
        # Augmentation parameters
        time_shift_prob=time_shift_prob,
        max_shift=max_shift,
        channel_dropout_prob=channel_dropout_prob,
        gaussian_noise_sigma=gaussian_noise_sigma,
        spike_dropout_prob=spike_dropout_prob,
        mixup_alpha=mixup_alpha,
        # Training recipe
        label_smoothing=label_smoothing,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
    )


def train_phase_a_improved(
    train_spikes: np.ndarray | torch.Tensor,
    train_labels: np.ndarray | torch.Tensor,
    test_spikes: np.ndarray | torch.Tensor,
    test_labels: np.ndarray | torch.Tensor,
    *,
    subject: int = 1,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    temperature: float = 0.1,
    n_classes: int = 4,
    batch_size: int = 32,
    seed: int = 42,
    validation_fraction: float = 0.2,
    checkpoint_dir: str | Path = "results/checkpoints",
    device: str = "cpu",
    hidden: int = 512,
    out_features: int = 512,
    model: SNNFeatureExtractor | DeepSNNFeatureExtractor | None = None,
    # Augmentation
    time_shift_prob: float = 0.3,
    max_shift: int = 10,
    channel_dropout_prob: float = 0.15,
    gaussian_noise_sigma: float = 0.05,
    spike_dropout_prob: float = 0.15,
    mixup_alpha: float = 0.2,
    # Training recipe
    label_smoothing: float = 0.1,
    warmup_epochs: int = 5,
    weight_decay: float = 1e-4,
) -> dict:
    """Improved Phase A training with 150 timesteps and 3-layer SNN.
    
    Uses LEARNED PROTOTYPES like baseline for stable convergence.
    Key improvements:
    - 150 timesteps (6× better temporal resolution)
    - DeepSNNFeatureExtractor (3-layer with skip connections)
    - Gentle augmentations (no aggressive label smoothing)
    - Simple CosineAnnealingLR schedule (like baseline)
    """
    from neurostream.training.phase_a_train import (
        _as_spike_tensor,
        _encode_labels,
        _class_prototypes,
        prototype_logits,
        _accuracy,
        _stratified_split,
        _set_seed,
        _evaluate_prototypes,
    )
    import logging

    logger = logging.getLogger(__name__)

    # Setup
    _set_seed(seed)
    device = torch.device(device)
    
    # Convert data to tensors
    train_x = _as_spike_tensor(train_spikes).to(device)
    test_x = _as_spike_tensor(test_spikes).to(device)
    train_y, label_values = _encode_labels(train_labels)
    
    test_values = np.asarray(test_labels).reshape(-1)
    test_y = torch.as_tensor(
        np.searchsorted(label_values, test_values), dtype=torch.long, device=device
    )
    train_y = train_y.to(device)
    
    # Split into fit/validation
    fit_indices, validation_indices = _stratified_split(train_y, validation_fraction, seed)
    fit_x, fit_y = train_x[:, :, fit_indices], train_y[fit_indices]
    validation_x = train_x[:, :, validation_indices]
    validation_y = train_y[validation_indices]
    
    # Create model (use provided model or default to baseline)
    network = model or SNNFeatureExtractor(in_features=train_x.shape[0])
    network.to(device)
    
    logger.info(
        "Starting improved Phase A training: epochs=%d, batch_size=%d, n_trials=%d, "
        "hidden=%d, n_timesteps=%d, use_augmentation=%s",
        epochs, batch_size, fit_x.shape[-1], hidden, fit_x.shape[1], 
        time_shift_prob > 0,
    )
    
    # Initialize prototypes as learned parameters (KEY FIX: like baseline)
    network.eval()
    with torch.no_grad():
        initial_features = network(fit_x)
        initial_prototypes = _class_prototypes(initial_features, fit_y, n_classes)
    learned_prototypes = nn.Parameter(initial_prototypes.detach().clone())
    
    # Optimizer includes both network and prototypes
    optimizer = torch.optim.Adam(
        list(network.parameters()) + [learned_prototypes],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Simple CosineAnnealingLR like baseline (not custom warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_function = nn.CrossEntropyLoss()  # No label smoothing (remove regularization)
    
    # Training loop
    losses = []
    best_validation = -1.0
    best_state = None
    patience = 20
    without_improvement = 0
    n_trials = fit_x.shape[-1]
    
    for epoch in range(epochs):
        network.train()
        order = torch.randperm(n_trials, device=fit_x.device)
        epoch_loss = 0.0
        
        for indices in order.split(batch_size):
            batch_x = fit_x[:, :, indices].clone()
            batch_y = fit_y[indices].clone()
            
            # Apply gentle augmentations (optional, controlled by prob)
            if time_shift_prob > 0 or channel_dropout_prob > 0:
                batch_x, _ = compose_augmentations(
                    batch_x,
                    None,  # Don't augment labels
                    time_shift_prob=time_shift_prob,
                    max_shift=max_shift,
                    channel_dropout_prob=channel_dropout_prob,
                    gaussian_noise_sigma=gaussian_noise_sigma,
                    spike_dropout_prob=spike_dropout_prob,
                    mixup_alpha=0.0,  # No mixup
                    apply_mixup=False,
                )
            
            # Forward pass
            features = network(batch_x)
            
            # Loss with learned prototypes (KEY: prototypes are updated via backprop)
            loss = loss_function(
                prototype_logits(features, learned_prototypes, temperature),
                batch_y,
            )
            
            # Regularization: spatial filter orthogonality (like baseline)
            if getattr(network, "spatial_layer", None) is not None:
                w = network.spatial_layer.spatial_conv.weight.squeeze(-1)
                w_norm = torch.nn.functional.normalize(w, dim=-1)
                gram = w_norm @ w_norm.T
                eye = torch.eye(gram.shape[0], device=gram.device)
                loss = loss + 0.05 * torch.nn.functional.mse_loss(gram, eye)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(network.parameters()) + [learned_prototypes], 1.0
            )
            optimizer.step()
            
            epoch_loss += float(loss.detach().cpu()) * indices.numel()
        
        scheduler.step()
        losses.append(epoch_loss / n_trials)
        
        # Validation
        network.eval()
        with torch.no_grad():
            val_features = network(validation_x)
            val_acc = _evaluate_prototypes(
                network, validation_x, validation_y, learned_prototypes, temperature
            )
        
        if val_acc > best_validation:
            best_validation = val_acc
            best_state = {
                name: value.detach().clone()
                for name, value in network.state_dict().items()
            }
            best_state['learned_prototypes'] = learned_prototypes.detach().clone()
            without_improvement = 0
        else:
            without_improvement += 1
        
        logger.info(
            "Epoch %d/%d loss=%.5f val_acc=%.2f%% lr=%.2e",
            epoch + 1, epochs, losses[-1], val_acc * 100,
            scheduler.get_last_lr()[0],
        )
        
        if without_improvement >= patience:
            logger.info("Early stopping after %d epochs without improvement", patience)
            break
    
    # Load best state
    if best_state is not None:
        proto_state = best_state.pop('learned_prototypes')
        network.load_state_dict(best_state)
        learned_prototypes.data.copy_(proto_state)
    
    # Final evaluation
    network.eval()
    with torch.no_grad():
        train_features = network(fit_x)
        train_preds = prototype_logits(train_features, learned_prototypes, temperature).argmax(dim=1)
        train_accuracy = _accuracy(train_preds, fit_y)
        
        val_features = network(validation_x)
        val_preds = prototype_logits(val_features, learned_prototypes, temperature).argmax(dim=1)
        val_accuracy = _accuracy(val_preds, validation_y)
        
        test_features = network(test_x)
        test_preds = prototype_logits(test_features, learned_prototypes, temperature).argmax(dim=1)
        test_accuracy = _accuracy(test_preds, test_y)
    
    logger.info(
        "Phase A subject %d: train_accuracy=%.2f%% test_accuracy=%.2f%% validation_accuracy=%.2f%%",
        subject, train_accuracy * 100, test_accuracy * 100, val_accuracy * 100,
    )
    
    # Save checkpoint
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / f"phase_a_subject_{subject}_improved.pt"
    torch.save({
        'model_state_dict': network.state_dict(),
        'prototypes': learned_prototypes.detach().cpu(),
        'subject': subject,
        'seed': seed,
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'validation_accuracy': val_accuracy,
        'feature_dim': network.out_features if hasattr(network, 'out_features') else 512,
    }, checkpoint_path)
    
    prototype_path = output_dir / f"prototypes_subject_{subject}_improved.pt"
    torch.save(learned_prototypes.detach().cpu(), prototype_path)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'validation_accuracy': val_accuracy,
        'checkpoint_path': checkpoint_path,
        'prototype_path': prototype_path,
        'prototypes': learned_prototypes.detach().cpu(),
    }
    train_x = _as_spike_tensor(train_spikes).to(device)
    test_x = _as_spike_tensor(test_spikes).to(device)
    test_y, label_values = _encode_labels(test_labels)
    test_y = test_y.to(device)
    train_y, _ = _encode_labels(train_labels)
    train_y = train_y.to(device)

    # Split train into fit and validation
    fit_indices, validation_indices = _stratified_split(train_y, validation_fraction, seed)
    fit_x, fit_y = train_x[:, :, fit_indices], train_y[fit_indices]
    validation_x = train_x[:, :, validation_indices]
    validation_y = train_y[validation_indices]

    # Initialize model
    if model is None:
        model = DeepSNNFeatureExtractor(
            in_features=train_x.shape[0],
            hidden1=512,
            hidden2=256,
            out_features=out_features,
            learnable_lif=True,
            use_attention=True,
        )
    model.to(device)

    # Training with improved recipe
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    # Use smooth label targets with label smoothing
    loss_function = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Setup optimizer and scheduler with warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Ensure warmup_epochs < epochs to avoid division by zero
    actual_warmup_epochs = min(warmup_epochs, max(1, epochs - 2))
    
    # Warmup + cosine annealing schedule
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < actual_warmup_epochs:
            return (current_epoch + 1) / actual_warmup_epochs
        else:
            if epochs - actual_warmup_epochs > 0:
                progress = (current_epoch - actual_warmup_epochs) / (epochs - actual_warmup_epochs)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
            else:
                return 0.5

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    losses = []
    best_validation = -1.0
    best_state = None
    patience = 20
    without_improvement = 0
    n_trials = fit_x.shape[-1]

    logger.info(
        "Starting improved Phase A training: epochs=%d, batch_size=%d, n_trials=%d, "
        "hidden=%d, n_timesteps=%d, label_smoothing=%.2f, warmup=%d",
        epochs, batch_size, n_trials, hidden, fit_x.shape[1], label_smoothing, warmup_epochs,
    )

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_trials, device=fit_x.device)
        epoch_loss = 0.0

        for indices in order.split(batch_size):
            batch_x = fit_x[:, :, indices].clone()
            batch_y = fit_y[indices].clone()

            # Apply advanced augmentations (without mixup for now - use hard labels only)
            batch_x, batch_y_aug = compose_augmentations(
                batch_x,
                batch_y,
                time_shift_prob=time_shift_prob,
                max_shift=max_shift,
                channel_dropout_prob=channel_dropout_prob,
                gaussian_noise_sigma=gaussian_noise_sigma,
                spike_dropout_prob=spike_dropout_prob,
                mixup_alpha=mixup_alpha,
                apply_mixup=False,  # Disable mixup for now - use hard labels only
            )

            # Forward pass
            features = model(batch_x)
            
            # Compute prototypes from full training set each epoch
            model.eval()
            with torch.no_grad():
                fit_features = model(fit_x)
                prototypes = _class_prototypes(fit_features, fit_y, n_classes)
            model.train()

            # Loss computation with hard labels
            loss = loss_function(
                prototype_logits(features, prototypes, temperature),
                batch_y_aug,
            )

            # Regularization: spatial filter orthogonality
            if getattr(model, "spatial_layer", None) is not None:
                w = model.spatial_layer.spatial_conv.weight.squeeze(-1)
                w_norm = torch.nn.functional.normalize(w, dim=-1)
                gram = w_norm @ w_norm.T
                eye = torch.eye(gram.shape[0], device=gram.device)
                loss = loss + 0.05 * torch.nn.functional.mse_loss(gram, eye)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu()) * indices.numel()

        scheduler.step()
        losses.append(epoch_loss / n_trials)

        # Validation
        if validation_x is not None and validation_y is not None:
            model.eval()
            with torch.no_grad():
                val_features = model(validation_x)
                val_preds = prototype_logits(val_features, prototypes, temperature).argmax(dim=1)
                val_acc = float((val_preds == validation_y).float().mean().cpu())

            if val_acc > best_validation:
                best_validation = val_acc
                best_state = {
                    name: value.detach().clone() for name, value in model.state_dict().items()
                }
                without_improvement = 0
            else:
                without_improvement += 1

            if (epoch + 1) % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "Epoch %d/%d loss=%.5f val_acc=%.2f%% lr=%.2e",
                    epoch + 1, epochs, losses[-1], val_acc * 100, lr,
                )

            if without_improvement >= patience:
                logger.info("Early stopping after %d epochs without improvement", patience)
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_features = model(train_x)
        train_prototypes = _class_prototypes(train_features, train_y, n_classes)
        train_preds = prototype_logits(train_features, train_prototypes, temperature).argmax(dim=1)
        train_accuracy = float((train_preds == train_y).float().mean().cpu())

        test_features = model(test_x)
        test_preds = prototype_logits(test_features, train_prototypes, temperature).argmax(dim=1)
        test_accuracy = float((test_preds == test_y).float().mean().cpu())

    logger.info(
        "Phase A subject %d: train_accuracy=%.2f%% test_accuracy=%.2f%% validation_accuracy=%.2f%%",
        subject, train_accuracy * 100, test_accuracy * 100, best_validation * 100,
    )

    # Save checkpoint
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"phase_a_subject_{subject}_improved.pt"
    prototype_path = output_dir / f"prototypes_subject_{subject}_improved.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "subject": subject,
            "seed": seed,
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
            "validation_accuracy": best_validation,
            "n_timesteps": fit_x.shape[1],
            "label_values": label_values.tolist(),
            "feature_dim": model.out_features,
        },
        checkpoint_path,
    )

    torch.save(train_prototypes.cpu(), prototype_path)

    return {
        "model": model,
        "prototypes": train_prototypes.cpu(),
        "test_accuracy": test_accuracy,
        "train_accuracy": train_accuracy,
        "validation_accuracy": best_validation,
        "checkpoint_path": checkpoint_path,
        "prototype_path": prototype_path,
        "label_values": label_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n-timesteps", type=int, default=150)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = train_subject_improved(
        subject=args.subject,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        n_timesteps=args.n_timesteps,
        hidden=args.hidden,
        out_features=args.out_features,
        checkpoint_dir=args.checkpoint_dir,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
    )

    print(
        f"Subject {args.subject}: "
        f"train={result['train_accuracy']:.1%}, "
        f"val={result['validation_accuracy']:.1%}, "
        f"test={result['test_accuracy']:.1%}"
    )
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Prototypes: {result['prototype_path']}")


if __name__ == "__main__":
    main()
