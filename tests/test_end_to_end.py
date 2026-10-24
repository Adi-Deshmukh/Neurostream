"""Deterministic end-to-end pipeline coverage."""

import numpy as np
import torch
import mne

from neurostream.data.preprocessing import preprocess_session
from neurostream.data.spike_encoder import apply_encoder, fit_encoder
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.training.phase_a_train import evaluate_phase_a, train_phase_a
from neurostream.training.phase_b_adapt import adapt_target_session, score_adaptation
from neurostream.evaluation.artifacts import load_evaluation_artifact, save_evaluation_artifact


def _make_raw() -> object:
    rng = np.random.default_rng(17)
    sfreq = 250.0
    info = mne.create_info(4, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(rng.standard_normal((4, 42_000)), info, verbose=False)
    onsets = np.arange(2.0, 162.0, 10.0)
    labels = np.tile(["769", "770", "771", "772"], 4)
    raw.set_annotations(
        mne.Annotations(onsets, [0.0] * len(onsets), labels)
    )
    return raw


def _run_pipeline(checkpoint_dir):
    X, y, metadata = preprocess_session(_make_raw(), tmin=2.0, tmax=6.0)
    train_X, eval_X = X[:8], X[8:]
    train_y, eval_y = y[:8], y[8:]
    calibration = fit_encoder(train_X, sfreq=250.0, n_timesteps=20)
    encoded = apply_encoder(train_X, calibration, sfreq=250.0)
    eval_encoded = apply_encoder(eval_X, calibration, sfreq=250.0)
    torch.manual_seed(42)
    model = SNNFeatureExtractor(in_features=encoded.spikes.shape[0], hidden=8, out_features=8)
    result = train_phase_a(
        encoded.spikes,
        train_y,
        epochs=1,
        batch_size=4,
        n_classes=4,
        model=model,
        checkpoint_dir=checkpoint_dir,
        encoder_calibration=calibration,
        label_names=["769", "770", "771", "772"],
    )
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    reloaded = SNNFeatureExtractor(
        in_features=encoded.spikes.shape[0], hidden=8, out_features=8
    )
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    prototypes = torch.load(result.prototype_path, map_location="cpu", weights_only=True)
    evaluation = evaluate_phase_a(
        reloaded, prototypes, eval_encoded.spikes, eval_y, checkpoint["label_values"]
    )
    adaptation = adapt_target_session(
        reloaded, prototypes, eval_encoded.spikes, confidence_threshold=0.0
    )
    scored = score_adaptation(adaptation, np.searchsorted(checkpoint["label_values"], eval_y))
    artifact_path = checkpoint_dir / "evaluation.json"
    save_evaluation_artifact(
        artifact_path,
        checkpoint_path=result.checkpoint_path,
        subject=1,
        train_session="train",
        evaluation_session="held_out",
        metrics={"prototype_accuracy": evaluation["prototype_accuracy"]},
        label_values=checkpoint["label_values"],
        label_names=checkpoint["label_names"],
        run_config={"protocol": "held_out"},
    )
    artifact = load_evaluation_artifact(
        artifact_path, checkpoint_path=result.checkpoint_path, subject=1
    )
    return X, encoded, checkpoint, evaluation, scored, artifact


def test_complete_pipeline_is_deterministic_and_contract_valid(tmp_path):
    first = _run_pipeline(tmp_path / "first")
    second = _run_pipeline(tmp_path / "second")

    X, encoded, checkpoint, evaluation, phase_b, artifact = first
    assert X.shape == (16, 4, 1000)
    assert encoded.spikes.shape == (12, 20, 8)
    assert 0.0 <= evaluation["prototype_accuracy"] <= 1.0
    assert 0.0 <= phase_b.before_accuracy <= 1.0
    assert 0.0 <= phase_b.after_accuracy <= 1.0
    assert phase_b.prequential_predictions.shape == (8,)
    assert checkpoint["encoder_calibration"]["sfreq"] == 250.0
    assert checkpoint["preprocessing_contract"]["encoder_n_timesteps"] == 20
    assert checkpoint["label_names"] == ["769", "770", "771", "772"]
    assert artifact["split_identity"] == "held_out_evaluation"
    assert artifact["metrics"]["prototype_accuracy"] == evaluation["prototype_accuracy"]

    np.testing.assert_array_equal(first[1].spikes, second[1].spikes)
    assert first[3] == second[3]
    torch.testing.assert_close(
        first[4].prequential_predictions, second[4].prequential_predictions
    )
