"""Simulation and trial-replay engine for the evaluation dashboard.

Provides streaming replay of Session 2 EEG trials against a frozen SNN model
and adapting prototype memory, precomputing a stable 2D projection basis (PCA or UMAP)
on Session 1 data so points and prototype positions move smoothly without jitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import joblib

import numpy as np
from sklearn.decomposition import PCA
import torch

from neurostream.data.loader import load_bnci2014_001
from neurostream.data.spike_encoder import EncoderCalibration, apply_encoder
from neurostream.evaluation.metrics import SpikeActivityMeasurement, measure_model_sparsity
from neurostream.models.prototype_memory import PrototypeMemory
from neurostream.models.snn_feature_extractor import SNNFeatureExtractor
from neurostream.training.sleep_consolidation import SleepConsolidator


@dataclass
class ReplayStepRecord:
    """Snapshot of system state at one trial step during streaming replay."""

    step: int
    feature_2d: np.ndarray  # shape (2,)
    true_label: int
    predicted_label: int
    confidence: float
    accepted: bool
    prototypes_2d: np.ndarray  # shape (n_classes, 2)
    cumulative_accuracy: float
    static_baseline_accuracy: float
    sleep_consolidation_triggered: bool = False
    drift_loss: float = 0.0


@dataclass
class ReplaySimulationData:
    """Precomputed artifacts and full replay history for a subject."""

    subject: int
    class_names: list[str]
    session1_features_2d: np.ndarray  # shape (N_s1, 2)
    session1_labels: np.ndarray  # shape (N_s1,)
    session2_features_2d: np.ndarray  # shape (N_s2, 2)
    session2_labels: np.ndarray  # shape (N_s2,)
    initial_prototypes_2d: np.ndarray  # shape (n_classes, 2)
    steps: list[ReplayStepRecord] = field(default_factory=list)
    measurement: SpikeActivityMeasurement | None = None
    run_id: str = ""
    checkpoint_ref: str = ""
    projection_method: str = "PCA"
    initial_target_accuracy: float = 0.0
    adapted_target_accuracy: float = 0.0
    adaptation_gain: float = 0.0
    session1_retention_accuracy: float = 0.0
    session1_initial_accuracy: float = 0.0
    prototype_history_2d: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 2), dtype=np.float32)
    )


class TrialReplayEngine:
    """Engine simulating online streaming adaptation trial-by-trial."""

    def __init__(
        self,
        model: SNNFeatureExtractor,
        initial_prototypes: torch.Tensor,
        session1_spikes: np.ndarray | torch.Tensor,
        session1_labels: np.ndarray | torch.Tensor,
        session2_spikes: np.ndarray | torch.Tensor,
        session2_labels: np.ndarray | torch.Tensor,
        class_names: Sequence[str] | None = None,
        projection_method: str = "PCA",
        subject: int = 1,
        run_id: str | None = None,
        checkpoint_ref: str = "in-memory",
    ) -> None:
        self.subject = subject
        self.checkpoint_ref = checkpoint_ref
        self.run_id = run_id or hashlib.sha256(
            np.asarray(session2_spikes).tobytes()
        ).hexdigest()[:16]
        self.model = model
        self.model.eval()
        self.initial_prototypes = initial_prototypes.detach().clone().float()

        self.s1_spikes = torch.as_tensor(session1_spikes, dtype=torch.float32)
        self.s1_labels = np.asarray(session1_labels).reshape(-1)

        self.s2_spikes = torch.as_tensor(session2_spikes, dtype=torch.float32)
        self.s2_labels = np.asarray(session2_labels).reshape(-1)
        if self.s1_spikes.ndim != 3 or self.s2_spikes.ndim != 3:
            raise ValueError("session spikes must have shape (channels, timesteps, trials)")
        if self.s1_spikes.shape[:2] != self.s2_spikes.shape[:2]:
            raise ValueError("session spikes must have matching channel and timestep dimensions")
        if self.s1_spikes.shape[-1] != self.s1_labels.size or self.s2_spikes.shape[-1] != self.s2_labels.size:
            raise ValueError("each session trial must have exactly one label")
        if not torch.isfinite(self.s1_spikes).all() or not torch.isfinite(self.s2_spikes).all():
            raise ValueError("session spikes must contain only finite values")

        n_classes = self.initial_prototypes.shape[0]
        if class_names is None or len(class_names) != n_classes:
            raise ValueError("class_names must contain one name per prototype class")
        self.class_names = list(class_names)
        self.projection_method = projection_method

        # 1. Extract Session 1 and Session 2 features from frozen SNN
        with torch.no_grad():
            self.s1_features = self.model(self.s1_spikes).cpu().numpy()
            self.s2_features = self.model(self.s2_spikes).cpu().numpy()

        # 2. Fit fixed 2D projection basis on Session 1 features once (stable reference frame)
        self.projector = None
        if projection_method.upper() == "UMAP":
            try:
                import umap
            except ImportError:
                # Fall back only when the optional dependency is unavailable.
                # Data/configuration errors during fitting must remain visible.
                self.projector = PCA(n_components=2, random_state=42)
                self.projector.fit(self.s1_features)
                self.projection_method = "PCA (UMAP unavailable; fallback)"
            else:
                self.projector = umap.UMAP(
                    n_components=2, random_state=42, min_dist=0.3, n_neighbors=15
                )
                self.projector.fit(self.s1_features)
        else:
            self.projector = PCA(n_components=2, random_state=42)
            self.projector.fit(self.s1_features)

        self.s1_features_2d = self.projector.transform(self.s1_features)
        self.s2_features_2d = self.projector.transform(self.s2_features)
        self.initial_prototypes_2d = self.projector.transform(self.initial_prototypes.cpu().numpy())

        # Measure neuromorphic spike sparsity via forward hooks
        aggregate_activity, layer_activity = measure_model_sparsity(self.model, self.s2_spikes)
        self.measurement = SpikeActivityMeasurement.from_activity(
            run_id=self.run_id,
            checkpoint_ref=self.checkpoint_ref,
            input_spikes=self.s2_spikes,
            layer_activity_fraction=layer_activity,
            aggregate_activity_fraction=aggregate_activity,
        )

        # Precompute static baseline accuracy on Session 2 (without adaptation)
        static_mem = PrototypeMemory(self.initial_prototypes.clone(), momentum=0.9999)
        s2_feat_all = torch.as_tensor(self.s2_features, dtype=torch.float32)
        static_preds, _ = static_mem.predict(s2_feat_all)
        self.static_preds = static_preds.cpu().numpy()
        self.initial_target_accuracy = float(np.mean(self.static_preds == self.s2_labels))

        # Baseline Session 1 accuracy
        s1_feat_all = torch.as_tensor(self.s1_features, dtype=torch.float32)
        s1_static_preds, _ = static_mem.predict(s1_feat_all)
        self.session1_initial_accuracy = float(np.mean(s1_static_preds.cpu().numpy() == self.s1_labels))

    def run_replay(
        self,
        momentum: float = 0.95,
        confidence_threshold: float = 0.8,
        temperature: float = 0.1,
        enable_consolidation: bool = True,
        consolidation_interval: int = 50,
        sleep_steps: int = 5,
        noise_std: float = 0.05,
        pull_rate: float = 0.5,
    ) -> ReplaySimulationData:
        """Simulate the full trial-by-trial streaming adaptation sequence."""
        memory = PrototypeMemory(self.initial_prototypes.clone(), momentum=momentum)

        consolidator = None
        if enable_consolidation:
            consolidator = SleepConsolidator(
                snapshot_prototypes=self.initial_prototypes.clone(),
                consolidation_interval=consolidation_interval,
                sleep_steps=sleep_steps,
                noise_std=noise_std,
                pull_rate=pull_rate,
                seed=int(hashlib.sha256(self.run_id.encode("utf-8")).hexdigest()[:8], 16),
            )

        steps: list[ReplayStepRecord] = []
        n_trials = self.s2_features.shape[0]
        correct_count = 0
        static_correct_count = 0
        accepted_count = 0

        for t in range(n_trials):
            feat_tensor = torch.as_tensor(self.s2_features[t : t + 1], dtype=torch.float32)
            true_lbl = int(self.s2_labels[t])

            # Classify using current prototype positions
            pred, conf = memory.predict(feat_tensor, temperature=temperature)
            pred_lbl = int(pred.item())
            conf_val = float(conf.item())

            if pred_lbl == true_lbl:
                correct_count += 1
            running_acc = correct_count / (t + 1)

            if int(self.static_preds[t]) == true_lbl:
                static_correct_count += 1
            static_running_acc = static_correct_count / (t + 1)

            # Adaptation event
            accepted = conf_val > confidence_threshold
            consolidation_triggered = False
            drift_loss = 0.0

            if accepted:
                memory.update(feat_tensor, pred)
                accepted_count += 1

                if consolidator is not None:
                    updated_proto, step_res = consolidator.step(
                        memory.prototypes, step_count=accepted_count
                    )
                    if step_res.triggered:
                        memory.prototypes = updated_proto
                        consolidation_triggered = True
                        drift_loss = step_res.drift_after

            # Project current prototypes into fixed 2D space
            proto_2d = self.projector.transform(memory.prototypes.cpu().numpy())

            steps.append(
                ReplayStepRecord(
                    step=t + 1,
                    feature_2d=self.s2_features_2d[t],
                    true_label=true_lbl,
                    predicted_label=pred_lbl,
                    confidence=conf_val,
                    accepted=accepted,
                    prototypes_2d=proto_2d,
                    cumulative_accuracy=running_acc,
                    static_baseline_accuracy=static_running_acc,
                    sleep_consolidation_triggered=consolidation_triggered,
                    drift_loss=drift_loss,
                )
            )

        # Final retention evaluation on Session 1
        s1_feat_all = torch.as_tensor(self.s1_features, dtype=torch.float32)
        final_s1_preds, _ = memory.predict(s1_feat_all)
        s1_retention = float(np.mean(final_s1_preds.cpu().numpy() == self.s1_labels))

        adapted_target_acc = steps[-1].cumulative_accuracy
        gain = adapted_target_acc - self.initial_target_accuracy

        return ReplaySimulationData(
            subject=self.subject,
            class_names=self.class_names,
            session1_features_2d=self.s1_features_2d,
            session1_labels=self.s1_labels,
            session2_features_2d=self.s2_features_2d,
            session2_labels=self.s2_labels,
            initial_prototypes_2d=self.initial_prototypes_2d,
            steps=steps,
            measurement=self.measurement,
            run_id=self.run_id,
            checkpoint_ref=self.checkpoint_ref,
            projection_method=self.projection_method,
            initial_target_accuracy=self.initial_target_accuracy,
            adapted_target_accuracy=adapted_target_acc,
            adaptation_gain=gain,
            session1_retention_accuracy=s1_retention,
            session1_initial_accuracy=self.session1_initial_accuracy,
            prototype_history_2d=np.stack(
                [record.prototypes_2d for record in steps], axis=0
            ),
        )

    def save_prepared_artifact(self, artifact_dir: str | Path) -> Path:
        """Persist replay-ready features only; raw EEG and encoded spikes are excluded."""
        target = Path(artifact_dir)
        target.mkdir(parents=True, exist_ok=True)
        arrays_path = target / "features.npz"
        projector_path = target / "projector.joblib"
        metadata_path = target / "metadata.json"

        np.savez_compressed(
            arrays_path,
            session1_features=self.s1_features,
            session2_features=self.s2_features,
            session1_labels=self.s1_labels,
            session2_labels=self.s2_labels,
            initial_prototypes=self.initial_prototypes.cpu().numpy(),
            session1_features_2d=self.s1_features_2d,
            session2_features_2d=self.s2_features_2d,
            initial_prototypes_2d=self.initial_prototypes_2d,
            static_preds=self.static_preds,
        )
        joblib.dump(self.projector, projector_path)
        metadata = {
            "schema_version": 1,
            "subject": self.subject,
            "class_names": self.class_names,
            "checkpoint_ref": self.checkpoint_ref,
            "run_id": self.run_id,
            "projection_method": self.projection_method,
            "initial_target_accuracy": self.initial_target_accuracy,
            "session1_initial_accuracy": self.session1_initial_accuracy,
            "measurement": self.measurement.as_dict(),
            "contains_raw_eeg": False,
            "contains_encoded_spikes": False,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return target

    @classmethod
    def from_prepared_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        checkpoint_ref: str,
        subject: int,
    ) -> "TrialReplayEngine":
        """Load feature-only replay artifacts after validating their provenance."""
        target = Path(artifact_dir)
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported prepared replay artifact schema")
        if int(metadata.get("subject", -1)) != int(subject):
            raise ValueError("prepared replay artifact subject mismatch")
        if metadata.get("checkpoint_ref") != checkpoint_ref:
            raise ValueError("prepared replay artifact checkpoint hash mismatch")
        if metadata.get("contains_raw_eeg") or metadata.get("contains_encoded_spikes"):
            raise ValueError("prepared artifact violates feature-only storage contract")

        with np.load(target / "features.npz", allow_pickle=False) as arrays:
            required = {
                "session1_features", "session2_features", "session1_labels",
                "session2_labels", "initial_prototypes", "session1_features_2d",
                "session2_features_2d", "initial_prototypes_2d", "static_preds",
            }
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError(f"prepared replay artifact missing arrays: {sorted(missing)}")
            instance = cls.__new__(cls)
            instance.s1_features = arrays["session1_features"].copy()
            instance.s2_features = arrays["session2_features"].copy()
            instance.s1_labels = arrays["session1_labels"].copy()
            instance.s2_labels = arrays["session2_labels"].copy()
            instance.initial_prototypes = torch.as_tensor(
                arrays["initial_prototypes"].copy(), dtype=torch.float32
            )
            instance.s1_features_2d = arrays["session1_features_2d"].copy()
            instance.s2_features_2d = arrays["session2_features_2d"].copy()
            instance.initial_prototypes_2d = arrays["initial_prototypes_2d"].copy()
            instance.static_preds = arrays["static_preds"].copy()

        if instance.s1_features.ndim != 2 or instance.s2_features.ndim != 2:
            raise ValueError("prepared feature matrices must be two-dimensional")
        if instance.s1_features.shape[1] != instance.s2_features.shape[1]:
            raise ValueError("prepared session feature dimensions do not match")
        if instance.s1_labels.size != instance.s1_features.shape[0]:
            raise ValueError("prepared Session 1 labels do not match feature rows")
        if instance.s2_labels.size != instance.s2_features.shape[0]:
            raise ValueError("prepared Session 2 labels do not match feature rows")
        if instance.static_preds.size != instance.s2_labels.size:
            raise ValueError("prepared static predictions do not match Session 2 labels")
        if instance.s1_features.shape[0] == 0 or instance.s2_features.shape[0] == 0:
            raise ValueError("prepared feature matrices must not be empty")
        if not all(np.isfinite(value).all() for value in (
            instance.s1_features, instance.s2_features,
            instance.initial_prototypes.numpy(), instance.s1_features_2d,
            instance.s2_features_2d, instance.initial_prototypes_2d,
        )):
            raise ValueError("prepared replay arrays must contain only finite values")
        n_classes = instance.initial_prototypes.shape[0]
        if len(metadata.get("class_names", [])) != n_classes:
            raise ValueError("prepared class names do not match prototype count")
        if instance.initial_prototypes.shape[1] != instance.s1_features.shape[1]:
            raise ValueError("prepared prototype and feature dimensions do not match")
        if instance.s1_features_2d.shape != (instance.s1_labels.size, 2):
            raise ValueError("prepared Session 1 projection shape is invalid")
        if instance.s2_features_2d.shape != (instance.s2_labels.size, 2):
            raise ValueError("prepared Session 2 projection shape is invalid")
        if instance.initial_prototypes_2d.shape != (n_classes, 2):
            raise ValueError("prepared prototype projection shape is invalid")
        for labels in (instance.s1_labels, instance.s2_labels, instance.static_preds):
            if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
                raise ValueError("prepared labels are outside the class range")

        instance.subject = int(subject)
        instance.checkpoint_ref = checkpoint_ref
        instance.run_id = str(metadata["run_id"])
        instance.class_names = list(metadata["class_names"])
        instance.projection_method = str(metadata["projection_method"])
        instance.projector = joblib.load(target / "projector.joblib")
        instance.model = None
        instance.s1_spikes = None
        instance.s2_spikes = None
        instance.initial_target_accuracy = float(metadata["initial_target_accuracy"])
        instance.session1_initial_accuracy = float(metadata["session1_initial_accuracy"])
        measurement_data = dict(metadata["measurement"])
        measurement_data.pop("units", None)
        instance.measurement = SpikeActivityMeasurement(**measurement_data)
        return instance

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path = "results/checkpoints/phase_a_subject_1.pt",
        prototypes_path: str | Path = "results/checkpoints/prototypes_subject_1.pt",
        subject: int = 1,
        projection_method: str = "PCA",
    ) -> TrialReplayEngine:
        """Instantiate engine by loading trained model and dataset."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        protos = torch.load(prototypes_path, map_location="cpu", weights_only=False)

        sessions = load_bnci2014_001(subjects=subject)
        train_name = next(name for name in sessions if "train" in name.lower())
        test_name = next(name for name in sessions if "test" in name.lower())

        required = ("encoder_calibration", "label_values", "label_names", "preprocessing_contract")
        missing = [key for key in required if key not in ckpt or ckpt[key] is None]
        if missing:
            raise ValueError(
                f"checkpoint is missing the preprocessing/label contract: {', '.join(missing)}"
            )
        payload = ckpt["encoder_calibration"]
        calibration = EncoderCalibration(
            alpha_thresholds=np.asarray(payload["alpha_thresholds"]),
            beta_thresholds=np.asarray(payload["beta_thresholds"]),
            sfreq=float(payload["sfreq"]),
            n_timesteps=int(payload["n_timesteps"]),
            rate_threshold_pct=float(payload["rate_threshold_pct"]),
            channel_count=int(payload["channel_count"]),
            filter_order=int(payload.get("filter_order", 4)),
        )
        if not np.isclose(sessions[train_name].sfreq, sessions[test_name].sfreq, atol=1e-3):
            raise ValueError("checkpoint replay requires matching train/test sampling frequencies")
        s1_spikes = apply_encoder(
            sessions[train_name].X, calibration, sfreq=sessions[train_name].sfreq
        ).spikes
        s2_spikes = apply_encoder(
            sessions[test_name].X, calibration, sfreq=sessions[test_name].sfreq
        ).spikes

        model = SNNFeatureExtractor(
            in_features=s1_spikes.shape[0],
            out_features=int(ckpt.get("feature_dim", 128)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        checkpoint_ref = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()

        label_values = torch.as_tensor(ckpt["label_values"])
        label_names = list(ckpt["label_names"])
        if label_values.numel() != protos.shape[0] or len(label_names) != protos.shape[0]:
            raise ValueError("checkpoint label mapping is incompatible with prototype count")
        s1_labels = torch.as_tensor(
            [torch.where(label_values == int(lbl))[0].item() for lbl in sessions[train_name].y],
            dtype=torch.long,
        )
        s2_labels = torch.as_tensor(
            [torch.where(label_values == int(lbl))[0].item() for lbl in sessions[test_name].y],
            dtype=torch.long,
        )

        return cls(
            model=model,
            initial_prototypes=protos,
            session1_spikes=s1_spikes,
            session1_labels=s1_labels,
            session2_spikes=s2_spikes,
            session2_labels=s2_labels,
            class_names=label_names,
            projection_method=projection_method,
            subject=subject,
            run_id=f"{checkpoint_ref[:16]}-subject-{subject}",
            checkpoint_ref=checkpoint_ref,
        )
