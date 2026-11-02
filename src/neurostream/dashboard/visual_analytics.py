"""Unified visual analytics workspace for validated NeuroStream artifacts.

Every panel is artifact-backed. Missing measurements are reported rather than
reconstructed from unrelated metrics or silently filled with synthetic values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from neurostream.models.prototype_memory import PrototypeMemory


COLORS = ("#3976A8", "#269B83", "#D39A45", "#8B79B8")


def _chart(fig, render_plotly):
    render_plotly(fig)


def render_visual_analytics(
    *,
    root: Path,
    valid_runs: list[dict],
    selected_run: dict | None,
    render_plotly,
    confusion_figure,
    discover_ablation_reports,
    paired_ablation_rows,
) -> None:
    """Render the eight requested visualisations in one navigable workspace."""
    st.markdown(
        '<div class="ns-eyebrow">NEUROSTREAM / VISUAL ANALYTICS</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ns-title">Visual analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ns-subtitle">Eight research views · validated experiment artifacts · '
        'offline analysis only</div>',
        unsafe_allow_html=True,
    )

    run = selected_run or (valid_runs[0] if valid_runs else None)
    engine = st.session_state.get("ns_engine")
    replay = st.session_state.get("ns_replay")
    if run:
        st.caption(
            f"Selected experiment: S{run['subject']:02d} · seed {run['seed']} · "
            f"{run['artifact'].get('run_id', 'unknown run')}"
        )
    else:
        st.warning("No validated evaluation run is available. Registry-backed panels will remain empty.")

    # 1. EEG signal viewer — reuses the explicitly loaded session from Signal Analysis.
    st.markdown("### 1 · EEG signal viewer")
    sessions = st.session_state.get("ns_signal_sessions")
    if sessions:
        names = list(sessions)
        session_name = st.selectbox("EEG session", names, key="va_eeg_session")
        data = sessions[session_name]
        trial = st.slider("Trial", 1, int(data.X.shape[0]), 1, key="va_eeg_trial") - 1
        channels = list(data.channel_names) or [f"Channel {i + 1}" for i in range(data.X.shape[1])]
        channel = st.selectbox("Channel", range(data.X.shape[1]), format_func=lambda i: channels[i], key="va_eeg_channel")
        time = np.arange(data.X.shape[-1], dtype=float) / float(data.sfreq)
        signal = np.asarray(data.X[trial, channel], dtype=float)
        fig = go.Figure(go.Scatter(x=time, y=signal, mode="lines", name=channels[channel], line=dict(width=1)))
        fig.update_layout(title=f"{session_name} · trial {trial + 1} · {channels[channel]}",
                          xaxis_title="Time from epoch start (s)", yaxis_title="Preprocessed amplitude", height=330)
        _chart(fig, render_plotly)
        st.caption("This is the preprocessed epoch held in the Signal Analysis session. It is not raw continuous EEG.")
    else:
        st.info("No EEG session is loaded in this browser session. Open Signal Analysis and click 'Load EEG sessions'; then return here.")

    # 2. EEG frequency-band heatmap.
    st.markdown("### 2 · EEG frequency-band heatmap")
    if sessions:
        data = sessions[session_name]
        from scipy.integrate import trapezoid
        from scipy.signal import welch
        freqs, psd = welch(
            np.asarray(data.X[trial], dtype=float),
            fs=float(data.sfreq), axis=-1,
            nperseg=min(256, data.X.shape[-1]),
        )
        nyquist = float(data.sfreq) / 2.0
        bands = {"Theta (4–8 Hz)": (4.0, 8.0), "Alpha (8–13 Hz)": (8.0, 13.0),
                 "Beta (13–30 Hz)": (13.0, 30.0), "Gamma (30+ Hz)": (30.0, min(100.0, nyquist))}
        values = []
        labels = []
        for label, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                values.append(trapezoid(psd[:, mask], freqs[mask], axis=1))
            else:
                values.append(np.full(data.X.shape[1], np.nan))
            labels.append(label)
        z = np.asarray(values, dtype=float).T
        heat = go.Figure(go.Heatmap(z=z, x=labels, y=channels, colorscale="Viridis",
                                    colorbar_title="PSD integral",
                                    hovertemplate="Channel %{y}<br>Band %{x}<br>Power %{z:.4g}<extra></extra>"))
        heat.update_layout(title="Channel × frequency-band power · selected trial",
                           xaxis_title="Frequency band", yaxis_title="EEG channel", height=470)
        _chart(heat, render_plotly)
        st.caption("Band power is computed from the selected preprocessed epoch using Welch PSD. Empty bands above Nyquist remain unavailable.")
    else:
        st.info("Load EEG sessions in Signal Analysis to populate this heatmap.")

    # 3. Confusion matrix.
    st.markdown("### 3 · Confusion matrix")
    eligible = [item for item in valid_runs if item.get("run_config", {}).get("prequential_confusion_matrix") is not None]
    if eligible:
        cm_run = st.selectbox("Evaluation run", eligible,
                              format_func=lambda item: f"S{item['subject']:02d} · seed {item['seed']}",
                              key="va_cm_run")
        matrix = cm_run["run_config"].get("prequential_confusion_matrix")
        labels = cm_run.get("label_names", [])
        fig = confusion_figure(matrix, labels, "Stored prequential target confusion matrix")
        if fig is not None:
            _chart(fig, render_plotly)
        else:
            st.warning("The stored confusion matrix dimensions do not match the label names.")
    else:
        st.info("No validated evaluation artifact currently contains a prequential confusion matrix. This panel will not infer one from aggregate accuracy.")

    # 4. Subject × seed accuracy heatmap.
    st.markdown("### 4 · Subject × seed accuracy heatmap")
    if valid_runs:
        metric_name = st.selectbox(
            "Accuracy metric",
            ["target_prequential_acc", "target_static_baseline_acc", "target_retrospective_acc", "source_accuracy_after_adaptation"],
            format_func=lambda value: value.replace("_", " ").title(),
            key="va_accuracy_metric",
        )
        records = []
        for item in valid_runs:
            metrics = item.get("metrics", {})
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)) and np.isfinite(value):
                records.append({"Subject": item["subject"], "Seed": item["seed"], "Value": float(value)})
        if records:
            frame = pd.DataFrame(records)
            pivot = frame.pivot_table(index="Subject", columns="Seed", values="Value", aggfunc="mean")
            heat = go.Figure(go.Heatmap(z=pivot.to_numpy(dtype=float),
                                        x=[str(value) for value in pivot.columns],
                                        y=[f"S{value:02d}" for value in pivot.index],
                                        colorscale="Blues", zmin=0, zmax=1,
                                        colorbar_title="Accuracy",
                                        hovertemplate="%{y}<br>Seed %{x}<br>Accuracy %{z:.1%}<extra></extra>"))
            heat.update_layout(title=metric_name.replace("_", " ").title(), xaxis_title="Seed",
                               yaxis_title="Subject", height=max(350, 70 * len(pivot.index)))
            _chart(heat, render_plotly)
        else:
            st.info("No numeric values are available for this metric.")
    else:
        st.info("Validated runs are required for this heatmap.")

    # 5. Prototype movement and drift.
    st.markdown("### 5 · Prototype movement and drift")
    if engine is not None:
        if replay is not None and replay.steps:
            positions = np.asarray(replay.prototype_history_2d, dtype=float)
            fig = go.Figure()
            if positions.ndim == 3 and positions.shape[0] == len(replay.steps):
                for index, name in enumerate(replay.class_names):
                    fig.add_trace(go.Scatter(
                        x=positions[:, index, 0], y=positions[:, index, 1],
                        mode="lines", name=f"{name} trajectory",
                    ))
            else:
                st.warning("Replay prototype history is malformed; showing the prepared starting state instead.")
            initial = np.asarray(engine.initial_prototypes_2d, dtype=float)
            for index, name in enumerate(engine.class_names):
                fig.add_trace(go.Scatter(
                    x=[initial[index, 0]], y=[initial[index, 1]],
                    mode="markers", marker=dict(symbol="x", size=13),
                    name=f"{name} initial",
                ))
            fig.update_layout(
                title="Prototype trajectories in the fixed Session 1 projection",
                xaxis_title="Projection dimension 1", yaxis_title="Projection dimension 2",
                height=420,
            )
            _chart(fig, render_plotly)
            drift = [float(step.drift_loss) for step in replay.steps]
            drift_fig = go.Figure(go.Scatter(
                x=np.arange(1, len(drift) + 1), y=drift, mode="lines",
                name="Consolidation drift",
            ))
            drift_fig.update_layout(
                title="Recorded consolidation drift diagnostic",
                xaxis_title="Target trial",
                yaxis_title="Drift loss (zero when no consolidation event)", height=300,
            )
            _chart(drift_fig, render_plotly)
            st.caption("Prototype paths are projected into the fixed Session 1 reference space. Drift values are recorded only when consolidation triggers; zero elsewhere means no trigger, not zero measured drift.")
        else:
            initial = np.asarray(engine.initial_prototypes_2d, dtype=float)
            fig = go.Figure()
            for index, name in enumerate(engine.class_names):
                fig.add_trace(go.Scatter(
                    x=[initial[index, 0]], y=[initial[index, 1]],
                    mode="markers+text", text=[name], textposition="top center",
                    marker=dict(size=14, symbol="x"), name=f"{name} initial prototype",
                ))
            fig.update_layout(
                title="Initial prototype positions · before target adaptation",
                xaxis_title="Projection dimension 1", yaxis_title="Projection dimension 2",
                height=380,
            )
            _chart(fig, render_plotly)
            st.info("This is the prepared starting state, not prototype movement. Run the offline replay to generate trajectories and consolidation drift.")
    else:
        st.info("Load the prepared features on Adaptation Replay. The starting prototype positions can be plotted before running adaptation.")

    # 6. Confidence distribution.
    st.markdown("### 6 · Confidence distribution")
    if engine is not None:
        memory = PrototypeMemory(engine.initial_prototypes.clone(), momentum=0.9999)
        _, baseline_confidence = memory.predict(
            torch.as_tensor(engine.s2_features, dtype=torch.float32), temperature=0.1
        )
        baseline_values = baseline_confidence.detach().cpu().numpy().reshape(-1)
        confidence_fig = go.Figure()
        confidence_fig.add_trace(go.Histogram(
            x=baseline_values, nbinsx=20, name="Static baseline",
            opacity=0.65,
        ))
        if replay is not None and replay.steps:
            replay_values = np.asarray([step.confidence for step in replay.steps], dtype=float)
            confidence_fig.add_trace(go.Histogram(
                x=replay_values, nbinsx=20, name="Adaptation replay",
                opacity=0.65,
            ))
        confidence_fig.update_layout(
            title="Prediction confidence distribution",
            xaxis_title="Prototype prediction confidence", yaxis_title="Trial count",
            barmode="overlay", height=350,
        )
        _chart(confidence_fig, render_plotly)
        st.caption("Static baseline confidence is computed from prepared Session 2 features and initial prototypes. Replay confidence is added after adaptation. These are model scores, not calibrated probabilities.")
    else:
        st.info("Load prepared features on Adaptation Replay to calculate the static baseline confidence distribution. Adapted confidence is added after replay.")

    # 7. Spike activity by layer.
    st.markdown("### 7 · Spike activity by layer")
    measurement = getattr(engine, "measurement", None) if engine is not None else None
    layer_values = (
        measurement.layer_activity_fraction
        if measurement is not None else {}
    )
    measurement_source = "measurement persisted in the loaded, checkpoint-matched prepared replay cache"
    if not layer_values and run:
        layer_values = (
            run.get("artifact", {}).get("layer_activity_fraction")
            or run.get("run_config", {}).get("layer_activity_fraction")
            or {}
        )
        if layer_values:
            measurement_source = "validated evaluation artifact"
    if isinstance(layer_values, dict) and layer_values:
        names = list(layer_values)
        vals = [float(layer_values[name]) for name in names]
        if all(np.isfinite(value) and 0 <= value <= 1 for value in vals):
            fig = go.Figure(go.Bar(
                x=names, y=vals, text=[f"{value:.1%}" for value in vals],
                textposition="outside",
            ))
            fig.update_layout(
                title="Measured activity fraction by instrumented LIF layer",
                xaxis_title="SNN layer",
                yaxis_title="Fraction of spike opportunities",
                yaxis_range=[0, max(1.0, max(vals) * 1.15)], height=340,
            )
            _chart(fig, render_plotly)
            st.caption(f"Source: {measurement_source}. Activity is a fraction of spike opportunities, not power or energy.")
        else:
            st.warning("Layer activity metadata failed validation and is not plotted.")
    else:
        st.info("Load the prepared feature cache for this checkpoint. The precompute step already records per-layer spike activity; no separate measurement run should be needed.")

    # 8. Consolidation ablation comparison.
    st.markdown("### 8 · Consolidation ablation comparison")
    reports = discover_ablation_reports(root)
    if reports:
        report_item = st.selectbox(
            "Paired ablation report", reports,
            format_func=lambda item: str(item["path"].relative_to(root)),
            key="va_ablation_report",
        )
        paired = paired_ablation_rows(report_item["payload"])
        if paired:
            frame = pd.DataFrame(paired)
            metric_options = {
                "Prequential target accuracy": "target_prequential_acc",
                "Retrospective target accuracy": "target_retrospective_acc",
                "Source retention": "source_accuracy_after_adaptation",
                "Adaptation gain": "adaptation_gain",
                "Measured spike activity": "spike_sparsity",
            }
            label = st.selectbox("Ablation outcome", list(metric_options), key="va_ablation_metric")
            key = metric_options[label]
            on = pd.to_numeric(frame[f"{key}_enabled"], errors="coerce")
            off = pd.to_numeric(frame[f"{key}_disabled"], errors="coerce")
            valid_mask = on.notna() & off.notna()
            compare = pd.DataFrame({
                "Subject": frame.loc[valid_mask, "subject"].astype(str),
                "Seed": frame.loc[valid_mask, "seed"].astype(str),
                "Consolidation ON": on[valid_mask].to_numpy(),
                "Consolidation OFF": off[valid_mask].to_numpy(),
            })
            if not compare.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=compare["Subject"] + " / " + compare["Seed"],
                    y=compare["Consolidation ON"], name="Consolidation ON",
                ))
                fig.add_trace(go.Bar(
                    x=compare["Subject"] + " / " + compare["Seed"],
                    y=compare["Consolidation OFF"], name="Consolidation OFF",
                ))
                fig.update_layout(
                    title=f"Paired experiment ablation · {label}",
                    barmode="group", xaxis_title="Matched subject / seed",
                    yaxis_title="Metric", height=380,
                )
                _chart(fig, render_plotly)
                st.caption(f"Only exact matched subject–seed pairs are included (n={len(compare)}).")
            else:
                st.info("The selected outcome is missing for all matched pairs.")
        else:
            st.warning("This report contains no exact matched ON/OFF subject–seed pairs.")
    else:
        st.info("No multi-run paired ablation report exists yet. You can run a single-checkpoint paired replay below for an exploratory comparison; it is not a substitute for the multi-subject experiment.")
        if engine is not None:
            if st.button("Run paired replay: consolidation ON vs OFF", key="va_run_paired_replay"):
                try:
                    with st.spinner("Running two feature-only replay simulations..."):
                        enabled = engine.run_replay(enable_consolidation=True)
                        disabled = engine.run_replay(enable_consolidation=False)
                    st.session_state["ns_replay_ablation"] = {
                        "enabled": enabled,
                        "disabled": disabled,
                        "subject": engine.subject,
                        "run_id": engine.run_id,
                    }
                except Exception as error:
                    st.error("Paired replay comparison failed.")
                    st.exception(error)
            paired_replay = st.session_state.get("ns_replay_ablation")
            if paired_replay and paired_replay.get("run_id") == engine.run_id:
                enabled = paired_replay["enabled"]
                disabled = paired_replay["disabled"]
                metrics = ["Target prequential accuracy", "Session 1 retention", "Acceptance rate"]
                on_values = [
                    enabled.adapted_target_accuracy,
                    enabled.session1_retention_accuracy,
                    float(np.mean([step.accepted for step in enabled.steps])),
                ]
                off_values = [
                    disabled.adapted_target_accuracy,
                    disabled.session1_retention_accuracy,
                    float(np.mean([step.accepted for step in disabled.steps])),
                ]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=metrics, y=on_values, name="Consolidation ON"))
                fig.add_trace(go.Bar(x=metrics, y=off_values, name="Consolidation OFF"))
                fig.update_layout(
                    title=f"Exploratory paired replay · Subject {engine.subject}",
                    barmode="group", yaxis_title="Score / rate", yaxis_range=[0, 1],
                    height=380,
                )
                _chart(fig, render_plotly)
                st.caption("Same prepared features, checkpoint and default replay settings; only consolidation is toggled. This is one exploratory replay pair, not a multi-seed statistical ablation.")
        else:
            st.info("Load prepared features on Adaptation Replay to run this exploratory paired comparison.")
