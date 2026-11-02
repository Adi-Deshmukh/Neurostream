"""NeuroStream Research Workstation.

The UI is a read-only client of validated local artifacts. Heavy data preparation
is explicit and opt-in; ordinary page navigation never downloads EEG, loads a
checkpoint, fits a projection, or reruns an experiment.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from neurostream.dashboard.registry import (
    discover_ablation_reports,
    discover_experiments,
    load_preprocessing_contract,
    paired_ablation_rows,
    project_root,
    resolve_artifact_path,
)

ROOT = project_root()
PAGES = (
    "Overview", "Signal Analysis", "Model Diagnostics", "Adaptation Replay",
    "Benchmark Lab", "Consolidation Study", "Experiment Registry",
)
CLASS_COLORS = ("#3976A8", "#269B83", "#D39A45", "#8B79B8")


st.set_page_config(
    page_title="NeuroStream Research Workstation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit changed chart/table sizing from use_container_width to width="stretch".
# Detect the installed API so both the project's older minimum and current releases work.
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
_DATAFRAME_HAS_WIDTH = "width" in inspect.signature(st.dataframe).parameters


def render_plotly(figure, **kwargs):
    if _PLOTLY_HAS_WIDTH:
        return st.plotly_chart(figure, width="stretch", **kwargs)
    return st.plotly_chart(figure, use_container_width=True, **kwargs)


def render_dataframe(data, **kwargs):
    if _DATAFRAME_HAS_WIDTH:
        return st.dataframe(data, width="stretch", **kwargs)
    return st.dataframe(data, use_container_width=True, **kwargs)


st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .block-container { padding-top: 1.35rem; padding-bottom: 2.2rem; max-width: 1600px; }
    [data-testid="stSidebar"] { background: #172538; }
    [data-testid="stSidebar"] * { color: #e7edf4; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #172538; }
    .ns-eyebrow { color:#65809a; font-size:.72rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
    .ns-title { color:#15283d; font-size:1.75rem; font-weight:700; letter-spacing:-.035em; margin:.1rem 0 .2rem; }
    .ns-subtitle { color:#65788c; font-size:.91rem; margin-bottom:1.25rem; }
    div[data-testid="stMetric"] { border:1px solid #dce4eb; border-radius:7px; padding:12px 14px; background:#fff; }
    div[data-testid="stMetricLabel"] { color:#64788b; }
    div[data-testid="stDataFrame"] { border:1px solid #e0e7ed; }
    hr { border-color:#e2e8ee; }
    </style>
    """,
    unsafe_allow_html=True,
)

def header(title: str, subtitle: str) -> None:
    st.markdown('<div class="ns-eyebrow">NEUROSTREAM / RESEARCH WORKSTATION</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ns-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ns-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def pct(value) -> str:
    if value is None or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return "N/A"
    return f"{float(value):.1%}"


def number(value, digits=3) -> str:
    if value is None or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def valid_runs(rows):
    return [row for row in rows if row["status"] == "valid"]


def selected_run(rows):
    valid = valid_runs(rows)
    if not valid:
        return None
    labels = [f"S{r['subject']:02d} · seed {r['seed']} · {r['artifact']['run_id']}" for r in valid]
    idx = st.sidebar.selectbox("Validated experiment", range(len(valid)), format_func=lambda i: labels[i])
    return valid[idx]


def metric_frame(rows):
    records = []
    for row in rows:
        m = row["metrics"]
        records.append({
            "Subject": row["subject"], "Seed": row["seed"],
            "Run ID": row["artifact"].get("run_id"),
            "Created UTC": row["artifact"].get("created_at_utc"),
            "Static target": m.get("target_static_baseline_acc"),
            "Prequential target": m.get("target_prequential_acc"),
            "Retrospective target": m.get("target_retrospective_acc"),
            "Source retention": m.get("source_accuracy_after_adaptation"),
            "Adaptation gain": m.get("adaptation_gain"),
            "Spike activity": m.get("spike_sparsity"),
            "Consolidation": bool(m.get("consolidation_enabled", False)),
        })
    return pd.DataFrame(records)


def confusion_figure(matrix, labels, title):
    z = np.asarray(matrix, dtype=float)
    if z.ndim != 2 or z.shape[0] != z.shape[1] or z.shape[0] != len(labels):
        return None
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, colorscale="Blues", colorbar_title="Trials",
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Predicted class", yaxis_title="True class",
                      height=420, margin=dict(l=15, r=15, t=55, b=15))
    return fig


# Discovery is cheap: JSON manifests + hash validation only. No checkpoint deserialization.
if "ns_registry" not in st.session_state:
    st.session_state.ns_registry = discover_experiments(ROOT)
registry = st.session_state.ns_registry
valid = valid_runs(registry)

st.sidebar.markdown("## NeuroStream")
st.sidebar.caption("Adaptive spiking BCI · BNCI2014-001")
page = st.sidebar.radio("Workspace", PAGES, index=0, key="ns_page")
st.sidebar.markdown('---')
st.sidebar.caption(f"{len(valid)} validated runs · {len(registry) - len(valid)} unavailable or rejected")
if st.sidebar.button("Refresh artifact registry"):
    st.session_state.ns_registry = discover_experiments(ROOT)
    st.rerun()

run = selected_run(registry) if page in ("Overview", "Signal Analysis", "Model Diagnostics", "Adaptation Replay") else None

if page == "Overview":
    header("Experiment overview", "Validated cross-session results. Every displayed metric is sourced from a checkpoint-matched evaluation artifact.")
    if not valid:
        st.warning("No validated experiment runs were found. Generate evaluation artifacts first; the dashboard will not invent placeholder metrics.")
        st.info("Run the repository's existing cross-session evaluation sweep to generate results/checkpoints/**/evaluation_manifest.json and matching evaluation JSON files.")
    else:
        run = run or valid[0]
        m = run["metrics"]
        cols = st.columns(6)
        cols[0].metric("Static target baseline", pct(m.get("target_static_baseline_acc")))
        cols[1].metric("Prequential target", pct(m.get("target_prequential_acc")))
        cols[2].metric("Retrospective diagnostic", pct(m.get("target_retrospective_acc")))
        cols[3].metric("Source retention", pct(m.get("source_accuracy_after_adaptation")))
        cols[4].metric("Adaptation gain", pct(m.get("adaptation_gain")))
        cols[5].metric("Measured spike activity", pct(m.get("spike_sparsity")))
        st.caption(f"Selected S{run['subject']:02d} · seed {run['seed']} · run {run['artifact']['run_id']} · evaluation {run['artifact']['evaluation_session']}")
        df = metric_frame(valid)
        left, right = st.columns([1.4, 1])
        with left:
            grouped = df.groupby("Subject", as_index=False)[["Static target", "Prequential target", "Retrospective target"]].mean(numeric_only=True)
            fig = go.Figure()
            for col, color in zip(("Static target", "Prequential target", "Retrospective target"), CLASS_COLORS[:3]):
                fig.add_trace(go.Bar(x=grouped["Subject"], y=grouped[col], name=col, marker_color=color))
            fig.update_layout(title="Target-session performance by subject (mean across available seeds)", barmode="group", yaxis_title="Accuracy", yaxis_range=[0,1], height=390)
            render_plotly(fig)
        with right:
            st.markdown("#### Run provenance")
            st.code(f"Checkpoint\n{run['checkpoint_path']}\n\nSHA-256\n{run['artifact']['checkpoint_sha256']}\n\nArtifact\n{run['evaluation_path']}", language="text")
            st.caption(f"Created: {run['artifact']['created_at_utc']}")
        st.markdown("#### Recent validated runs")
        recent = sorted(valid, key=lambda r: r["artifact"].get("created_at_utc", ""), reverse=True)[:8]
        render_dataframe(metric_frame(recent), hide_index=True)
        st.caption("Prequential accuracy is trial-by-trial prediction before that trial's prototype update. Retrospective accuracy is a separate post-replay diagnostic.")

elif page == "Signal Analysis":
    header("EEG signal analysis", "Inspect preprocessed BNCI2014-001 epochs. Dataset loading is explicit and uses the selected checkpoint's preprocessing contract when available.")
    if run is not None:
        subject = int(run["subject"])
        signal_identity = run["artifact"]["checkpoint_sha256"]
        st.caption(f"Bound to S{subject:02d} · seed {run['seed']} · run {run['artifact']['run_id']}")
    else:
        subject = st.selectbox("Subject", list(range(1, 10)), format_func=lambda s: f"S{s:02d}")
        signal_identity = f"default:{subject}"
        st.info("No validated experiment is selected. Loading uses the BNCI2014-001 loader defaults and is not bound to a model checkpoint.")
    if st.button("Load EEG sessions", type="primary"):
        try:
            with st.spinner("Loading the selected subject's EEG sessions..."):
                from neurostream.data.loader import load_bnci2014_001
                sessions = None
                cue_offset = 2.0
                contract_summary = None
                if run is not None:
                    config = run.get("run_config", {})
                    if config.get("dataset_name", "BNCI2014_001") not in {"BNCI2014_001", "BNCI2014-001"}:
                        raise ValueError("The selected run does not use BNCI2014-001.")
                    contract = load_preprocessing_contract(
                        run["checkpoint_path"],
                        expected_sha256=run["artifact"]["checkpoint_sha256"],
                        expected_subject=subject,
                        expected_seed=int(run["seed"]),
                    )
                    epoch = contract.get("epoch_window_seconds", {})
                    filters = contract.get("filter_settings", {})
                    tmin = float(epoch.get("tmin", config.get("epoch_tmin", 2.0)))
                    tmax = float(epoch.get("tmax", config.get("epoch_tmax", 6.0)))
                    l_freq = float(filters.get("l_freq_hz", config.get("filter_l_freq_hz", 4.0)))
                    h_freq = float(filters.get("h_freq_hz", config.get("filter_h_freq_hz", 100.0)))
                    reference = epoch.get("reference", config.get("epoch_reference", "trial_start"))
                    cue_offset = float(epoch.get("cue_onset_offset_seconds", config.get("cue_onset_offset_seconds", 2.0)))
                    if reference != "trial_start":
                        raise ValueError(f"Unsupported epoch reference in checkpoint contract: {reference}")
                    expected_sfreq = contract.get("sampling_frequency_hz", config.get("sampling_frequency_hz"))
                    dataset_path_value = config.get("dataset_path")
                    cache_path_value = config.get("cache_path")
                    dataset_path = resolve_artifact_path(ROOT, dataset_path_value) if dataset_path_value else None
                    cache_path = resolve_artifact_path(ROOT, cache_path_value) if cache_path_value else None
                    sessions = load_bnci2014_001(
                        subjects=subject, tmin=tmin, tmax=tmax,
                        l_freq=l_freq, h_freq=h_freq, epoch_reference=reference,
                        normalization="demean",
                        dataset_path=str(dataset_path) if dataset_path else None,
                        cache_path=str(cache_path) if cache_path else None,
                    )
                    channel_order = contract.get("channel_order")
                    norm_payload = contract.get("normalization")
                    for session in sessions.values():
                        if expected_sfreq is not None and not np.isclose(
                            session.sfreq, float(expected_sfreq), atol=1e-3
                        ):
                            raise ValueError(
                                f"Session sampling frequency {session.sfreq:g} Hz does not match "
                                f"checkpoint contract {float(expected_sfreq):g} Hz."
                            )
                        if channel_order and list(session.channel_names) != list(channel_order):
                            raise ValueError("EEG channel order does not match the checkpoint preprocessing contract.")
                        if not isinstance(norm_payload, dict):
                            raise ValueError("Checkpoint normalization metadata is missing or malformed.")
                        method = norm_payload.get("method")
                        if method == "training_fitted_channel_zscore":
                            mean = np.asarray(norm_payload.get("mean"), dtype=np.float32).reshape(-1)
                            std = np.asarray(norm_payload.get("std"), dtype=np.float32).reshape(-1)
                            if mean.size != session.X.shape[1] or std.size != session.X.shape[1]:
                                raise ValueError("Saved channel normalizer dimensions do not match the EEG channels.")
                            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                                raise ValueError("Saved channel normalizer contains invalid values.")
                            session.X = (
                                (session.X - mean.reshape(1, -1, 1))
                                / std.reshape(1, -1, 1)
                            ).astype(np.float32, copy=False)
                        elif method == "per_trial_channel_zscore":
                            from neurostream.data.preprocessing import zscore_channels
                            session.X = zscore_channels(session.X)
                        elif method != "per_trial_demean_preserve_channel_scale":
                            raise ValueError(f"Unsupported checkpoint normalization method: {method}")
                        session.normalization = {"method": method}
                    contract_summary = {
                        "epoch_window_seconds": {"tmin": tmin, "tmax": tmax, "reference": reference},
                        "filter_settings": {"l_freq_hz": l_freq, "h_freq_hz": h_freq},
                        "normalization": norm_payload.get("method") if isinstance(norm_payload, dict) else None,
                        "sampling_frequency_hz": expected_sfreq,
                        "channel_order": channel_order,
                    }
                else:
                    sessions = load_bnci2014_001(subjects=subject)
                st.session_state["ns_signal_sessions"] = sessions
                st.session_state["ns_signal_subject"] = subject
                st.session_state["ns_signal_identity"] = signal_identity
                st.session_state["ns_signal_cue_offset"] = cue_offset
                st.session_state["ns_signal_contract"] = contract_summary
        except Exception as error:
            st.error("EEG session loading failed. Check the MOABB dataset cache, network access and selected checkpoint preprocessing contract.")
            st.exception(error)
    sessions = st.session_state.get("ns_signal_sessions")
    if (
        not sessions
        or st.session_state.get("ns_signal_subject") != subject
        or st.session_state.get("ns_signal_identity") != signal_identity
    ):
        st.info("EEG is not loaded for this subject/experiment. Click Load EEG sessions; loading is separate from dashboard navigation.")
    else:
        names = list(sessions)
        session_name = st.selectbox("Session", names)
        data = sessions[session_name]
        trial = st.slider("Trial", 1, int(data.X.shape[0]), 1) - 1
        channel_names = list(data.channel_names) or [f"Channel {i+1}" for i in range(data.X.shape[1])]
        channel = st.selectbox("Channel", range(data.X.shape[1]), format_func=lambda i: channel_names[i])
        start_s = st.number_input("Window start (s)", min_value=0.0, max_value=max(0.0, (data.X.shape[-1]-2)/data.sfreq), value=0.0, step=0.25)
        max_time = data.X.shape[-1] / data.sfreq
        end_min = min(max_time, start_s + 0.01)
        end_default = min(max_time, max(start_s + 0.01, 4.0))
        end_s = st.number_input(
            "Window end (s)", min_value=end_min, max_value=max_time,
            value=end_default, step=0.25,
        )
        a, b = int(start_s*data.sfreq), min(data.X.shape[-1], int(end_s*data.sfreq))
        time = np.arange(a, b) / data.sfreq
        signal = data.X[trial, channel, a:b]
        fig = go.Figure(go.Scatter(x=time, y=signal, mode="lines", line=dict(color="#3976A8", width=1.2), name=channel_names[channel]))
        cue_time = float(st.session_state.get("ns_signal_cue_offset", 2.0)) - data.tmin
        if data.epoch_reference == "trial_start" and start_s <= cue_time <= end_s:
            fig.add_vline(
                x=cue_time, line_dash="dash", line_color="#D39A45",
                annotation_text="Imagery cue onset",
            )
        fig.update_layout(title=f"{session_name} · trial {trial+1} · {channel_names[channel]}",
                          xaxis_title="Time (s) from epoch start", yaxis_title="Amplitude (preprocessed units)", height=380)
        render_plotly(fig)
        st.caption(f"Sampling rate: {data.sfreq:g} Hz · epoch: {data.tmin:g} to {data.tmax:g} s relative to {data.epoch_reference} · preprocessing: {data.l_freq:g}–{data.h_freq:g} Hz band-pass.")
        if data.epoch_reference == "trial_start":
            st.caption("The dataset event marks trial start, which precedes this epoch. The dashed line marks the imagery cue onset when it falls inside the selected window.")
        if st.session_state.get("ns_signal_contract"):
            st.caption(f"Checkpoint preprocessing contract: {st.session_state['ns_signal_contract']}")
        if st.button("Compute channel × band power"):
            from scipy.integrate import trapezoid
            from scipy.signal import welch
            freqs, psd = welch(data.X[trial], fs=data.sfreq, axis=-1, nperseg=min(256, data.X.shape[-1]))
            bands = {"Alpha (8–13 Hz)": (8,13), "Beta (13–30 Hz)": (13,30), "Gamma (30–100 Hz)": (30,min(100,data.sfreq/2))}
            z = []
            for _, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs < hi)
                z.append(trapezoid(psd[:, mask], freqs[mask], axis=1) if mask.any() else np.full(data.X.shape[1], np.nan))
            heat = go.Figure(go.Heatmap(z=np.asarray(z).T, x=list(bands), y=channel_names, colorscale="Viridis", colorbar_title="Power (preprocessed units²)"))
            heat.update_layout(title="Channel × frequency-band power · selected trial", xaxis_title="Frequency band", yaxis_title="EEG channel", height=620)
            render_plotly(heat)
        st.markdown("#### Session 1 vs Session 2")
        train_name = next((name for name in names if "train" in name.lower()), None)
        test_name = next((name for name in names if "test" in name.lower()), None)
        if train_name and test_name:
            train_data, test_data = sessions[train_name], sessions[test_name]
            compare_trial = st.number_input(
                "Example trial index in each session (not a paired trial)",
                min_value=1, max_value=max(1, min(train_data.X.shape[0], test_data.X.shape[0])),
                value=1, step=1,
            ) - 1
            compare_fig = go.Figure()
            for session_label, session_data, color in (
                (train_name, train_data, "#3976A8"),
                (test_name, test_data, "#269B83"),
            ):
                values = session_data.X[compare_trial, channel]
                x_values = np.arange(values.size) / session_data.sfreq
                compare_fig.add_trace(go.Scatter(
                    x=x_values, y=values, mode="lines", name=session_label,
                    line=dict(color=color, width=1.1),
                ))
            compare_fig.update_layout(
                title=f"Illustrative session comparison · {channel_names[channel]}",
                xaxis_title="Time (s) from epoch start",
                yaxis_title="Amplitude (preprocessed units)", height=360,
            )
            render_plotly(compare_fig)
        else:
            st.info("Both training and test sessions are not present in the loaded dataset result.")
        st.markdown("#### Epoch quality checks")
        epoch = data.X[trial]
        flat_channels = [
            channel_names[i] for i in range(epoch.shape[0])
            if float(np.std(epoch[i])) <= 1e-12
        ]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Non-finite samples", str(int((~np.isfinite(epoch)).sum())))
        q2.metric("Flat channels", str(len(flat_channels)))
        q3.metric("Mean absolute amplitude", number(float(np.mean(np.abs(epoch))), 5))
        q4.metric("Max peak-to-peak", number(float(np.ptp(epoch, axis=-1).max()), 5))
        if flat_channels:
            st.warning("Flat channels: " + ", ".join(flat_channels))
        st.info("The loader returns preprocessed epochs, not raw continuous EEG. Raw, filtered-only and normalized-stage signals are not separately retained here; the displayed waveform must not be labelled raw.")

elif page == "Model Diagnostics":
    header("Model diagnostics", "Feature-space and measured spike-activity diagnostics from the selected prepared replay artifact.")
    if run is None:
        st.warning("No validated run is available.")
    else:
        st.caption(f"S{run['subject']:02d} · seed {run['seed']} · checkpoint {run['artifact']['checkpoint_sha256'][:16]}…")
        artifact = run["artifact"]
        config = artifact.get("run_config", {})
        c1,c2,c3 = st.columns(3)
        c1.metric("Training epochs", str(config.get("epochs", "N/A")))
        c2.metric("Encoder timesteps", str(config.get("encoder_timesteps", "N/A")))
        c3.metric("Feature dimension", str(config.get("model_out_features", "N/A")))
        activity = artifact.get("metrics", {}).get("spike_sparsity")
        st.metric("Measured LIF spike activity", pct(activity))
        layer = artifact.get("layer_activity_fraction") or config.get("layer_activity_fraction")
        if isinstance(layer, dict) and layer:
            fig = go.Figure(go.Bar(x=list(layer), y=list(layer.values()), marker_color="#269B83"))
            fig.update_layout(title="Measured activity fraction by instrumented layer", yaxis_title="Fraction of spike opportunities", yaxis_range=[0,1], height=360)
            render_plotly(fig)
        else:
            st.info("Per-layer activity is not present in this evaluation artifact. No per-neuron or layer-level values are inferred.")
        st.markdown("#### Architecture")
        hidden_width = config.get("model_hidden", "checkpoint-defined")
        feature_width = config.get("model_out_features", "checkpoint-defined")
        st.code(
            f"Input spike tensor → FC-{hidden_width} → LIF1 → "
            f"FC-{feature_width} → LIF2 → mean spike-rate feature ({feature_width}-D)",
            language="text",
        )
        if hidden_width == "checkpoint-defined":
            st.caption("The evaluation manifest does not repeat the hidden width; the checkpoint remains the architecture source of truth.")
        st.caption("Architecture is read from the project model definition. Spike activity is not hardware power or energy consumption.")
        cache_root = ROOT / "results" / "dashboard_cache"
        matching_metadata = []
        if cache_root.exists():
            for metadata_path in cache_root.glob("*/metadata.json"):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if not isinstance(metadata, dict):
                        continue
                    if (
                        metadata.get("checkpoint_ref") == artifact.get("checkpoint_sha256")
                        and int(metadata.get("subject", -1)) == int(run["subject"])
                    ):
                        matching_metadata.append((metadata_path, metadata))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
        if matching_metadata:
            metadata_path, metadata = matching_metadata[0]
            measured = metadata.get("measurement", {})
            if not isinstance(measured, dict):
                measured = {}
            layer_rates = measured.get("layer_activity_fraction", {})
            if isinstance(layer_rates, dict) and layer_rates:
                layer_fig = go.Figure(go.Bar(
                    x=list(layer_rates), y=list(layer_rates.values()),
                    marker_color="#269B83",
                ))
                layer_fig.update_layout(
                    title="Measured LIF layer activity",
                    yaxis_title="Fraction of spike opportunities",
                    yaxis_range=[0, 1], height=340,
                )
                render_plotly(layer_fig)
            st.caption(
                f"Prepared replay provenance: {metadata.get('run_id')} · "
                f"projection {metadata.get('projection_method')} · "
                f"measurement {measured.get('measured_at_utc', 'N/A')}"
            )
            if st.button("Load feature-space distribution"):
                try:
                    with np.load(metadata_path.parent / "features.npz", allow_pickle=False) as arrays:
                        s1_xy = arrays["session1_features_2d"].copy()
                        s2_xy = arrays["session2_features_2d"].copy()
                        s1_y = arrays["session1_labels"].copy()
                        s2_y = arrays["session2_labels"].copy()
                    st.session_state["ns_feature_view"] = {
                        "identity": artifact["checkpoint_sha256"],
                        "s1_xy": s1_xy, "s2_xy": s2_xy, "s1_y": s1_y, "s2_y": s2_y,
                    }
                except Exception as error:
                    st.error("Prepared feature arrays could not be loaded or validated.")
                    st.exception(error)
            feature_view = st.session_state.get("ns_feature_view")
            if feature_view and feature_view.get("identity") == artifact["checkpoint_sha256"]:
                feature_fig = go.Figure()
                names = artifact.get("label_names", [])
                for class_index, class_name in enumerate(names):
                    for coords, labels, session_label, opacity in (
                        (feature_view["s1_xy"], feature_view["s1_y"], "Session 1", .25),
                        (feature_view["s2_xy"], feature_view["s2_y"], "Session 2", .75),
                    ):
                        mask = labels == class_index
                        if np.any(mask):
                            feature_fig.add_trace(go.Scatter(
                                x=coords[mask, 0], y=coords[mask, 1], mode="markers",
                                marker=dict(size=5, opacity=opacity, color=CLASS_COLORS[class_index % 4]),
                                name=f"{session_label} · {class_name}",
                            ))
                feature_fig.update_layout(
                    title="Fixed reference feature-space distribution",
                    xaxis_title="Projection dimension 1", yaxis_title="Projection dimension 2",
                    height=520,
                )
                render_plotly(feature_fig)
        else:
            st.info("No prepared replay metadata matches this checkpoint. Per-layer activity and feature-space views are unavailable until the matching cache is prepared.")
        confusion = config.get("prequential_confusion_matrix")
        labels = artifact.get("label_names", [])
        if confusion is not None and labels:
            fig = confusion_figure(confusion, labels, "Prequential target confusion matrix")
            if fig is not None:
                render_plotly(fig)
        else:
            st.info("No prequential confusion matrix was stored for this run.")

elif page == "Adaptation Replay":
    header("Offline adaptation replay", "Scrub a precomputed feature replay. No live EEG acquisition or model inference occurs during trial navigation.")
    if run is None:
        st.warning("Select a validated run on the Overview page after adding a matching prepared replay artifact.")
    else:
        subject, seed = run["subject"], run["seed"]
        checkpoint = run["checkpoint_path"]
        prototype = checkpoint.with_name(f"prototypes_subject_{subject}.pt")
        projection = st.selectbox("Projection basis", ["PCA", "UMAP"], index=0)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else "missing"
        prototype_hash = hashlib.sha256(prototype.read_bytes()).hexdigest() if prototype.is_file() else "missing"
        cache_dir = ROOT / "results" / "dashboard_cache" / (
            f"subject_{subject}_{checkpoint.stem}_"
            f"{checkpoint_hash[:12]}_{prototype_hash[:12]}_{projection.lower()}"
        )
        cache_file = cache_dir / "metadata.json"
        cache_inputs = (
            cache_file, cache_dir / "features.npz", cache_dir / "projector.joblib"
        )
        if not checkpoint.is_file():
            st.error(f"Selected checkpoint no longer exists: {checkpoint}")
        elif checkpoint_hash != run["artifact"]["checkpoint_sha256"]:
            st.error("Checkpoint changed after registry validation. Refresh the artifact registry before replay.")
        elif not prototype.is_file():
            st.warning(f"Prototype file not found next to selected checkpoint: {prototype}")
        elif not all(path.is_file() for path in cache_inputs):
            st.warning(f"Prepared replay cache is missing or incomplete: {cache_dir}")
            st.code(
                f'python scripts/precompute_dashboard_artifacts.py --subject {subject} '
                f'--checkpoint "{checkpoint}" --prototypes "{prototype}" --projection {projection}',
                language="bash",
            )
        else:
            cache_signature = ":".join(
                f"{path.stat().st_mtime_ns}-{path.stat().st_size}"
                for path in cache_inputs
            )
            identity = hashlib.sha256(
                f"{checkpoint_hash}:{prototype_hash}:"
                f"{projection}:{cache_signature}".encode()
            ).hexdigest()
            if st.session_state.get("ns_engine_identity") != identity:
                st.session_state.pop("ns_engine", None)
                st.session_state.pop("ns_replay", None)
                st.session_state["ns_engine_identity"] = identity
            if st.button("Load prepared features", type="primary"):
                try:
                    from neurostream.dashboard.simulation import TrialReplayEngine
                    st.session_state["ns_engine"] = TrialReplayEngine.from_prepared_artifact(
                        cache_dir, checkpoint_ref=checkpoint_hash,
                        subject=subject, prototypes_ref=prototype_hash,
                    )
                    st.session_state.pop("ns_replay", None)
                except Exception as error:
                    st.error("Prepared replay cache was rejected. Re-run the preparation command shown above.")
                    st.exception(error)
            engine = st.session_state.get("ns_engine")
            if engine is not None:
                if projection == "UMAP" and not str(engine.projection_method).upper().startswith("UMAP"):
                    st.warning(
                        f"UMAP was requested, but this prepared artifact uses "
                        f"{engine.projection_method}. The fallback is shown explicitly."
                    )
                else:
                    st.caption(f"Prepared reference projection: {engine.projection_method}")
                with st.form("ns_replay_settings"):
                    momentum = st.slider("Prototype momentum", 0.50, 0.99, 0.90, 0.01)
                    threshold = st.slider("Confidence threshold", 0.25, 0.95, 0.60, 0.05)
                    consolidation = st.checkbox("Enable sleep consolidation", value=True)
                    interval = st.number_input("Consolidation interval", 10, 100, 25, 5)
                    sleep_steps = st.number_input("Correction iterations", 1, 15, 5, 1)
                    pull_rate = st.slider("Pull strength", 0.1, 1.0, 0.5, 0.1)
                    execute = st.form_submit_button("Run offline replay")
                config_id = (momentum, threshold, consolidation, int(interval), int(sleep_steps), pull_rate)
                if execute:
                    try:
                        st.session_state["ns_replay"] = engine.run_replay(
                            momentum=momentum, confidence_threshold=threshold,
                            enable_consolidation=consolidation, consolidation_interval=int(interval),
                            sleep_steps=int(sleep_steps), pull_rate=pull_rate,
                        )
                        st.session_state["ns_replay_config"] = config_id
                    except Exception as error:
                        st.error("Offline replay failed for this configuration. The other dashboard pages remain available.")
                        st.exception(error)
                replay = st.session_state.get("ns_replay")
                if replay is not None:
                    total = len(replay.steps)
                    if total == 0:
                        st.error("Replay contains no target trials.")
                    else:
                        slider_key = f"ns_trial_{identity}"
                        if slider_key not in st.session_state:
                            st.session_state[slider_key] = 1
                        col1,col2,col3 = st.columns([3,1,1])
                        with col2:
                            if st.button("Jump to start"):
                                st.session_state[slider_key] = 1
                        with col3:
                            if st.button("Jump to end"):
                                st.session_state[slider_key] = total
                        with col1:
                            current = st.slider("Trial position", 1, total, key=slider_key)
                        step = replay.steps[current-1]
                        a,b,c,d = st.columns(4)
                        a.metric("Prequential accuracy", pct(step.cumulative_accuracy))
                        b.metric("Static baseline", pct(step.static_baseline_accuracy))
                        c.metric("Confidence", pct(step.confidence))
                        d.metric("Update decision", "Accepted" if step.accepted else "Rejected")
                        left,right = st.columns([1.3,1])
                        with left:
                            fig = go.Figure()
                            for idx,name in enumerate(replay.class_names):
                                mask = replay.session1_labels == idx
                                fig.add_trace(go.Scatter(x=replay.session1_features_2d[mask,0], y=replay.session1_features_2d[mask,1], mode="markers", marker=dict(size=5,opacity=.22,color=CLASS_COLORS[idx%4]), name=f"Session 1 · {name}"))
                            history = replay.prototype_history_2d[:current]
                            for idx,name in enumerate(replay.class_names):
                                fig.add_trace(go.Scatter(x=history[:,idx,0],y=history[:,idx,1],mode="lines",line=dict(color=CLASS_COLORS[idx%4],dash="dot"),name=f"{name} prototype"))
                            current_proto = step.prototypes_2d
                            fig.add_trace(go.Scatter(x=current_proto[:,0],y=current_proto[:,1],mode="markers",marker=dict(symbol="star",size=17,color=list(CLASS_COLORS[:len(replay.class_names)])),name="Current prototypes"))
                            fig.add_trace(go.Scatter(x=[step.feature_2d[0]],y=[step.feature_2d[1]],mode="markers",marker=dict(symbol="diamond",size=13,color="#D39A45"),name="Current trial"))
                            fig.update_layout(title="Fixed reference feature space",xaxis_title="Projection dimension 1",yaxis_title="Projection dimension 2",height=540)
                            render_plotly(fig)
                        with right:
                            xs = np.arange(1,current+1)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=xs,y=[x.cumulative_accuracy for x in replay.steps[:current]],name="Prequential"))
                            fig.add_trace(go.Scatter(x=xs,y=[x.static_baseline_accuracy for x in replay.steps[:current]],name="Static",line=dict(dash="dash")))
                            fig.update_layout(title="Accuracy progression",xaxis_title="Trial",yaxis_title="Cumulative accuracy",yaxis_range=[0,1],height=360)
                            render_plotly(fig)
                            st.metric("Session 1 train retention", pct(replay.session1_retention_accuracy))
                            st.metric("Adaptation gain", pct(replay.adaptation_gain))
                        accepted_count = sum(1 for item in replay.steps[:current] if item.accepted)
                        consolidation_count = sum(1 for item in replay.steps[:current] if item.sleep_consolidation_triggered)
                        accepted_col, rejected_col, consolidation_col = st.columns(3)
                        accepted_col.metric("Accepted updates", str(accepted_count))
                        rejected_col.metric("Rejected trials", str(current - accepted_count))
                        consolidation_col.metric("Consolidation triggers", str(consolidation_count))
                        drift_fig = go.Figure()
                        drift_fig.add_trace(go.Scatter(
                            x=np.arange(1, current + 1),
                            y=[item.drift_loss for item in replay.steps[:current]],
                            mode="lines", name="Recorded drift loss",
                        ))
                        drift_fig.update_layout(
                            title="Recorded consolidation drift diagnostic",
                            xaxis_title="Trial", yaxis_title="Drift loss (recorded at trigger)",
                            height=300,
                        )
                        render_plotly(drift_fig)
                        confidence_fig = go.Figure(go.Histogram(
                            x=[item.confidence for item in replay.steps[:current]],
                            nbinsx=20, marker_color="#3976A8",
                        ))
                        confidence_fig.add_vline(
                            x=threshold, line_dash="dash", line_color="#D39A45",
                            annotation_text="Acceptance threshold",
                        )
                        confidence_fig.update_layout(
                            title="Confidence distribution through current trial",
                            xaxis_title="Prediction confidence", yaxis_title="Trial count",
                            height=300,
                        )
                        render_plotly(confidence_fig)
                        displacement = np.linalg.norm(
                            step.prototypes_2d - replay.initial_prototypes_2d, axis=1
                        )
                        st.markdown("#### Per-class prototype movement")
                        render_dataframe(pd.DataFrame({
                            "Class": replay.class_names,
                            "2D projected displacement": displacement,
                        }), hide_index=True)
                        st.caption("Displacement is measured in the fixed 2D projection coordinates. High-dimensional per-trial prototype history is not stored by this replay schema.")
                        decisions = pd.DataFrame([{"Trial":x.step,"True":replay.class_names[x.true_label],"Predicted":replay.class_names[x.predicted_label],"Confidence":x.confidence,"Accepted":x.accepted,"Consolidation":x.sleep_consolidation_triggered,"Drift":x.drift_loss} for x in replay.steps])
                        render_dataframe(decisions.iloc[max(0,current-20):current],hide_index=True)

elif page == "Benchmark Lab":
    header("Benchmark lab", "Compare validated subjects and seeds. Missing metrics remain missing; they are never replaced with zero.")
    if not valid:
        st.warning("No validated evaluation runs are available.")
    else:
        df = metric_frame(valid)
        subjects = sorted(df["Subject"].unique())
        seeds = sorted(df["Seed"].unique())
        chosen_subjects = st.multiselect("Subjects", subjects, default=subjects)
        chosen_seeds = st.multiselect("Seeds", seeds, default=seeds)
        filtered = df[df["Subject"].isin(chosen_subjects) & df["Seed"].isin(chosen_seeds)]
        if filtered.empty:
            st.info("No runs match the selected filters.")
        else:
            metric = st.selectbox("Heatmap metric", ["Prequential target", "Static target", "Retrospective target", "Source retention", "Adaptation gain"])
            pivot = filtered.pivot_table(index="Subject", columns="Seed", values=metric, aggfunc="mean")
            fig = go.Figure(go.Heatmap(z=pivot.to_numpy(dtype=float), x=[str(x) for x in pivot.columns], y=[str(x) for x in pivot.index], colorscale="Blues", zmin=0 if "gain" not in metric.lower() else None, zmax=1 if "gain" not in metric.lower() else None, colorbar_title="Accuracy / gain", hovertemplate="Subject %{y}<br>Seed %{x}<br>Value %{z:.3f}<extra></extra>"))
            fig.update_layout(title=f"Subject × seed · {metric}", xaxis_title="Seed", yaxis_title="Subject", height=420)
            render_plotly(fig)
            means = filtered.groupby("Subject")[metric].mean()
            stds = filtered.groupby("Subject")[metric].std()
            summary = pd.DataFrame({"Mean":means,"Seed SD":stds})
            st.markdown("#### Subject-wise variability")
            render_dataframe(summary)
            seed_summary = filtered.groupby("Seed")[metric].agg(["mean", "std", "count"])
            st.markdown("#### Seed-wise variability")
            render_dataframe(seed_summary)
            fig2 = go.Figure()
            for col,color in zip(("Static target","Prequential target","Retrospective target"),CLASS_COLORS[:3]):
                grp=filtered.groupby("Subject")[col].mean()
                fig2.add_trace(go.Bar(x=grp.index,y=grp.values,name=col,marker_color=color))
            fig2.update_layout(title="Static, prequential and retrospective comparison",barmode="group",yaxis_title="Accuracy",yaxis_range=[0,1],height=390)
            render_plotly(fig2)
            eligible = [
                item for item in valid
                if item["subject"] in chosen_subjects and item["seed"] in chosen_seeds
            ]
            if eligible:
                cm_run = st.selectbox(
                    "Confusion matrix run", eligible,
                    format_func=lambda item: f"S{item['subject']:02d} · seed {item['seed']}",
                )
                cm = cm_run["run_config"].get("prequential_confusion_matrix")
                labels = cm_run["label_names"]
                if cm is not None and labels:
                    fig_cm = confusion_figure(cm, labels, "Stored prequential confusion matrix")
                    if fig_cm:
                        render_plotly(fig_cm)
                else:
                    st.info("The selected evaluation artifact does not contain a confusion matrix.")
        render_dataframe(filtered,hide_index=True)

elif page == "Consolidation Study":
    header("Consolidation study", "Paired comparison of consolidation ON versus OFF. Only matched subject–seed pairs are compared.")
    reports = discover_ablation_reports(ROOT)
    if not reports:
        st.warning("No paired ablation report found under results/ablations/consolidation/. Run the paired ablation experiment before drawing conclusions.")
    else:
        report_item = st.selectbox("Ablation report", reports, format_func=lambda x: str(x["path"].relative_to(ROOT)))
        report = report_item["payload"]
        paired = paired_ablation_rows(report)
        st.caption(f"Protocol: {report['protocol']} · matched pairs: {len(paired)}")
        if not paired:
            st.warning("The report contains no matching enabled/disabled subject–seed pairs.")
        else:
            df = pd.DataFrame(paired)
            choices = {
                "Prequential target accuracy":"target_prequential_acc",
                "Retrospective target accuracy":"target_retrospective_acc",
                "Source retention":"source_accuracy_after_adaptation",
                "Static target baseline":"target_static_baseline_acc",
                "Adaptation gain":"adaptation_gain",
                "Forgetting (source accuracy drop)":"source_retention_delta",
                "Measured spike activity":"spike_sparsity",
            }
            label=st.selectbox("Paired outcome",list(choices))
            key=choices[label]
            sign = -1.0 if key == "source_retention_delta" else 1.0
            diffs=df[f"{key}_difference"].dropna().to_numpy(dtype=float) * sign
            if diffs.size:
                mean=float(np.mean(diffs))
                sd=float(np.std(diffs,ddof=1)) if diffs.size>1 else float("nan")
                se=sd/np.sqrt(diffs.size) if diffs.size>1 else float("nan")
                from scipy.stats import t
                half=float(t.ppf(.975,diffs.size-1)*se) if diffs.size>1 else float("nan")
                a,b,c=st.columns(3)
                a.metric("Mean paired difference (ON − OFF)",f"{mean:+.3%}")
                b.metric("95% t interval",f"[{mean-half:+.3%}, {mean+half:+.3%}]" if np.isfinite(half) else "N/A (n=1)")
                c.metric("Matched pairs",str(diffs.size))
                fig=go.Figure(go.Box(y=diffs,boxpoints="all",name="ON − OFF",marker_color="#3976A8"))
                fig.add_hline(y=0,line_dash="dash",line_color="#65788c")
                fig.update_layout(title=f"Paired differences · {label}",yaxis_title="Difference (fraction)",height=380)
                render_plotly(fig)
            else:
                st.info("Selected outcome is not available for the matched pairs.")
            trigger_rows = []
            for condition in ("enabled", "disabled"):
                for item in report.get("conditions", {}).get(condition, {}).get("runs", []):
                    trigger_rows.append({
                        "Condition": condition,
                        "Subject": item.get("subject"),
                        "Seed": item.get("seed"),
                        "Consolidation phases": item.get("consolidation_phases"),
                    })
            if trigger_rows:
                st.markdown("#### Consolidation trigger frequency")
                trigger_frame = pd.DataFrame(trigger_rows)
                render_dataframe(trigger_frame, hide_index=True)
            st.caption("The current ablation report records consolidation phase counts but not prototype coordinates immediately before and after each consolidation event. Those drift measurements are therefore not fabricated or plotted.")
            render_dataframe(df,hide_index=True)

elif page == "Experiment Registry":
    header("Experiment registry", "Manifest, checkpoint and evaluation-artifact integrity. No model is instantiated to populate this page.")
    if not registry:
        st.info("No evaluation manifests found under results/.")
    else:
        rows=[]
        for item in registry:
            rows.append({
                "Subject":item["subject"],"Seed":item["seed"],"Status":item["status"],
                "Checkpoint":str(item["checkpoint_path"] or ""),
                "Evaluation artifact":str(item["evaluation_path"] or ""),
                "Manifest":str(item["manifest_path"]),
                "Error":item["error"],
            })
        render_dataframe(pd.DataFrame(rows),hide_index=True)
        st.markdown("#### Selected artifact provenance")
        if valid:
            item=st.selectbox("Run",valid,format_func=lambda r:f"S{r['subject']:02d} · seed {r['seed']}")
            st.json({
                "run_id":item["artifact"]["run_id"],
                "created_at_utc":item["artifact"]["created_at_utc"],
                "checkpoint_sha256":item["artifact"]["checkpoint_sha256"],
                "train_session":item["artifact"]["train_session"],
                "evaluation_session":item["artifact"]["evaluation_session"],
                "protocol":item["artifact"]["protocol"],
                "label_values":item["artifact"]["label_values"],
                "label_names":item["artifact"]["label_names"],
                "run_config":item["artifact"]["run_config"],
            })
        cache_root=ROOT/"results"/"dashboard_cache"
        cache_entries=list(cache_root.glob("*/metadata.json")) if cache_root.exists() else []
        st.caption(f"Prepared feature caches discovered: {len(cache_entries)}")
        if cache_entries:
            render_dataframe(pd.DataFrame([{"Cache":str(p.parent.relative_to(ROOT)),"Metadata bytes":p.stat().st_size} for p in cache_entries]),hide_index=True)

