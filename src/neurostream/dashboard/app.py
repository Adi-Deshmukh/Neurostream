"""NeuroStream Interactive Evaluation Dashboard.

Professional research platform visualizing offline trial-by-trial replay, prototype
trajectories, and online-style adaptation across all 9 subjects from checkpoints.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

from neurostream.dashboard.helpers import resolve_repo_path, trial_slider_key
from neurostream.dashboard.simulation import ReplaySimulationData, TrialReplayEngine
from neurostream.data.loader import load_bnci2014_001
from neurostream.evaluation.artifacts import load_evaluation_artifact

# Resolve paths independently of the process working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = next(
    (path.resolve() for path in (
        PROJECT_ROOT / 'results' / 'checkpoints',
        PROJECT_ROOT / 'scripts' / 'results' / 'checkpoints',
    ) if path.is_dir()),
    (PROJECT_ROOT / 'results' / 'checkpoints').resolve(),
)

# ── Page Configuration & Enterprise Styling ────────────────────────────────────

st.set_page_config(
    page_title="NeuroStream | Adaptive Neuromorphic BCI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom enterprise CSS with professional typography and color system
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Banner */
    .dashboard-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        color: #F8FAFC;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 0.98rem;
        margin-top: 6px;
        font-weight: 400;
    }
    .badge-pill {
        display: inline-block;
        background-color: #0284C7;
        color: white;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    /* Theme-adaptive Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(148, 163, 184, 0.4);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.85;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 2px solid rgba(148, 163, 184, 0.2);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        color: #0284C7 !important;
        border-bottom: 3px solid #0284C7 !important;
        background-color: transparent !important;
    }

    /* Card Panels */
    .content-card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render Top Banner
st.markdown(
    """
    <div class="dashboard-header">
        <span class="badge-pill">RESEARCH PLATFORM</span>
        <span class="badge-pill" style="background-color: #10B981;">REAL DATA • NO MOCKS</span>
        <div class="header-title">🧠 NeuroStream: Adaptive Spiking BCI Architecture</div>
        <div class="header-subtitle">
            Empirical evaluation of online motor-imagery adaptation with zero raw-data storage and measured spike activity.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Controls ─────────────────────────────────────────────────────────

st.sidebar.markdown("### 🎛️ Subject & Model Select")

subject_id = st.sidebar.selectbox(
    "Subject Profile",
    options=list(range(1, 10)),
    index=0,
    format_func=lambda s: f"Subject S{s:02d} (BNCI2014-001)",
    help="Select any of the 9 benchmark subjects. Loads real Phase A SNN weights and prototypes.",
)

manifest_path = CHECKPOINT_DIR / "evaluation_manifest.json"
selected_run = None
manifest_runs_by_subject: dict[int, list[dict[str, object]]] = {}
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_runs = manifest.get("runs", [])
        manifest_runs_by_subject = {
            subject: [run for run in manifest_runs if int(run.get("subject", -1)) == subject]
            for subject in range(1, 10)
        }
        subject_runs = [
            run for run in manifest.get("runs", [])
            if int(run.get("subject", -1)) == subject_id
        ]
        if subject_runs:
            seeds = [int(run["seed"]) for run in subject_runs]
            selected_seed = st.sidebar.selectbox(
                "Evaluation seed",
                options=seeds,
                help="Select the exact checkpoint and held-out artifact run.",
            )
            selected_run = next(run for run in subject_runs if int(run["seed"]) == selected_seed)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        st.sidebar.error(f"Evaluation manifest rejected: {error}")

if selected_run is not None:
    ckpt_file = resolve_repo_path(PROJECT_ROOT, selected_run["checkpoint"])
    artifact_file = resolve_repo_path(PROJECT_ROOT, selected_run["evaluation_artifact"])
    proto_file = ckpt_file.with_name(f"prototypes_subject_{subject_id}.pt")
else:
    ckpt_file = CHECKPOINT_DIR / f"phase_a_subject_{subject_id}.pt"
    proto_file = CHECKPOINT_DIR / f"prototypes_subject_{subject_id}.pt"
    artifact_file = ckpt_file.parent / f"evaluation_subject_{subject_id}.json"

projection_type = st.sidebar.radio(
    "2D Reference Basis",
    options=["PCA", "UMAP"],
    index=0,
    help="Fixed 2D reference projection fitted on Session 1 features.",
)

active_page = st.sidebar.radio(
    "Research view",
    options=[
        "Offline Trial Replay",
        "EEG Signal Analysis",
        "Cross-Subject Benchmarks",
        "Measured Spike Activity",
        "Platform Guide",
    ],
    key="dashboard_research_view",
)
requires_replay = active_page in ("Offline Trial Replay", "Measured Spike Activity")

if requires_replay:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Replay Configuration")
    st.sidebar.caption(
        "Adjust the parameters, then apply them once. This prevents every slider "
        "movement from launching a complete replay."
    )
    with st.sidebar.form("replay_configuration_form"):
        momentum = st.slider(
            "Prototype Momentum (1 - α)",
            min_value=0.50,
            max_value=0.99,
            value=0.90,
            step=0.01,
            key="replay_momentum",
            help="Higher values adapt more smoothly; lower values follow distribution shifts more quickly.",
        )
        confidence_thresh = st.slider(
            "Confidence Threshold (τ)",
            min_value=0.25,
            max_value=0.95,
            value=0.60,
            step=0.05,
            key="replay_confidence_threshold",
            help="Minimum cosine softmax confidence required to update prototype positions.",
        )
        st.markdown("#### Sleep Consolidation")
        enable_consolidation = st.checkbox(
            "Enable Sleep Consolidation",
            value=True,
            key="replay_enable_consolidation",
            help="Uses synthetic noisy feature snapshots around Session 1 prototypes.",
        )
        cons_interval = st.number_input(
            "Interval (steps)", min_value=10, max_value=100, value=25, step=5,
            key="replay_consolidation_interval",
        )
        sleep_steps = st.number_input(
            "Correction Iterations", min_value=1, max_value=15, value=5,
            key="replay_sleep_steps",
        )
        pull_rate = st.slider(
            "Pull Strength (λ)", min_value=0.1, max_value=1.0,
            value=0.5, step=0.1, key="replay_pull_rate",
        )
        apply_replay_config = st.form_submit_button(
            "Apply replay settings", type="primary", width="stretch"
        )

else:
    momentum = 0.90
    confidence_thresh = 0.60
    enable_consolidation = True
    cons_interval = 25
    sleep_steps = 5
    pull_rate = 0.5


@st.cache_data(ttl=900, show_spinner="Loading and preprocessing EEG sessions...")
def load_signal_sessions(subject: int):
    """Load preprocessed EEG epochs in memory for signal inspection only."""
    return load_bnci2014_001(subjects=subject)


@st.cache_data(show_spinner=False)
def file_sha256(path: str, modified_ns: int, file_size: int) -> str:
    """Hash a model artifact once per file version, not on every widget rerun."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@st.cache_resource(show_spinner="Loading prepared replay features...")
def load_engine(
    c_path: str,
    p_path: str,
    s_id: int,
    proj: str,
    checkpoint_ref: str,
    prototypes_ref: str,
) -> TrialReplayEngine:
    artifact_dir = (
        PROJECT_ROOT / "results" / "dashboard_cache"
        / f"subject_{s_id}_{Path(c_path).stem}_{proj.lower()}"
    )
    if (artifact_dir / "metadata.json").is_file():
        try:
            return TrialReplayEngine.from_prepared_artifact(
                artifact_dir,
                checkpoint_ref=checkpoint_ref,
                subject=s_id,
                prototypes_ref=prototypes_ref,
            )
        except (OSError, ValueError, KeyError, ImportError, EOFError, json.JSONDecodeError) as error:
            st.warning(
                f"Prepared replay cache was rejected ({error}). "
                "Rebuilding from the source checkpoint and dataset."
            )
    return TrialReplayEngine.from_checkpoint(
        checkpoint_path=c_path,
        prototypes_path=p_path,
        subject=s_id,
        projection_method=proj,
    )


@st.cache_data(show_spinner="Simulating streaming trial adaptation...")
def run_cached_replay(
    _eng: TrialReplayEngine,
    replay_identity: str,
    mom: float,
    thresh: float,
    cons_on: bool,
    interval: int,
    steps: int,
    p_rate: float,
) -> ReplaySimulationData:
    return _eng.run_replay(
        momentum=mom,
        confidence_threshold=thresh,
        enable_consolidation=cons_on,
        consolidation_interval=interval,
        sleep_steps=steps,
        pull_rate=p_rate,
    )


engine = None
sim_data = None
measurement = None
checkpoint_data = {}
encoder_timesteps = None
encoder_channels = None
total_steps = 0

if requires_replay:
    if not ckpt_file.exists() or not proto_file.exists():
        st.sidebar.error(
            f"Checkpoint for Subject {subject_id} not found in results/checkpoints/"
        )
        st.stop()

    if active_page == "Measured Spike Activity":
        checkpoint_data = torch.load(
            ckpt_file, map_location="cpu", weights_only=False
        )
    prepared_cache_dir = (
        PROJECT_ROOT / "results" / "dashboard_cache"
        / f"subject_{subject_id}_{ckpt_file.stem}_{projection_type.lower()}"
    )
    if not (prepared_cache_dir / "metadata.json").is_file():
        st.sidebar.info(
            "No prepared feature cache exists for this selection. First load will "
            "preprocess EEG and extract SNN features. To precompute once, run "
            "`python scripts/precompute_dashboard_artifacts.py --subject "
            + str(subject_id) + " --projection " + projection_type + "`."
        )

    checkpoint_stat = ckpt_file.stat()
    prototype_stat = proto_file.stat()
    checkpoint_ref = file_sha256(
        str(ckpt_file), checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size
    )
    prototypes_ref = file_sha256(
        str(proto_file), prototype_stat.st_mtime_ns, prototype_stat.st_size
    )
    engine = load_engine(
        str(ckpt_file), str(proto_file), subject_id, projection_type,
        checkpoint_ref, prototypes_ref,
    )
    replay_identity = (
        f"{engine.checkpoint_ref}:{engine.prototypes_ref}:"
        f"subject-{subject_id}:projection-{projection_type}"
    )
    sim_data = run_cached_replay(
        engine,
        replay_identity,
        momentum,
        confidence_thresh,
        enable_consolidation,
        int(cons_interval),
        int(sleep_steps),
        pull_rate,
    )
    measurement = sim_data.measurement
    if measurement is None or measurement.checkpoint_ref != engine.checkpoint_ref:
        st.error(
            "Current simulation measurements are unavailable or do not match "
            "the selected checkpoint."
        )
        measurement = None

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Run Provenance")
    st.sidebar.caption(f"Checkpoint: `{ckpt_file}`")
    st.sidebar.caption(f"Checkpoint SHA-256: `{engine.checkpoint_ref}`")
    if artifact_file.exists():
        try:
            selected_artifact = load_evaluation_artifact(
                artifact_file, checkpoint_path=ckpt_file, subject=subject_id
            )
            st.sidebar.caption(
                f"Evaluation session: `{selected_artifact['evaluation_session']}`"
            )
            st.sidebar.caption(
                f"Protocol: `{selected_artifact['protocol']}`"
            )
            st.sidebar.caption(
                f"Evaluation run ID: `{selected_artifact['run_id']}`"
            )
            st.sidebar.caption(
                f"Artifact created: `{selected_artifact['created_at_utc']}`"
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            st.sidebar.warning(f"Evaluation provenance unavailable: {error}")
    if measurement is not None:
        st.sidebar.caption(f"Replay run ID: `{measurement.run_id}`")
        st.sidebar.caption(f"Measurement time: `{measurement.measured_at_utc}`")

    if active_page == "Measured Spike Activity":
        encoder_metadata = checkpoint_data.get("encoder_calibration", {})
        encoder_timesteps = encoder_metadata.get("n_timesteps")
        encoder_channels = encoder_metadata.get("channel_count")
        if encoder_timesteps is None or encoder_channels is None:
            st.error("Selected checkpoint lacks encoder timestep/channel metadata.")
    total_steps = len(sim_data.steps)

def _format_activity(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"

# Views are selected with a sidebar navigator so inactive pages are not computed.

# =============================================================================
# TAB 1: LIVE STREAMING REPLAY
# =============================================================================
if active_page == "Offline Trial Replay":
    # Playback Controls Bar
    if total_steps < 1:
        st.error(
            "The replay contains no Session 2 trials. Check the selected checkpoint, "
            "dataset download, and evaluation manifest."
        )
        st.stop()

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.4, 1.0, 1.0])
    slider_key = trial_slider_key(subject_id, projection_type, engine.checkpoint_ref)

    # Initialize/clamp the widget state before Streamlit creates the slider.
    if slider_key not in st.session_state:
        st.session_state[slider_key] = 1
    else:
        st.session_state[slider_key] = min(
            max(int(st.session_state[slider_key]), 1), total_steps
        )

    with ctrl_col2:
        st.write("")
        st.write("")
        if st.button("⏩ Jump to End", width="stretch", key="jump_replay_end"):
            st.session_state[slider_key] = total_steps

    with ctrl_col3:
        st.write("")
        st.write("")
        if st.button("⏮️ Jump to Start", width="stretch", key="jump_replay_start"):
            st.session_state[slider_key] = 1

    with ctrl_col1:
        current_step = st.slider(
            "Streaming Replay Position (Trial Index)",
            min_value=1,
            max_value=total_steps,
            key=slider_key,
            help="Move through the precomputed replay. This changes the displayed trial, not the underlying model run.",
        )

    current_step = min(max(int(current_step), 1), total_steps)
    current_record = sim_data.steps[current_step - 1]

    # Metrics Ribbon (Professional 5-Column Grid)
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("STREAMED TRIALS", f"{current_step} / {total_steps}")
    
    gain_so_far = current_record.cumulative_accuracy - current_record.static_baseline_accuracy
    m_col2.metric(
        "STREAMING ACCURACY",
        f"{current_record.cumulative_accuracy:.1%}",
        delta=f"{gain_so_far:+.1%} vs Static",
        delta_color="normal" if gain_so_far >= 0 else "inverse",
    )
    m_col3.metric(
        "TRIAL ADAPTATION",
        "ACCEPTED" if current_record.accepted else "FILTERED",
        delta=f"Conf: {current_record.confidence:.1%}",
        delta_color="normal" if current_record.accepted else "off",
    )
    m_col4.metric(
        "SLEEP CONSOLIDATION",
        "TRIGGERED 🌙" if current_record.sleep_consolidation_triggered else "MONITORING",
        delta=f"Drift: {current_record.drift_loss:.4f}" if current_record.sleep_consolidation_triggered else "Stable",
    )
    m_col5.metric(
        "NETWORK SPIKE ACTIVITY",
        _format_activity(
            measurement.aggregate_activity_fraction if measurement is not None else None
        ),
        delta="Measured fraction of spike opportunities",
        delta_color="off",
    )
    retention_delta = sim_data.session1_retention_accuracy - sim_data.session1_initial_accuracy
    m_col6.metric(
        "SESSION 1 TRAIN RETENTION",
        f"{sim_data.session1_retention_accuracy:.1%}",
        delta=f"{retention_delta:+.1%} vs initial S1 train",
        delta_color="normal" if retention_delta >= 0 else "inverse",
        help="In-sample Session 1 training-set retention after the complete replay. This is a diagnostic, not held-out source generalization.",
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2-Column Main Workspace
    plot_col, side_col = st.columns([1.6, 1.0])

    # Class color palette (ColorBrewer-inspired professional palette)
    class_palette = ["#2563EB", "#059669", "#D97706", "#7C3AED"]

    with plot_col:
        st.markdown(
            f"**📍 Live Feature-Space Manifold & Trajectories ({sim_data.projection_method})**"
        )
        st.caption("Faded dots = Session 1 baseline. Solid dots = Streamed Session 2 trials. Stars (★) = Live prototypes adapting over time.")

        fig_manifold = go.Figure()

        # 1. Background Session 1 clusters (faded reference)
        for c_idx, c_name in enumerate(sim_data.class_names):
            mask_s1 = sim_data.session1_labels == c_idx
            fig_manifold.add_trace(
                go.Scatter(
                    x=sim_data.session1_features_2d[mask_s1, 0],
                    y=sim_data.session1_features_2d[mask_s1, 1],
                    mode="markers",
                    marker=dict(size=5, opacity=0.18, color=class_palette[c_idx]),
                    name=f"S1 {c_name} (Baseline)",
                    legendgroup=f"class_{c_idx}",
                    showlegend=True,
                )
            )

        # 2. Session 2 trials streamed up to current_step
        s2_pts = sim_data.session2_features_2d[:current_step]
        s2_labels = sim_data.session2_labels[:current_step]
        for c_idx, c_name in enumerate(sim_data.class_names):
            mask_s2 = s2_labels == c_idx
            if np.any(mask_s2):
                fig_manifold.add_trace(
                    go.Scatter(
                        x=s2_pts[mask_s2, 0],
                        y=s2_pts[mask_s2, 1],
                        mode="markers",
                        marker=dict(size=8, opacity=0.85, color=class_palette[c_idx]),
                        name=f"S2 {c_name} (Streamed)",
                        legendgroup=f"class_{c_idx}",
                        showlegend=False,
                    )
                )

        # 3. Initial Session 1 Prototypes (hollow reference rings)
        fig_manifold.add_trace(
            go.Scatter(
                x=sim_data.initial_prototypes_2d[:, 0],
                y=sim_data.initial_prototypes_2d[:, 1],
                mode="markers+text",
                text=[f"P{i+1}_init" for i in range(len(sim_data.class_names))],
                textposition="top center",
                textfont=dict(size=10, color="#64748B"),
                marker=dict(symbol="circle-open", size=14, line=dict(width=2, color="#475569")),
                name="S1 Initial Prototypes",
            )
        )

        # 4. Trajectory Trails of Adapting Prototypes
        for c_idx in range(len(sim_data.class_names)):
            traj = sim_data.prototype_history_2d[:current_step, c_idx, :]
            fig_manifold.add_trace(
                go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode="lines",
                    line=dict(width=2.5, dash="dot", color=class_palette[c_idx]),
                    name=f"Trajectory: {sim_data.class_names[c_idx]}",
                    showlegend=False,
                )
            )

        # 5. Live Adapted Prototype Positions (Prominent Stars)
        current_protos_2d = current_record.prototypes_2d
        fig_manifold.add_trace(
            go.Scatter(
                x=current_protos_2d[:, 0],
                y=current_protos_2d[:, 1],
                mode="markers+text",
                text=[f"★ {sim_data.class_names[i]}" for i in range(len(sim_data.class_names))],
                textposition="bottom center",
                textfont=dict(size=12, color="#0F172A", family="Inter"),
                marker=dict(
                    symbol="star",
                    size=22,
                    color=class_palette,
                    line=dict(width=2, color="#FFFFFF"),
                ),
                name="Live Adapted Prototypes",
            )
        )

        # 6. Active Trial Marker
        fig_manifold.add_trace(
            go.Scatter(
                x=[current_record.feature_2d[0]],
                y=[current_record.feature_2d[1]],
                mode="markers+text",
                text=[f"Trial #{current_step}"],
                textposition="top right",
                textfont=dict(size=11, color="#0F172A", family="Inter"),
                marker=dict(
                    symbol="diamond",
                    size=16,
                    color="#10B981" if current_record.accepted else "#EF4444",
                    line=dict(width=2.5, color="#FFFFFF"),
                ),
                name="Current Active Trial",
            )
        )

        fig_manifold.update_layout(
            height=600,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(
                title=f"{sim_data.projection_method} Dimension 1",
                gridcolor="#F1F5F9",
                zerolinecolor="#E2E8F0",
            ),
            yaxis=dict(
                title=f"{sim_data.projection_method} Dimension 2",
                gridcolor="#F1F5F9",
                zerolinecolor="#E2E8F0",
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
        )
        st.plotly_chart(fig_manifold, width="stretch")

    with side_col:
        st.markdown("**📈 Online Adaptation vs. Static SNN Baseline**")

        # Running curves
        acc_history = [s.cumulative_accuracy for s in sim_data.steps[:current_step]]
        static_history = [s.static_baseline_accuracy for s in sim_data.steps[:current_step]]
        conf_history = [s.confidence for s in sim_data.steps[:current_step]]
        steps_x = list(range(1, current_step + 1))

        fig_dyn = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("Accuracy Comparison", "Confidence & Threshold"))

        fig_dyn.add_trace(
            go.Scatter(x=steps_x, y=acc_history, mode="lines", name="Adapted SNN", line=dict(color="#0284C7", width=2.5)),
            row=1, col=1,
        )
        fig_dyn.add_trace(
            go.Scatter(x=steps_x, y=static_history, mode="lines", name="Static Baseline (Unadapted)", line=dict(color="#94A3B8", width=1.8, dash="dot")),
            row=1, col=1,
        )

        fig_dyn.add_trace(
            go.Scatter(x=steps_x, y=conf_history, mode="lines", name="Confidence", line=dict(color="#F59E0B", width=1.5)),
            row=2, col=1,
        )
        fig_dyn.add_hline(
            y=confidence_thresh,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Threshold ({confidence_thresh:.2f})",
            annotation_position="bottom right",
            row=2, col=1,
        )

        fig_dyn.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=25, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.2, x=0),
        )
        fig_dyn.update_yaxes(gridcolor="#F1F5F9", row=1, col=1)
        fig_dyn.update_yaxes(gridcolor="#F1F5F9", row=2, col=1)
        fig_dyn.update_xaxes(gridcolor="#F1F5F9", row=2, col=1, title_text="Trial Index")

        st.plotly_chart(fig_dyn, width="stretch")

        st.markdown(
            f"""
            <div style="background-color: rgba(255, 255, 255, 0.04); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 8px; padding: 14px; font-size: 0.90rem;">
                <div style="font-weight: 700; margin-bottom: 8px;">🔍 Active Trial #{current_step} State</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="opacity: 0.7;">True Motor Imagery:</span>
                    <span style="font-weight: 600;">{sim_data.class_names[current_record.true_label]}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="opacity: 0.7;">SNN Prediction:</span>
                    <span style="font-weight: 600; color: {'#10B981' if current_record.predicted_label == current_record.true_label else '#EF4444'};">
                        {sim_data.class_names[current_record.predicted_label]}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="opacity: 0.7;">Softmax Confidence:</span>
                    <span style="font-weight: 600;">{current_record.confidence:.2%}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="opacity: 0.7;">Decision:</span>
                    <span style="font-weight: 600; color: {'#10B981' if current_record.accepted else '#94A3B8'};">
                        {'ADAPTED (P ← EMA)' if current_record.accepted else 'FILTERED (Below Threshold)'}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="opacity: 0.7;">Accepted So Far:</span>
                    <span style="font-weight: 600;">{sum(1 for s in sim_data.steps[:current_step] if s.accepted)} / {current_step}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # Trial-level diagnostics derived from the same prequential replay records.
    st.markdown('#### Trial-level classification diagnostics')
    diagnostic_col1, diagnostic_col2 = st.columns(2)
    replay_truth = np.asarray([step.true_label for step in sim_data.steps[:current_step]], dtype=int)
    replay_predictions = np.asarray([step.predicted_label for step in sim_data.steps[:current_step]], dtype=int)
    n_classes = len(sim_data.class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for truth, prediction in zip(replay_truth, replay_predictions):
        if 0 <= truth < n_classes and 0 <= prediction < n_classes:
            confusion[truth, prediction] += 1
    with diagnostic_col1:
        confusion_fig = go.Figure(go.Heatmap(
            z=confusion, x=sim_data.class_names, y=sim_data.class_names,
            colorscale='Blues', showscale=True,
            text=confusion, texttemplate='%{text}',
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Trials: %{z}<extra></extra>',
        ))
        confusion_fig.update_layout(title='Prequential confusion matrix',
            xaxis_title='Predicted class', yaxis_title='True class',
            height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(confusion_fig, width='stretch')
    with diagnostic_col2:
        confidence_fig = go.Figure()
        for class_index, class_name in enumerate(sim_data.class_names):
            values = [step.confidence for step in sim_data.steps[:current_step]
                      if step.true_label == class_index]
            if values:
                confidence_fig.add_trace(go.Box(y=values, name=class_name,
                    boxpoints='outliers', marker_size=4))
        confidence_fig.add_hline(y=confidence_thresh, line_dash='dash',
            line_color='#D97706', annotation_text='Acceptance threshold')
        confidence_fig.update_layout(title='Confidence by true class',
            yaxis_title='Model confidence', yaxis_range=[0, 1],
            height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(confidence_fig, width='stretch')
# =============================================================================
# TAB 2: CROSS-SUBJECT BENCHMARKS (TABLE 5)
# =============================================================================
if active_page == "Cross-Subject Benchmarks":
    st.markdown("### 📊 Cross-Subject Experimental Results (Table 5)")
    st.caption("Metrics are loaded from hash-validated held-out evaluation artifacts; synthetic smoke-test outputs are not benchmark evidence.")

    table_data = []
    heatmap_records: list[dict[str, object]] = []
    for s in range(1, 10):
        subject_manifest_runs = manifest_runs_by_subject.get(s, [])
        if subject_manifest_runs:
            run_records = subject_manifest_runs
        else:
            run_records = [{
                "checkpoint": str(CHECKPOINT_DIR / f"phase_a_subject_{s}.pt"),
                "evaluation_artifact": (
                    artifact_file if s == subject_id
                    else CHECKPOINT_DIR / f"evaluation_subject_{s}.json"
                ),
                "seed": "legacy",
            }]
        metric_values: dict[str, list[float]] = {
            "target_static_baseline_acc": [],
            "target_prequential_acc": [],
            "target_retrospective_acc": [],
            "source_accuracy_after_adaptation": [],
        }
        checkpoint_data_rows = []
        valid_seeds = []
        for run_record in run_records:
            c_p = resolve_repo_path(PROJECT_ROOT, run_record["checkpoint"])
            artifact_path = resolve_repo_path(PROJECT_ROOT, run_record["evaluation_artifact"])
            if not c_p.exists() or not artifact_path.exists():
                continue
            try:
                data = torch.load(c_p, map_location="cpu", weights_only=False)
                evaluation_artifact = load_evaluation_artifact(
                    artifact_path, checkpoint_path=c_p, subject=s
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                st.error(f"Subject S{s:02d} run {run_record.get('seed')} rejected: {error}")
                continue
            required_contract = (
                "label_values",
                "label_names",
                "encoder_calibration",
                "preprocessing_contract",
            )
            missing_contract = [key for key in required_contract if not data.get(key)]
            if missing_contract:
                continue
            if len(data["label_values"]) != len(data["label_names"]):
                continue
            metrics = evaluation_artifact["metrics"]
            for name in metric_values:
                metric_values[name].append(float(metrics[name]))
            heatmap_records.append({
                "Subject": f"S{s:02d}",
                "Seed": str(run_record.get("seed", "unknown")),
                **{name: float(metrics[name]) for name in metric_values},
            })
            checkpoint_data_rows.append(data)
            valid_seeds.append(str(run_record.get("seed")))
        if not checkpoint_data_rows:
            st.info(f"Subject S{s:02d}: no valid held-out evaluation runs are available.")
            continue
        def summary(name: str) -> str:
            values = metric_values[name]
            mean = float(np.mean(values))
            spread = float(np.std(values))
            return f"{mean * 100:.2f}% ± {spread * 100:.2f}%"
        first_data = checkpoint_data_rows[0]
        table_data.append({
            "Subject": f"Subject S{s:02d}",
            "Seeds": ", ".join(valid_seeds),
            "Phase A Train": f"{np.mean([d.get('train_accuracy', 0) for d in checkpoint_data_rows]) * 100:.2f}%",
            "Phase A Val": f"{np.mean([d.get('validation_accuracy', 0) for d in checkpoint_data_rows]) * 100:.2f}%",
            "Static Target Baseline": summary("target_static_baseline_acc"),
            "Prequential Target Accuracy": summary("target_prequential_acc"),
            "Retrospective Diagnostic": summary("target_retrospective_acc"),
            "Source Accuracy After": summary("source_accuracy_after_adaptation"),
            "Feature Dim": first_data["feature_dim"],
            "Classes": len(first_data["label_values"]),
        })

    df_results = pd.DataFrame(table_data)
    st.dataframe(df_results, width="stretch")

    if heatmap_records:
        metric_labels = {
            "target_static_baseline_acc": "Static target baseline",
            "target_prequential_acc": "Prequential target accuracy",
            "target_retrospective_acc": "Retrospective diagnostic",
            "source_accuracy_after_adaptation": "Source retention",
        }
        selected_heatmap_metric = st.selectbox(
            "Heatmap metric", options=list(metric_labels),
            format_func=lambda key: metric_labels[key],
            key="benchmark_heatmap_metric",
        )
        heatmap_frame = pd.DataFrame(heatmap_records).pivot_table(
            index="Subject", columns="Seed", values=selected_heatmap_metric,
            aggfunc="mean",
        ).sort_index()
        heatmap_fig = go.Figure(go.Heatmap(
            z=heatmap_frame.to_numpy() * 100,
            x=[str(value) for value in heatmap_frame.columns],
            y=heatmap_frame.index.tolist(), colorscale="Blues",
            colorbar=dict(title="Accuracy (%)"),
            hovertemplate="Subject: %{y}<br>Seed: %{x}<br>Score: %{z:.2f}%<extra></extra>",
        ))
        heatmap_fig.update_layout(
            title=f"Subject × seed — {metric_labels[selected_heatmap_metric]}",
            xaxis_title="Seed", yaxis_title="Subject",
            height=max(300, 42 * len(heatmap_frame.index) + 100),
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(heatmap_fig, width="stretch")
    else:
        st.info("No validated subject/seed evaluation artifacts are available for the heatmap.")

    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        # Grouped Bar: SNN Prototype vs Linear Baseline
        fig_acc = go.Figure()
        test_rows = [d for d in table_data if d["Prequential Target Accuracy"] != "N/A"]
        sub_labels = [d["Subject"] for d in test_rows]
        snn_accs = [float(d["Prequential Target Accuracy"].split(" ±")[0].replace("%", "")) for d in test_rows]
        static_accs = [float(d["Static Target Baseline"].split(" ±")[0].replace("%", "")) for d in test_rows]

        fig_acc.add_trace(go.Bar(x=sub_labels, y=static_accs, name="Static Target Baseline", marker_color="#CBD5E1"))
        fig_acc.add_trace(go.Bar(x=sub_labels, y=snn_accs, name="Prequential Adaptation", marker_color="#0284C7"))

        fig_acc.update_layout(
            title="Held-Out Target Baseline vs Prequential Adaptation",
            yaxis_title="Accuracy (%)",
            barmode="group",
            height=380,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#F1F5F9"),
            legend=dict(orientation="h", y=1.1, x=0),
        )
        st.plotly_chart(fig_acc, width="stretch")

    with chart_c2:
        # Train vs Val vs Test Generalization Gap
        fig_gap = go.Figure()
        train_accs = [float(d["Phase A Train"].replace("%", "")) for d in test_rows]
        val_accs = [float(d["Phase A Val"].replace("%", "")) for d in test_rows]

        fig_gap.add_trace(go.Scatter(x=sub_labels, y=train_accs, mode="lines+markers", name="Train Accuracy", line=dict(color="#10B981", width=2)))
        fig_gap.add_trace(go.Scatter(x=sub_labels, y=val_accs, mode="lines+markers", name="Val Accuracy", line=dict(color="#F59E0B", width=2)))
        fig_gap.add_trace(go.Scatter(x=sub_labels, y=snn_accs, mode="lines+markers", name="Prequential Target", line=dict(color="#0284C7", width=2)))

        fig_gap.update_layout(
            title="Generalization Profile Across Train / Validation / Test",
            yaxis_title="Accuracy (%)",
            height=380,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#F1F5F9"),
            legend=dict(orientation="h", y=1.1, x=0),
        )
        st.plotly_chart(fig_gap, width="stretch")


# =============================================================================
# TAB 3: NEUROMORPHIC ENERGY & SPARSITY
# =============================================================================
if active_page == "Measured Spike Activity":
    st.markdown("### ⚡ Measured Spike Activity")
    st.caption("Offline, trial-by-trial replay using zero-phase epoch preprocessing; not a real-time BCI or direct energy measurement.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            #### Real Empirical Measurements
            - **Input Spike Encoding:** {encoder_channels or 'N/A'} EEG channels × 3 configured frequency bands = **{encoder_channels * 3 if encoder_channels else 'N/A'} Virtual Channels** over **{encoder_timesteps or 'N/A'} Timesteps**.
            - **Spike Encoder Activity:** Alpha/Beta rate coding + Gamma TTFS measured on replay input.
            - **LIF layer activity fraction:** **{_format_activity(measurement.layer_activity_fraction.get('lif1') if measurement else None)}**.
            - **LIF layer activity fraction:** **{_format_activity(measurement.layer_activity_fraction.get('lif2') if measurement else None)}**.
            - **Aggregate activity fraction:** **{_format_activity(measurement.aggregate_activity_fraction if measurement else None)}**

            ---
            
            #### Interpretation
            These values report observed spike activity fractions for this
            offline trial-by-trial replay only. Hardware energy and savings require a
            defined device, workload, measurement protocol, and baseline.
            """
        )

    with c2:
        active_fraction = measurement.aggregate_activity_fraction if measurement else None
        active_pct = active_fraction * 100 if active_fraction is not None else 0
        silent_pct = 100.0 - active_pct if active_fraction is not None else 0

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Silent Neuron Timesteps", "Active Spiking Neuron Timesteps"],
                    values=[silent_pct, active_pct] if active_fraction is not None else [],
                    hole=0.60,
                    marker_colors=["#10B981", "#0284C7"],
                    textinfo="label+percent",
                    textfont=dict(size=12, family="Inter"),
                )
            ]
        )
        fig_donut.update_layout(
            title="Neuron Duty Cycle Distribution",
            height=340,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, width="stretch")

        # Layer-by-layer sparsity comparison bar
        layer_names = []
        layer_vals = []
        if measurement is not None:
            layer_names.append("Input activity")
            layer_vals.append(measurement.input_activity_fraction * 100)
            for layer_name, value in measurement.layer_activity_fraction.items():
                if np.isfinite(value):
                    layer_names.append(layer_name)
                    layer_vals.append(value * 100)
            layer_names.append("Aggregate activity")
            layer_vals.append(measurement.aggregate_activity_fraction * 100)

        fig_layers = go.Figure(
            data=[
                go.Bar(
                    x=layer_names,
                    y=layer_vals,
                    marker_color=["#6366F1", "#3B82F6", "#0284C7", "#10B981"],
                    text=[f"{v:.1f}%" for v in layer_vals],
                    textposition="auto",
                )
            ]
        )
        fig_layers.update_layout(
            title="Layer-by-Layer Mean Firing Rates",
            yaxis_title="Firing Rate (%)",
            height=260,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#F1F5F9", range=[0, 45]),
        )
        if layer_vals:
            st.plotly_chart(fig_layers, width="stretch")
        else:
            st.info("No valid spike-activity measurements are available for this run.")

    st.markdown("---")
    st.markdown("### Paired Sleep-Consolidation Ablation")
    st.caption(
        "Runs the same checkpoint, subject, trial order, momentum and confidence "
        "threshold with consolidation ON and OFF. The ON run uses deterministic "
        "synthetic-noise seeding derived from the checkpoint run ID."
    )
    run_ablation = st.checkbox(
        "Compute paired consolidation ON/OFF comparison",
        value=False,
        key="run_consolidation_ablation",
    )
    if run_ablation:
        ablation_identity = (
            f"{engine.checkpoint_ref}:subject-{subject_id}:"
            f"projection-{projection_type}:paired-ablation"
        )
        ablation_on = run_cached_replay(
            engine, f"{ablation_identity}:on", momentum, confidence_thresh,
            True, int(cons_interval), int(sleep_steps), pull_rate,
        )
        ablation_off = run_cached_replay(
            engine, f"{ablation_identity}:off", momentum, confidence_thresh,
            False, int(cons_interval), int(sleep_steps), pull_rate,
        )
        ablation_rows = [
            {
                "Condition": "Consolidation OFF",
                "Target prequential accuracy": ablation_off.adapted_target_accuracy * 100,
                "Session 1 train retention": ablation_off.session1_retention_accuracy * 100,
                "Session 1 forgetting": (
                    ablation_off.session1_initial_accuracy
                    - ablation_off.session1_retention_accuracy
                ) * 100,
            },
            {
                "Condition": "Consolidation ON",
                "Target prequential accuracy": ablation_on.adapted_target_accuracy * 100,
                "Session 1 train retention": ablation_on.session1_retention_accuracy * 100,
                "Session 1 forgetting": (
                    ablation_on.session1_initial_accuracy
                    - ablation_on.session1_retention_accuracy
                ) * 100,
            },
        ]
        ablation_frame = pd.DataFrame(ablation_rows)
        st.dataframe(
            ablation_frame.style.format({
                "Target prequential accuracy": "{:.2f}%",
                "Session 1 train retention": "{:.2f}%",
                "Session 1 forgetting": "{:.2f} pp",
            }),
            width="stretch",
        )
        ablation_fig = go.Figure()
        for metric_name in (
            "Target prequential accuracy",
            "Session 1 train retention",
        ):
            ablation_fig.add_trace(go.Bar(
                x=ablation_frame["Condition"],
                y=ablation_frame[metric_name],
                name=metric_name,
            ))
        ablation_fig.update_layout(
            title="Paired consolidation ablation",
            yaxis_title="Accuracy (%)",
            barmode="group",
            height=380,
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(ablation_fig, width="stretch")
        st.caption(
            "Session 1 retention is measured on the Session 1 training features "
            "used by this replay engine. It is an in-sample diagnostic, not "
            "held-out source-session generalization. One subject/checkpoint run "
            "is not sufficient evidence for a general consolidation claim."
        )


# =============================================================================
# EEG SIGNAL ANALYSIS
# =============================================================================
if active_page == "EEG Signal Analysis":
    st.markdown("### EEG Signal Analysis")
    st.caption(
        "Inspect preprocessed BNCI2014-001 trial epochs. These are offline, "
        "zero-phase filtered and demeaned epochs—not live EEG and not raw continuous recordings."
    )
    try:
        signal_sessions = load_signal_sessions(subject_id)
    except Exception as error:
        st.error(
            "Could not load the EEG sessions. Check the MOABB dataset download, "
            "MNE cache location, and network access."
        )
        st.exception(error)
        st.stop()

    session_names = list(signal_sessions.keys())
    if not session_names:
        st.warning("No EEG sessions were returned for this subject.")
        st.stop()

    selected_session_name = st.selectbox(
        "EEG session",
        options=session_names,
        format_func=lambda value: value,
        key=f"signal_session_{subject_id}",
    )
    signal_session = signal_sessions[selected_session_name]
    n_trials, n_channels, n_samples = signal_session.X.shape
    if n_trials < 1 or n_channels < 1 or n_samples < 2:
        st.error("The selected session contains no usable EEG epochs.")
        st.stop()

    signal_trial_key = f"signal_trial_{subject_id}_{selected_session_name}"
    if signal_trial_key not in st.session_state:
        st.session_state[signal_trial_key] = 1
    else:
        st.session_state[signal_trial_key] = min(
            max(int(st.session_state[signal_trial_key]), 1), n_trials
        )
    selected_trial = st.slider(
        "Trial",
        min_value=1,
        max_value=n_trials,
        key=signal_trial_key,
    )
    trial_index = selected_trial - 1
    trial = np.asarray(signal_session.X[trial_index], dtype=np.float32)
    trial_metadata = (
        signal_session.metadata[trial_index]
        if trial_index < len(signal_session.metadata) else {}
    )
    event_label = str(trial_metadata.get("event", "Unknown"))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Trials", f"{n_trials:,}")
    metric_cols[1].metric("EEG channels", str(n_channels))
    metric_cols[2].metric("Sampling rate", f"{signal_session.sfreq:.1f} Hz")
    metric_cols[3].metric("Event class", event_label.replace("_", " ").title())

    channel_names = list(signal_session.channel_names)
    if len(channel_names) != n_channels:
        channel_names = [f"EEG {index + 1}" for index in range(n_channels)]
    selected_channel = st.selectbox(
        "Channel",
        options=channel_names,
        key=f"signal_channel_{subject_id}_{selected_session_name}",
    )
    channel_index = channel_names.index(selected_channel)
    time_seconds = np.arange(n_samples, dtype=np.float64) / signal_session.sfreq

    waveform_fig = go.Figure()
    waveform_fig.add_trace(go.Scatter(
        x=time_seconds,
        y=trial[channel_index] * 1e6,
        mode="lines",
        name=selected_channel,
        line=dict(width=1.2),
    ))
    waveform_fig.update_layout(
        title=f"Preprocessed EEG waveform — {selected_channel} — Trial {selected_trial}",
        xaxis_title="Time relative to imagery cue (s)",
        yaxis_title="Amplitude (µV)",
        height=390,
        margin=dict(l=15, r=15, t=55, b=15),
    )
    st.plotly_chart(waveform_fig, width="stretch")

    from scipy.signal import welch

    frequencies, psd = welch(
        trial,
        fs=signal_session.sfreq,
        axis=-1,
        nperseg=min(256, n_samples),
        scaling="density",
    )
    selected_psd = psd[channel_index] * 1e12
    psd_fig = go.Figure(go.Scatter(
        x=frequencies,
        y=selected_psd,
        mode="lines",
        name="Power spectral density",
        line=dict(width=1.5),
    ))
    psd_fig.update_layout(
        title=f"Power spectral density — {selected_channel}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD (µV²/Hz)",
        xaxis=dict(range=[0, min(100, signal_session.sfreq / 2)]),
        height=350,
        margin=dict(l=15, r=15, t=55, b=15),
    )
    st.plotly_chart(psd_fig, width="stretch")

    frequency_bands = {
        "Alpha 8–13 Hz": (8.0, 13.0),
        "Beta 13–30 Hz": (13.0, 30.0),
        "Gamma 30–100 Hz": (30.0, min(100.0, signal_session.sfreq / 2 - 1e-6)),
    }
    band_power_rows = []
    band_power_matrix = []
    valid_band_names = []
    for band_name, (low_hz, high_hz) in frequency_bands.items():
        mask = (frequencies >= low_hz) & (frequencies < high_hz)
        if int(mask.sum()) < 2:
            continue
        power = np.trapz(psd[:, mask], frequencies[mask], axis=1) * 1e12
        band_power_matrix.append(power)
        valid_band_names.append(band_name)
        band_power_rows.append({
            "Band": band_name,
            "Selected-channel power (µV²)": float(power[channel_index]),
        })

    if band_power_matrix:
        band_power = np.column_stack(band_power_matrix)
        heatmap_values = 10.0 * np.log10(np.maximum(band_power, np.finfo(float).tiny))
        heatmap_fig = go.Figure(go.Heatmap(
            z=heatmap_values,
            x=valid_band_names,
            y=channel_names,
            colorbar=dict(title="dB re µV²"),
            hovertemplate="Channel: %{y}<br>Band: %{x}<br>Power: %{z:.2f} dB re µV²<extra></extra>",
        ))
        heatmap_fig.update_layout(
            title=f"Band-power distribution across EEG channels — Trial {selected_trial}",
            xaxis_title="Frequency band",
            yaxis_title="EEG channel",
            height=max(420, min(760, 22 * n_channels + 180)),
            margin=dict(l=15, r=15, t=55, b=15),
        )
        st.plotly_chart(heatmap_fig, width="stretch")
        st.dataframe(pd.DataFrame(band_power_rows), hide_index=True, width="stretch")

    st.caption(
        f"Preprocessing: {signal_session.l_freq:g}–{signal_session.h_freq:g} Hz "
        f"zero-phase Butterworth bandpass; {signal_session.tmin:g}–"
        f"{signal_session.tmax:g} s epoch window relative to trial start; "
        f"per-trial channel demeaning. The PSD and band-power values describe "
        f"this processed epoch and are not source-session or model-accuracy metrics."
    )


# =============================================================================
# TAB 5: DASHBOARD GUIDE & ARCHITECTURE
# =============================================================================
if active_page == "Platform Guide":
    st.markdown("### 📖 Architecture & Experimental Protocol Guide")
    st.markdown(
        """
        #### 1. Why Fixed 2D Manifold Reference?
        When visualizing high-dimensional feature embeddings across sequential frames, re-fitting algorithms like UMAP or t-SNE causes stochastic coordinate jumping (frame jitter).
        To provide a reliable, professional visualization:
        - We compute a fixed projection basis $W_{\\text{proj}} = \\text{fit}(F_{\\text{Session 1}})$.
        - All incoming Session 2 trials $f_t$ and live adapting prototypes $P_t$ are stably projected into this invariant coordinate system ($z_t = W_{\\text{proj}} f_t$).
        - Thus, prototype movement represents genuine physiological adaptation drift rather than manifold artifacts.

        ---

        #### 2. Phase B Prototype Adaptation (EMA)
        Unlike conventional models that retrain backpropagation weights online (which is computationally prohibitive on low-power devices), NeuroStream leaves SNN synaptic weights **strictly frozen**:
        $$P_c^{(t)} = \\mu P_c^{(t-1)} + (1 - \\mu) f_t$$
        where $\\mu = 1 - \\alpha$ represents the momentum inertia and $f_t \\in \\mathbb{R}^{128}$ is the continuous feature vector from the LIF layer.

        ---

        #### 3. Privacy-Preserving Sleep Consolidation
        To prevent catastrophic forgetting of Session 1 without violating the **zero raw data storage** privacy guarantee:
        - At adaptation intervals (e.g. every 25 accepted trials), the system generates synthetic features:
          $$\\tilde{P}_c = P_{\\text{snapshot}, c} + \\mathcal{N}(0, \\sigma^2 I), \\quad \\sigma = 0.05$$
        - A gradient-free EMA pull attracts drifted prototypes back toward this snapshot:
          $$P_c \\leftarrow (1 - \\lambda) P_c + \\lambda P_{\\text{snapshot}, c}$$
        - This preserves long-term retention without retaining a single raw EEG signal.
        """
    )
