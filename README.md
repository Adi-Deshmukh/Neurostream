# NeuroStream: Adaptive Spiking Neural Network BCI Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroStream** is an event-driven, privacy-preserving Brain-Computer Interface (BCI) framework designed for non-invasive motor-imagery decoding. It combines a custom **Leaky Integrate-and-Fire (LIF)** spiking neural network with **online Exponential Moving Average (EMA) prototype adaptation** and **privacy-preserving sleep-phase consolidation**.

---

## 🚀 Key Architectural Contributions

1. **Zero Raw-Data Storage Privacy**: Once Session 1 training completes, raw EEG signals are permanently discarded. During Session 2 replay, the implementation can generate synthetic noisy feature vectors ($\tilde{P} = P_{\text{snapshot}} + \mathcal{N}(0, \sigma^2 I)$) around stored prototype snapshots. This mechanism requires paired ablation results before making claims about preventing catastrophic forgetting.
2. **Measured Spike Activity**: Forward hooks report LIF firing rates and neuron-timestep activity for the configured model and workload. These measurements are not direct hardware-energy measurements.
3. **Fixed-Manifold 2D Reference**: High-dimensional feature trajectories are stably projected into an invariant Session 1 basis (PCA/UMAP), eliminating coordinate jitter during live streaming adaptation.
4. **Frozen Synaptic Weights**: Synaptic weights $\Theta$ are frozen after Phase A, eliminating computationally expensive online backpropagation on edge devices.

---

## 🖥️ Interactive Evaluation Platform & Dashboard

The project includes an interactive research dashboard built with Streamlit and Plotly, exposing offline trial-by-trial replay, prototype movements, and empirical multi-subject benchmarks. It is not a causal real-time BCI.

### Launching the Dashboard

Run these commands from the repository root (the folder containing `pyproject.toml`):

```powershell
python -m pip install -e ".[dashboard]"
python -m streamlit run dashboard/app.py
```

Use the top-level `dashboard/app.py` launcher. Do **not** run
`python -m streamlit run src/neurostream/dashboard/app.py`, and do not import
`dashboard.simulation`. The Python package path is
`neurostream.dashboard.simulation`. For example:

```powershell
python -c "from neurostream.dashboard.simulation import TrialReplayEngine; print('NeuroStream dashboard import OK')"
```

Then navigate to `http://localhost:8501` in your browser.

The dashboard is an artifact-first research workstation with seven views:

- **Overview:** selected experiment metrics, subject-wise comparisons, recent validated runs and checkpoint provenance.
- **Signal Analysis:** explicit, on-demand loading of preprocessed EEG epochs; channel/trial/window selection, session comparison, band-power heatmaps and epoch quality checks. It does not label preprocessed epochs as raw EEG.
- **Model Diagnostics:** model architecture, validated confusion matrix, measured layer activity when present in a matching prepared cache, and feature-space distributions loaded only on request.
- **Adaptation Replay:** offline trial-by-trial replay from feature-only prepared artifacts. Settings are applied once; the trial slider and jump controls only navigate stored replay output.
- **Benchmark Lab:** subject × seed heatmaps, metric comparisons, per-subject variability and confusion matrices from validated evaluation artifacts.
- **Consolidation Study:** paired ON/OFF analysis joined on exact subject and seed, with paired differences and uncertainty intervals. No conclusion is shown when a paired report is absent.
- **Experiment Registry:** manifest entries, checkpoint/evaluation validation status, hashes, run metadata and prepared-cache inventory.

The overview, benchmark and registry views do not instantiate the SNN, load EEG, deserialize checkpoints, fit projections, or execute replay. The registry reads local JSON manifests and validates evaluation artifacts against the SHA-256 of the referenced checkpoint. Missing, stale, malformed or mismatched artifacts are reported as unavailable rather than represented as zero.

### Prepare replay features once

Replay requires a feature-only prepared artifact matching the selected checkpoint and prototype files. For a run in a sweep directory, pass the exact checkpoint and prototype paths:

```powershell
python scripts/precompute_dashboard_artifacts.py --subject 1 --checkpoint "results/checkpoints/sweep/sub_1/seed_42/phase_a_subject_1.pt" --prototypes "results/checkpoints/sweep/sub_1/seed_42/prototypes_subject_1.pt" --projection PCA
```

The output is written under `results/dashboard_cache/`. The app loads it only when subject, checkpoint SHA-256 and prototype SHA-256 match. If it is absent or rejected, the replay page shows the required preparation command; it does not silently run EEG preprocessing or SNN feature extraction on page navigation.

The preparation command stores feature matrices, labels, prototypes, projection coordinates/state, measured spike activity and provenance. It does not store raw EEG epochs or encoded spike tensors. Signal Analysis is a separate explicit operation that loads the dataset into the current Streamlit session.

### Preprocessing and calibration contract

The current pipeline is **offline and epoch-based**. Preprocessing uses a
zero-phase `sosfiltfilt` band-pass, so each sample may depend on future samples
within its epoch; it is not a causal sample-by-sample streaming filter and has
no streaming latency guarantee. A causal filter with carried state must be
implemented and validated before using this path for live raw-sample inference.

Phase A fits rate-encoding thresholds only on Session 1 training data. Those
thresholds, the true sampling frequency, filter and epoch settings, channel
order, normalization, frequency bands, label mapping, and timestep count are
stored in the checkpoint. Validation, held-out evaluation, dashboard replay,
and Phase B load that contract and apply it unchanged. Mismatched sampling
rates or channel contracts are rejected rather than silently using the 250 Hz
default.

For BNCI2014-001, MOABB's class trigger denotes trial start. The cue appears
2 seconds after that trigger and imagery ends at 6 seconds, so the default
epoch window `[2.0, 6.0]` is trial-start-relative and corresponds to `[0.0,
4.0]` relative to cue onset. The selected reference, cue offset, and window are
recorded in checkpoint and evaluation provenance; these values must be changed
together if a different event convention is used.

---

## 📖 Dashboard page guide

### Overview
Choose a validated subject/seed run from the sidebar. The metric ribbon distinguishes
static target baseline, prequential target accuracy, retrospective target diagnostic,
source-session retention, adaptation gain and measured spike activity. Subject-wise
charts aggregate only the available validated runs. The recent-runs table and
provenance panel expose run IDs, timestamps and checkpoint hashes.

### Signal Analysis
EEG loading is an explicit action because MOABB may need to access a local dataset
cache or download the dataset. Select session, trial, channel and time window to
inspect an epoch waveform. The band-power heatmap integrates Welch PSD over alpha,
beta and gamma bands. Session 1/Session 2 waveforms are illustrative same-index
examples, not paired trials. Quality checks report non-finite samples, flat channels,
mean absolute amplitude and peak-to-peak range. The loader returns preprocessed
epochs; raw continuous EEG and intermediate filter/normalization stages are not
available from this interface.

### Model Diagnostics
The architecture summary reflects the configured SNN widths. The stored prequential
confusion matrix is shown when present, with class-wise precision, recall and F1
derived from that matrix. Layer activity and Session 1/Session 2 feature-space
distributions are read from a matching prepared cache only when available. No
per-neuron activity or spike raster is fabricated when instrumentation did not
persist it.

### Adaptation Replay
Select a validated experiment, projection and prepared feature cache. If the cache
is absent, the page prints a preparation command with the selected checkpoint and
prototype paths. After loading, set replay parameters in the form and submit once.
The trial slider and start/end buttons scrub the computed offline replay; they do
not rerun the model. The view includes prequential/static accuracy progression,
confidence, accepted/rejected decisions, projected prototype trajectories,
2D projected prototype displacement and consolidation drift values recorded by
the replay engine. The schema does not currently preserve per-trial
high-dimensional prototype history.

### Benchmark Lab
Filter by subject and seed, inspect a subject × seed heatmap for the selected metric,
compare static/prequential/retrospective target accuracy, and inspect subject-wise
and seed-wise variability. Confusion matrices and class-wise metrics appear only
when the selected evaluation artifact contains the underlying predictions.

### Consolidation Study
The page requires a paired report generated by the existing ON/OFF ablation. Only
exact subject/seed matches are compared. It reports paired mean differences and
95% t intervals when there are enough pairs, plus recorded consolidation phase
counts. It does not claim that consolidation improves retention unless the paired
results support that claim. Prototype drift immediately before/after each
consolidation event is not present in the current ablation report and is therefore
not plotted.

### Experiment Registry
The registry lists manifest entries, checkpoint/evaluation paths, validation
status, errors, checkpoint hashes, dataset/session identity, protocol, label
mapping, run configuration and prepared-cache inventory. Refresh the registry
after adding or regenerating local experiment artifacts.

### Scientific interpretation rules

- Prequential accuracy is the prediction made before the current trial updates
  prototype memory.
- Retrospective target accuracy is a separate diagnostic after replay; it is not
  substituted for prequential accuracy.
- Source retention is reported separately from target-session accuracy.
- Spike activity is a measured fraction of spike opportunities, not power,
  energy consumption or energy savings.
- Offline replay is not real-time EEG acquisition or causal sample-by-sample
  inference.
- Missing artifacts and missing metrics remain unavailable; they are not zero.

## 🧪 Testing & Verification

Install the project and run the complete suite from the repository root:

```powershell
python -m pip install -e ".[dashboard]"
python -m pytest -q
python -m compileall -q src tests scripts
```

The suite covers model layers, spike encoding, training/adaptation, evaluation
artifacts, replay behavior, dashboard registry discovery, paired ablation joins
and dashboard source contracts. The GitHub Actions workflow also runs the test
suite, compiles Python sources and smoke-tests dashboard dependencies. A passing
CI run does not replace a local browser walkthrough with the user's actual
checkpoints and prepared caches.

**Dataset storage:** raw MOABB/MNE downloads are local machine caches and are excluded from Git under `data/mne_data/`. If you previously relied on the repository-tracked copy, MOABB may need to download the BNCI2014-001 files again or be pointed at your existing local MNE data directory.
