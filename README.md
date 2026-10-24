# NeuroStream: Adaptive Spiking Neural Network BCI Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroStream** is an event-driven, privacy-preserving Brain-Computer Interface (BCI) framework designed for non-invasive motor-imagery decoding. It combines a custom **Leaky Integrate-and-Fire (LIF)** spiking neural network with **online Exponential Moving Average (EMA) prototype adaptation** and **privacy-preserving sleep-phase consolidation**.

---

## 🚀 Key Architectural Contributions

1. **Zero Raw-Data Storage Privacy**: Once Session 1 training completes, raw EEG signals are permanently discarded. During Session 2 streaming, catastrophic forgetting is mitigated by generating synthetic noisy feature vectors ($\tilde{P} = P_{\text{snapshot}} + \mathcal{N}(0, \sigma^2 I)$) around stored prototype snapshots.
2. **Orders-of-Magnitude Neuromorphic Efficiency**: Evaluated via forward hooks on LIF layers. Neurons remain silent during **>92% of operational timesteps**, demanding zero dynamic power on event-driven neuromorphic processors (Intel Loihi 2, SynSense Xylo).
3. **Fixed-Manifold 2D Reference**: High-dimensional feature trajectories are stably projected into an invariant Session 1 basis (PCA/UMAP), eliminating coordinate jitter during live streaming adaptation.
4. **Frozen Synaptic Weights**: Synaptic weights $\Theta$ are frozen after Phase A, eliminating computationally expensive online backpropagation on edge devices.

---

## 🖥️ Interactive Evaluation Platform & Dashboard

The project includes an interactive research dashboard built with Streamlit and Plotly, exposing live signal drift, prototype movements, and empirical multi-subject benchmarks.

### Launching the Dashboard

```powershell
# From the project root
& ".\FYP\Scripts\python.exe" -m streamlit run dashboard/app.py
```

Or if your virtual environment is activated:
```bash
streamlit run dashboard/app.py
```
Then navigate to `http://localhost:8501` in your browser.

---

## 📖 Component-by-Component Dashboard Guide

This section explains what every widget, control, metric card, and visualization signifies:

### 1. Sidebar Controls (Left Configuration Panel)

| Control | Description | Purpose / Significance |
| :--- | :--- | :--- |
| **Subject Profile (S01–S09)** | Selects which human participant's data from BCI IV 2a is loaded. | Loads the corresponding trained PyTorch checkpoint (`phase_a_subject_X.pt`) and initial class prototypes (`prototypes_subject_X.pt`). |
| **2D Reference Basis** | Switch between **PCA** (Linear principal axes) and **UMAP** (Manifold projection). | Fits the 2D coordinate system once on Session 1 features. Session 2 trials and live prototypes project into this space to avoid frame jitter. |
| **Prototype Momentum ($1 - \alpha$)** | Slider $[0.50, 0.99]$, default $0.90$. | Controls the Exponential Moving Average inertia: $P_c^{(t)} = \mu P_c^{(t-1)} + (1 - \mu) f_t$. Higher values make prototype movement gradual; lower values adapt aggressively. |
| **Confidence Threshold ($\tau$)** | Slider $[0.25, 0.95]$, default $0.60$. | Minimum cosine softmax confidence required to update prototype memory. Prevents noisy or misclassified trials from corrupting prototypes. |
| **Sleep Consolidation Toggle** | Checkbox (`ON` by default). | Periodically generates synthetic Gaussian features around initial snapshots to pull drifted prototypes back and protect Session 1 retention. |
| **Consolidation Interval** | Number input (e.g. 25 trials). | Frequency of sleep consolidation phases (every $N$ accepted trials). |
| **Pull Strength ($\lambda$)** | Slider $[0.1, 1.0]$, default $0.5$. | Weighting of the gradient-free pull toward the snapshot: $P_c \leftarrow (1 - \lambda) P_c + \lambda P_{\text{snapshot}, c}$. |

---

### 2. Top Metrics Ribbon

- **STREAMED TRIALS**: Displays the current position in the Session 2 trial sequence (e.g., $142 / 282$) along with a visual progress bar.
- **STREAMING ACCURACY**: Cumulative accuracy of the adapting SNN model from trial $1$ to the current trial. Includes a green delta badge showing accuracy gain over the unadapted static SNN baseline.
- **TRIAL ADAPTATION**: Displays **ACCEPTED** (prototype was updated) if trial confidence $\ge \tau$, or **FILTERED** (trial rejected from prototype update). Shows exact softmax cosine confidence.
- **SLEEP CONSOLIDATION**: Displays **TRIGGERED 🌙** when synthetic anchor correction fires, showing the measured cosine drift loss ($L_{\text{sleep}} = 1 - \cos(P_{\text{curr}}, P_{\text{snapshot}})$).
- **NETWORK SPARSITY**: The measured fraction of neuron timesteps that fired spikes across both LIF layers ($~7.94\%$), proving **>92% silence**.

---

### 3. Tab 1: 🎯 Live Streaming Replay

#### A. Playback Controls
- **Streaming Replay Position Slider**: Drag to immediately scrub to any trial from $1$ to $282$.
- **⏩ Jump to End**: Fast-forwards to the final trial to inspect post-adaptation prototypes.
- **⏮️ Jump to Start**: Resets the stream to Trial 1 to observe initial Session 1 prototype locations.

#### B. Plot 1: Live Feature-Space Manifold & Trajectories
- **Faded Background Circles**: Session 1 reference trials colored by class:
  - 🔵 **Left Hand**
  - 🟢 **Right Hand**
  - 🟡 **Both Feet**
  - 🟣 **Tongue**
- **Solid Colored Dots**: Streaming Session 2 trials arriving sequentially over time.
- **Hollow Rings ($P_{\text{init}}$)**: Fixed initial locations of Session 1 prototypes before adaptation.
- **Prominent Star Markers ($\star$)**: Current live locations of adapted prototypes.
- **Dotted Trajectory Trails**: The exact historical path taken by each prototype as it adapts to distribution shift.
- **Diamond Marker**: The active trial currently being classified by the SNN.

#### C. Plot 2: Online Adaptation vs. Static SNN Baseline Curves
- **Top Subplot (Accuracy Comparison)**:
  - **Blue Solid Line**: Running cumulative accuracy of our **Adaptive SNN**.
  - **Grey Dotted Line**: Accuracy of the **Static SNN Baseline** (frozen prototypes without online adaptation). Demonstrates how adaptation actively counters sensor degradation and subject fatigue.
- **Bottom Subplot (Confidence & Threshold)**:
  - **Amber Line**: Per-trial classification confidence.
  - **Red Dashed Line**: The acceptance threshold $\tau$. Trials above this line trigger prototype updates.

#### D. Active Trial Diagnostics Card
- Displays the ground-truth motor imagery label vs. SNN predicted class.
- Prediction confidence percentage and adaptation acceptance status.
- Number of total accepted trials vs. total trials seen so far.

---

### 4. Tab 2: 📊 Cross-Subject Benchmarks (Table 5)

Displays genuine experimental results loaded directly from our 9 trained checkpoints (`phase_a_subject_1.pt` through `phase_a_subject_9.pt`):

1. **Benchmark Table**:
   - **Phase A Train Acc**: SNN training accuracy on Session 1 fit split.
   - **Phase A Val Acc**: Validation accuracy used for early stopping.
   - **SNN Prototype Test Acc**: Held-out Session 1 test set accuracy using prototype cosine classification.
   - **Linear Head Baseline**: Accuracy achieved by a standard linear classifier trained on SNN features.
   - **Advantage Column**: Empirical gain of prototype cosine classification over linear head.
2. **Comparison Bar Chart**: Side-by-side visualization of SNN Prototype Cosine accuracy vs. Linear Head Baseline across all 9 subjects.
3. **Generalization Gap Profile**: Multi-line plot tracking the performance trajectory from Train $\rightarrow$ Validation $\rightarrow$ Held-Out Test.

---

### 5. Tab 3: ⚡ Neuromorphic Energy & Sparsity

Validates the neuromorphic hardware claims:
- **Duty Cycle Donut Chart**: Shows the exact ratio of **Silent Neurons (92.06%)** vs. **Active Spiking Neurons (7.94%)**.
- **Layer-by-Layer Firing Rate Hierarchy**:
  - **Input Spike Encoder**: $34.7\%$ firing rate (dual rate + TTFS encoding).
  - **LIF Layer 1 (FC-256)**: $11.25\%$ mean firing rate.
  - **LIF Layer 2 (FC-128)**: $1.33\%$ mean firing rate.
  - **Aggregate Network Sparsity**: **$7.94\%$**.
- **Theoretical Explanation**: Explains why dense matrix multiplications on standard GPUs consume $\mathcal{O}(B \cdot T \cdot M \cdot N)$ power, whereas event-driven neuromorphic silicon consumes dynamic energy only when a spike arrives.

---

### 6. Tab 4: 📖 Platform Guide & Theory

Contains the formal mathematical formulations of:
1. Fixed Manifold Projections ($z_t = W_{\text{proj}} f_t$).
2. Prototype EMA update equations ($P_c^{(t)} = \mu P_c^{(t-1)} + (1 - \mu) f_t$).
3. Synthetic Gaussian sleep consolidation pulls ($\tilde{P}_c = P_{\text{snapshot}, c} + \mathcal{N}(0, \sigma^2 I)$).

---

## 🧪 Testing & Verification

The codebase includes an automated unit test suite covering all modules:

```powershell
& ".\FYP\Scripts\python.exe" -m pytest tests/ -v
```

**Test Coverage Summary (54/54 Passed)**:
- `test_cross_session_sweep.py`: Multi-subject deterministic seeding and sweep execution.
- `test_simulation.py`: Replay engine step progression, 2D basis consistency, and prototype tracking.
- `test_lif_neuron.py`: Leaky Integrate-and-Fire membrane dynamics, threshold firing, and reset mechanics.
- `test_snn_feature_extractor.py`: Two-layer SNN feature extractor forward passes and surrogate gradient backprop.
- `test_sleep_consolidation.py`: Synthetic feature generation, cosine loss computation, and prototype anchor pulls.
- `test_spike_encoder.py`: Rate and TTFS spike encoding, output sparsity bounds, and tensor dimensions.
- `test_metrics.py`: Accuracy, adaptation gain, retention, forgetting, and forward-hook sparsity trackers.
