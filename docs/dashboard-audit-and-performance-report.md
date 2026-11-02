# NeuroStream dashboard audit and redesign report

Date: 2026-11-02

## Scope and evidence

Reviewed the repository's dashboard entrypoint and implementation, replay engine,
evaluation artifact serializer/validator, cross-session sweep manifest writer,
EEG loader, spike encoder, SNN feature extractor, prototype memory, README and
dashboard tests. The redesign preserves the existing model and evaluation
methodology. It changes the dashboard's data access and presentation layer.

The dashboard is an offline research interface. It is not a live EEG acquisition
system or a causal real-time BCI.

## Findings and remediation

| Severity | Area / files | Root cause or risk | Remediation | Verification |
|---|---|---|---|---|
| Critical | `src/neurostream/dashboard/app.py`, `src/neurostream/evaluation/cross_session_eval.py` | The sweep writes `evaluation_manifest.json` under the configured checkpoint root (the default sweep root is `results/checkpoints/sweep`); the former app checked only `results/checkpoints/evaluation_manifest.json`. Valid runs could be missed. | New `registry.py` recursively discovers manifests below `results/`. | Added registry tests; runtime discovery still requires the user's local artifacts. |
| High | Former dashboard startup / `simulation.py` | A missing prepared cache could fall through to checkpoint loading, MOABB EEG loading, encoding, SNN feature extraction and projection fitting during a page request. | Replay page now requires an explicit prepared feature cache and shows a command with the exact selected checkpoint/prototype paths. No automatic full-pipeline fallback. | Static source contract checks the prepared-artifact path; cold-start timing not measured here. |
| High | Former dashboard cache functions | Global resource caching of model/replay objects created a difficult cache identity and mutable-resource boundary; stale session/selection combinations were possible. | Removed Streamlit global cache decorators from the new app. The small replay engine/result are held in explicit per-session state and invalidated using checkpoint hash, prototype hash, projection and prepared-cache timestamp. | Static source test asserts no `st.cache_data` or `st.cache_resource` calls in the new app. Multi-user runtime test remains outstanding. |
| High | Former dashboard page gates | Page-specific load flags and reruns were coupled to navigation state, making a return to a page look like a failed load. | Replaced the monolithic load gate with independent page rendering and explicit action buttons. | Page routing is covered by source-level tests; browser navigation is not yet exercised. |
| High | Evaluation artifact selection | A subject/seed selection must bind the evaluation JSON to the exact checkpoint, not merely a matching filename. | Registry calls the existing `load_evaluation_artifact` validator, which checks schema, held-out split, subject and checkpoint SHA-256; seed is also checked against the manifest. | Existing validator plus new missing-checkpoint and paired-join tests. |
| Medium | EEG signal page | Loading a dataset during ordinary dashboard startup/navigation is expensive and may require MOABB download/cache access. | Signal Analysis loads only after the user clicks **Load EEG sessions**. Failures are rendered with their exception details. | Source contract checks explicit loading; actual dataset access depends on local MOABB cache/network. |
| Medium | Signal labeling | The loader returns preprocessed epochs; the old presentation could be read as showing raw EEG. | The new page labels amplitude as preprocessed units and explicitly states that raw continuous, filtered-only and normalized-stage signals are not separately retained by this loader. | UI text reviewed against `SessionData` and preprocessing loader contract. |
| Medium | Benchmark missing values | Missing data must not be interpreted as zero. | Metrics remain null/NaN in tables and heatmaps; no fill-with-zero behavior is used. Confusion/class metrics render only when a stored matrix exists. | Static review; numerical helper tests can be expanded with real fixtures. |
| Medium | Consolidation claims | ON/OFF comparisons are meaningful only when subject and seed are paired. | Paired report rows are joined on exact `(subject, seed)`; unpaired runs are excluded. Mean paired differences and a t interval are shown only for available pairs. | Added exact-pair join test. |
| Medium | Replay interaction | Widget changes could rerun the expensive replay. | Replay parameters are submitted in a form; the result is computed once and kept in session state. The trial slider, start/end buttons and plots read stored records. | Source contract verifies explicit submit and prepared-artifact loading; interaction timing remains unmeasured. |
| High | Prepared-cache identity | Cache directory names based only on subject and checkpoint filename collided across seeds (sweep checkpoints reuse the same filename); changing prototypes could overwrite a prior cache. | Cache paths now include checkpoint and prototype SHA-256 prefixes. Runtime identity also includes projection plus metadata/features/projector file stats; the app rejects a checkpoint that changed after registry validation. | Added source-contract checks for hash-based cache identity; latest full CI run pending. |
| Medium | Streamlit API compatibility | `use_container_width` works on the old minimum but is deprecated in current Streamlit; `width="stretch"` is not accepted by the old minimum. | Rendering wrappers inspect the installed `st.plotly_chart` and `st.dataframe` signatures and choose the compatible argument. | AppTest exercises chart/table rendering on the current CI Streamlit; compatibility fallback is source-tested. |
| Medium | EEG preprocessing provenance | A signal page using loader defaults can misrepresent the selected checkpoint's preprocessing. | When a validated run is selected, the page reads its checkpoint contract, verifies subject/seed/hash, loads the matching epoch/filter/channel settings and applies saved normalization. The no-run path is labelled as default/unbound. | Added checkpoint contract tests; real MOABB/checkpoint load remains a local verification step. |
| Medium | Replay drift interpretation | The replay schema stores per-step projected prototypes and a drift loss only at consolidation triggers; it does not store full high-dimensional prototype history per step. | UI reports 2D projected displacement and stored trigger drift only; it explicitly says high-dimensional per-trial drift is unavailable. | Reviewed `ReplayStepRecord` and `ReplaySimulationData` fields. |
| Medium | Spike activity | Spike opportunity fractions are not hardware energy or power. | UI retains fraction-of-spike-opportunities terminology and avoids energy claims. Per-layer activity is displayed only when the matching prepared artifact contains it. | Reviewed `SpikeActivityMeasurement` schema and dashboard labels. |
| Low | Responsive/rendering behavior | Oversized styling and broad page computation increase rendering fragility. | Replaced the old banner-heavy monolith with compact page sections, standard Plotly containers and seven focused workspaces. | Static code review; visual browser testing remains outstanding. |
| Medium | Repository hygiene | Git tracked four copies of the two raw BNCI session files (about 173 MB total), a Matplotlib cache, a Python bytecode file and an empty invalid DOCX placeholder. | Removed those tracked cache/data/empty-placeholder files, added `data/mne_data/` to `.gitignore`, and retained the small Phase A checkpoint/prototype pair because it is a useful model artifact. | Latest Git tree confirms no tracked `.cache/`, `__pycache__/` or `data/mne_data/` files; the two checkpoint files remain. |

## New navigation

1. **Overview** — selected validated run, static target baseline, prequential
   target accuracy, retrospective diagnostic, source retention, adaptation gain,
   measured spike activity, subject-wise results and recent run metadata.
2. **Signal Analysis** — explicit session load, trial/channel/window selection,
   waveform, alpha/beta/gamma channel power, session comparison and epoch quality
   checks. The comparison is illustrative, not a paired-trial experiment.
3. **Model Diagnostics** — model structure, stored confusion matrix, class-wise
   precision/recall/F1 when a confusion matrix is present, prepared layer activity
   when available, and optional feature-space distribution.
4. **Adaptation Replay** — offline replay from a validated feature-only cache,
   parameter form, trial navigation, prediction/confidence/acceptance, projected
   prototype paths, accuracy progression, accepted/rejected counts, confidence
   distribution and recorded consolidation drift.
5. **Benchmark Lab** — subject × seed heatmap, static/prequential/retrospective
   comparisons, subject-wise and seed-wise variability, and stored confusion
   matrices/class-wise metrics.
6. **Consolidation Study** — paired ON/OFF outcomes, matched-pair differences,
   uncertainty interval and recorded consolidation phase counts.
7. **Experiment Registry** — manifest/checkpoint/evaluation status, provenance,
   hashes, validation errors and prepared-cache inventory.

## Data and execution boundaries

- Overview, Benchmark Lab and Experiment Registry read local JSON artifacts and
  do not load EEG or deserialize model checkpoints.
- Evaluation metrics are exposed only after checkpoint hash validation.
- Signal Analysis is an explicit, user-triggered dataset operation.
- Replay uses `TrialReplayEngine.from_prepared_artifact`; it does not silently
  fall back to the raw EEG → encoder → SNN → projection preparation pipeline.
- Prepared replay artifacts remain feature-only: raw EEG and encoded spike
  tensors are not added to the dashboard cache.
- The app does not use Streamlit's global cache decorators. Session state is
  used for explicit, user-session state; the registry can be refreshed manually.
- Checkpoint metadata is loaded with PyTorch's restricted `weights_only=True`
  loader first. Legacy checkpoints that store NumPy calibration arrays require
  the same explicit `weights_only=False` compatibility fallback used by the
  existing Phase B runner. Only load checkpoints from trusted local project
  outputs; a matching SHA-256 proves identity, not publisher trust.
- The local `projector.joblib` file is deserialized by the existing replay
  artifact loader. Only load prepared caches from trusted local project outputs;
  do not treat an untrusted joblib file as safe input.

## Performance report

No performance numbers are claimed in this change. This environment did not run
the user's local MOABB dataset, checkpoint files, prepared caches or browser.
The requested targets (overview <3 s, indexed benchmark selection <1 s, replay
scrub <300 ms) remain targets, not measured outcomes.

The redesign removes expensive work from normal navigation by construction.
Actual cold/warm timings should be recorded on the user's machine for:
- Streamlit cold start and warm rerun;
- registry refresh;
- signal dataset load;
- prepared cache load;
- replay computation after settings submission;
- trial slider scrub and chart render.

## Verification status

Added/updated tests:
- `tests/test_dashboard_source_contract.py`: Python AST parse, seven-page
  navigation, no global Streamlit cache decorators, explicit EEG loading,
  prepared replay loading and compatible chart arguments.
- `tests/test_dashboard_registry.py`: missing-checkpoint handling, exact
  subject/seed pairing and repository-root resolution.

**Verified by GitHub Actions:** at commit `5ceb8e1`, the complete suite
reported **109 passed, 5 warnings**; Python source compilation and the dashboard
dependency smoke test also passed. The suite includes Streamlit AppTest navigation
through all seven pages and a synthetic-epoch Signal Analysis chart/band-power
render. The warnings were dependency/runtime warnings, not test failures.

**Still not verified here:** launching a long-running Streamlit server in the
user's Windows environment, loading the user's actual MOABB dataset/checkpoints/
prepared replay cache, scrubbing replay against their real experiment artifacts,
browser-console behavior, multi-user load and performance timings. AppTest uses
synthetic epochs for the signal chart; it does not prove a real MOABB download or
a real replay artifact works. Do not treat CI as a substitute for that local
data-backed walkthrough.

## Local verification commands

```powershell
python -m pip install -e ".[dashboard]"
python -m pytest -q
python -m streamlit run dashboard/app.py
```

If replay cache is absent, use the exact checkpoint and prototype paths printed
on the Adaptation Replay page. Do not use a default checkpoint path when the
selected run belongs to a nested subject/seed sweep directory.

## Implementation changelog

- Replaced `src/neurostream/dashboard/app.py` with a seven-page, artifact-first
  Streamlit workstation and explicit session-scoped replay state.
- Added `src/neurostream/dashboard/registry.py` for recursive manifest discovery,
  checkpoint/evaluation hash validation, orphan checkpoint visibility, paired
  ablation joins and safe checkpoint preprocessing-contract extraction.
- Hardened `scripts/precompute_dashboard_artifacts.py` so its default checkpoint
  directory is selected by actual checkpoint-file presence, not merely by an
  existing empty directory.
- Added dashboard source, registry, artifact-validation and Streamlit AppTest
  coverage under `tests/test_dashboard_source_contract.py` and
  `tests/test_dashboard_registry.py`.
- Rewrote the dashboard guide and verification instructions in `README.md`.
- Added this audit/performance report.
- Updated `.gitignore` for local MOABB/MNE dataset downloads and generated
  `scripts/results/` outputs. Removed the tracked local EEG cache copies,
  Matplotlib cache, bytecode file and empty DOCX placeholder. Retained the
  existing small Phase A checkpoint/prototype pair.
