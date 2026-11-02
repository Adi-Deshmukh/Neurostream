"""Static source contracts for the artifact-first Streamlit dashboard."""
import ast
from pathlib import Path


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "neurostream" / "dashboard" / "app.py"
)


def source_tree():
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def test_dashboard_app_is_valid_python():
    assert isinstance(source_tree(), ast.Module)


def test_navigation_contains_all_research_workspaces():
    source = APP_PATH.read_text(encoding="utf-8")
    for page in (
        "Overview", "Signal Analysis", "Model Diagnostics",
        "Adaptation Replay", "Benchmark Lab", "Consolidation Study",
        "Experiment Registry",
    ):
        assert f'"{page}"' in source


def test_normal_dashboard_does_not_use_streamlit_global_cache():
    tree = source_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "st"
                and node.func.attr in {"cache_data", "cache_resource"}
            ), "Dashboard should use explicit session-scoped state, not opaque global caches"


def test_heavy_eeg_loading_is_explicitly_triggered():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'if st.button("Load EEG sessions", type="primary"):' in source
    assert "load_bnci2014_001(subjects=subject)" in source


def test_replay_uses_prepared_artifacts_and_explicit_submit():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "TrialReplayEngine.from_prepared_artifact" in source
    assert 'st.form_submit_button("Run offline replay")' in source
    assert "engine.run_replay(" in source


def test_chart_width_arguments_remain_compatible_with_streamlit_baseline():
    tree = source_tree()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"plotly_chart", "dataframe"}:
            continue
        for keyword in node.keywords:
            assert not (
                keyword.arg == "width"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "stretch"
            )


def test_all_dashboard_pages_render_without_experiment_artifacts():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    page_names = (
        "Signal Analysis", "Model Diagnostics", "Adaptation Replay",
        "Benchmark Lab", "Consolidation Study", "Experiment Registry",
        "Overview",
    )
    for page_name in page_names:
        app.sidebar.radio[0].set_value(page_name).run()
        assert not app.exception, f"Dashboard page failed to render: {page_name}"
