"""Static contracts for dashboard widget compatibility and navigation."""

import ast
from pathlib import Path


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "neurostream"
    / "dashboard"
    / "app.py"
)


def _dashboard_tree() -> ast.Module:
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def test_streamlit_chart_width_arguments_are_compatible_with_minimum_version() -> None:
    tree = _dashboard_tree()
    chart_functions = {"plotly_chart", "dataframe"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute):
            continue
        if function.attr not in chart_functions:
            continue

        for keyword in node.keywords:
            assert not (
                keyword.arg == "width"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "stretch"
            ), (
                f"{function.attr} at line {node.lineno} uses width='stretch', "
                "which is not supported by the project's Streamlit >=1.30 baseline"
            )


def test_replay_configuration_is_submitted_as_one_form() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'with st.sidebar.form("replay_configuration_form")' in source
    assert "st.form_submit_button(" in source


def test_dashboard_pages_have_distinct_navigation_options() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    expected_pages = (
        "Offline Trial Replay",
        "EEG Signal Analysis",
        "Model Diagnostics",
        "Cross-Subject Benchmarks",
        "Measured Spike Activity",
        "Platform Guide",
    )
    for page in expected_pages:
        assert f'"{page}"' in source


def test_replay_result_uses_resource_cache_for_fast_widget_reruns() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert '@st.cache_resource(show_spinner="Simulating streaming trial adaptation...")' in source
