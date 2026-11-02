from pathlib import Path

from neurostream.dashboard.helpers import (
    clamp_trial_index,
    resolve_repo_path,
    trial_slider_key,
)


def test_trial_slider_key_is_scoped() -> None:
    assert trial_slider_key(1, "PCA") == "trial_slider_1_pca"
    assert trial_slider_key(2, "UMAP") == "trial_slider_2_umap"
    assert trial_slider_key(1, "PCA") != trial_slider_key(2, "PCA")


def test_trial_slider_key_includes_checkpoint_identity() -> None:
    first = trial_slider_key(1, "PCA", "seed-42")
    second = trial_slider_key(1, "PCA", "seed-43")
    assert first == "trial_slider_1_pca_seed-42"
    assert first != second


def test_resolve_repo_path_prefers_root(tmp_path: Path) -> None:
    target = tmp_path / "results" / "run.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    assert resolve_repo_path(tmp_path, "results/run.json") == target.resolve()


def test_resolve_repo_path_supports_scripts_fallback(tmp_path: Path) -> None:
    target = tmp_path / "scripts" / "results" / "run.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    assert resolve_repo_path(tmp_path, "results/run.json") == target.resolve()


def test_resolve_repo_path_preserves_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "run.json"
    target.write_text("{}", encoding="utf-8")
    assert resolve_repo_path(tmp_path, target) == target.resolve()


def test_clamp_trial_index_within_bounds() -> None:
    assert clamp_trial_index(1, 100) == 1
    assert clamp_trial_index(55, 100) == 55
    assert clamp_trial_index(100, 100) == 100


def test_clamp_trial_index_clamps_out_of_range_values() -> None:
    assert clamp_trial_index(-5, 100) == 1
    assert clamp_trial_index(150, 100) == 100


def test_clamp_trial_index_rejects_empty_replay() -> None:
    import pytest

    with pytest.raises(ValueError, match="total_steps must be at least 1"):
        clamp_trial_index(1, 0)
