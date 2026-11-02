from pathlib import Path

from neurostream.dashboard.helpers import resolve_repo_path, trial_slider_key


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
