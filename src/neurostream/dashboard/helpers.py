"""Small, dependency-free helpers for dashboard path and widget identity handling."""

from pathlib import Path


def resolve_repo_path(project_root: str | Path, value: str | Path) -> Path:
    """Resolve an artifact path against the repository root and legacy scripts/ root.

    Absolute paths are preserved. Relative paths prefer the repository root, then
    the historical scripts/ location. If neither exists, return the root candidate
    so callers can report a useful missing-file path.
    """
    root = Path(project_root).expanduser().resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = (root / path, root / "scripts" / path)
    return next(
        (candidate.resolve() for candidate in candidates if candidate.exists()),
        candidates[0].resolve(),
    )


def trial_slider_key(subject: int, projection: str) -> str:
    """Return a stable, subject/projection-scoped Streamlit slider key."""
    return f"trial_slider_{int(subject)}_{str(projection).lower()}"
