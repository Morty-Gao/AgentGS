from pathlib import Path


def project_root(start: Path | str | None = None) -> Path:
    candidate = Path(start or Path.cwd()).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"No pyproject.toml found above {candidate}; supply --project")


def within(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Path escapes its declared root")
    return target


def data_resource(root: Path, relative: str) -> Path:
    """Resolve inside the declared data root, including a linked local data bundle."""
    path = Path(relative).relative_to("data")
    return within(root / "data", str(path))
