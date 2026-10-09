"""Explicit local benchmark version selection, without an external source dependency."""

import json
from pathlib import Path

from agentgs.config.paths import data_resource
from agentgs.datasets.geotooleval import benchmark_dataset as benchmark_dataset
from agentgs.datasets.geotooleval import benchmark_manifest


def compatibility_contracts(dataset: Path, *, project: Path | None = None) -> Path | None:
    """Resolve optional historical schemas through the current prepared-data layout."""
    if project is None:
        project = next(
            (
                parent
                for parent in dataset.resolve().parents
                if (parent / "pyproject.toml").is_file()
            ),
            None,
        )
    selected = benchmark_manifest(project) if project is not None else {}
    if not selected.get("compatibility_contracts"):
        legacy = dataset.parent / "version_manifest.json"
        selected = json.loads(legacy.read_text("utf-8")) if legacy.exists() else {}
    relative = selected.get("compatibility_contracts")
    if not relative:
        return None
    if project is None:
        raise ValueError("Supply the project root to resolve compatibility contracts")
    path = data_resource(project, relative)
    if not path.is_file():
        raise FileNotFoundError(f"Selected compatibility contracts are missing: {path}")
    return path
