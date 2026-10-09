"""Explicit local benchmark version selection, without an external source dependency."""

import json
from pathlib import Path

from agentgs.config.paths import data_resource


def benchmark_manifest(root: Path):
    path = root / "data/metadata/geotooleval.json"
    return json.loads(path.read_text("utf-8")).get("version_manifest", {}) if path.exists() else {}


def benchmark_dataset(root: Path):
    manifest = benchmark_manifest(root)
    relative = manifest.get("selected_dataset", "data/geotooleval.jsonl")
    return data_resource(root, relative)
