"""Explicit local benchmark version selection, without an external source dependency."""

import json
from pathlib import Path

from agentgs.config.paths import data_resource


def benchmark_manifest(root: Path):
    path = root / "data/geotooleval/version_manifest.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def benchmark_workbook(root: Path):
    manifest = benchmark_manifest(root)
    relative = manifest.get("selected_workbook", "data/geotooleval/GeoToolEval.xlsx")
    return data_resource(root, relative)
