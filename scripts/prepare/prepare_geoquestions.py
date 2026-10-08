"""Build the pinned 40-query replay snapshot from already downloaded original files."""

import argparse
import hashlib
import json
from pathlib import Path

from agentgs.datasets.geoquestions import (
    SNAPSHOT_RESOURCE,
    prepare_snapshot,
    snapshot_bytes,
    snapshot_profile,
)
from agentgs.datasets.registry import DatasetRegistry


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=root)
    args = parser.parse_args()
    registry = DatasetRegistry(args.project)
    profile = snapshot_profile()
    originals = {}
    for name, expected in profile["source_files"].items():
        content = registry.resolve("geoquestions1089", name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Original file differs from the pinned source: {name}")
        originals[name] = json.loads(content)
    rows = prepare_snapshot(
        originals["GeoQuestions1089.json"], originals["GeoQuestions1089_answers.json"]
    )
    content = snapshot_bytes(rows)
    if hashlib.sha256(content).hexdigest() != profile["snapshot_sha256"]:
        raise ValueError("Generated snapshot differs from the approved replay version")
    target = args.project / "data" / "geoquestions1089" / SNAPSHOT_RESOURCE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print(json.dumps({"snapshot": str(target), "records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
