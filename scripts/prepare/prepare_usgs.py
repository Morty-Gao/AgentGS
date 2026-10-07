"""Validate the original frozen catalog and write normalized data/provenance separately."""

import argparse
import hashlib
import json
from pathlib import Path

from agentgs.datasets.registry import DatasetRegistry


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/usgs/processed/catalog.json")
    args = parser.parse_args()
    registry = DatasetRegistry(root)
    source = registry.resolve("usgs", "usgs_2010_2020_m7.csv")
    rows = registry.csv_rows("usgs", "usgs_2010_2020_m7.csv")
    if not rows or not {"id", "mag", "time"}.issubset(rows[0]):
        raise ValueError("Invalid USGS catalog columns")
    path = root / args.output
    if path.resolve() == source.resolve():
        raise ValueError("Cannot overwrite original input")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "records": rows},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
