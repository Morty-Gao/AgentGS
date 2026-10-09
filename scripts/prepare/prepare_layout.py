"""Organize prepared files by dataset; raw downloads remain in data/raw/."""

import argparse
import json
import shutil
from pathlib import Path

from _common import REPO, sha256, write_json

from agentgs.datasets.layout import FLAT_FILES, PREPARED_FILES, prepared_locator


def migrate(root):
    data = root / "data"
    metadata = data / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for filename, key in [
        ("geotooleval_manifest.json", "dataset_manifest"),
        ("geotooleval_workbook_metadata.jsonl", "workbook_metadata"),
        ("geotooleval_version_manifest.json", "version_manifest"),
    ]:
        path = data / filename
        if path.exists():
            provenance[key] = (
                [json.loads(line) for line in path.read_text().splitlines() if line]
                if path.suffix == ".jsonl"
                else json.loads(path.read_text())
            )
            backup = root / "tmp/data-layout-backup/layout" / filename
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, backup)
    if provenance:
        write_json(metadata / "geotooleval.json", provenance)
    moved = []
    for key, flat in FLAT_FILES.items():
        source = data / flat
        target = data / PREPARED_FILES[key]
        if not source.is_file() or source == target:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"Dataset destination already exists: {target}")
        shutil.move(source, target)
        moved.append(str(target.relative_to(data)))
    aliases = metadata / "resources.json"
    if aliases.exists():
        value = json.loads(aliases.read_text())
        old_to_new = {
            "data/" + flat: "data/" + PREPARED_FILES[key] for key, flat in FLAT_FILES.items()
        }
        value["aliases"] = {
            old: old_to_new.get(new, prepared_locator(new))
            for old, new in value.get("aliases", {}).items()
        }
        write_json(aliases, value)
    manifest = data / "geoquestions/manifest.json"
    if manifest.exists():
        value = json.loads(manifest.read_text())
        value["outputs"] = {
            name: sha256(data / "geoquestions" / name)
            for name in ["queries.jsonl", "questions.jsonl"]
        }
        write_json(manifest, value)
    manifest = data / "deep_earth/manifest.json"
    if manifest.exists():
        value = json.loads(manifest.read_text())
        value["prepared_files"] = {
            Path(name).name: sha256(data / name)
            for key, name in PREPARED_FILES.items()
            if key.startswith("processed/deep_earth/") and Path(name).name != "manifest.json"
        }
        write_json(manifest, value)
    print(json.dumps({"organized_files": sorted(moved)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=REPO)
    migrate(parser.parse_args().project.resolve())
