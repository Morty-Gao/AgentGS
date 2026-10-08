"""Import frozen scientific inputs and indexes; never copy benchmark gold into runtime tables."""

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows_alias(path):
    text = str(path).replace("\\", "/")
    if text.startswith("/mnt/") and len(text) > 6:
        return text[5].upper() + ":" + text[6:]
    return text


def main():
    project = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()
    source, root = args.source.resolve(), args.project.resolve()
    if root == source or root.is_relative_to(source):
        parser.error("Import destination must be outside the read-only source")
    groups = [
        ("datasets/deep_earth_toolqa/data/derived", "data/processed/deep_earth/derived"),
        ("datasets/deep_earth_toolqa/data/source_snapshot", "data/processed/deep_earth/source_snapshot"),
        ("datasets/energy_resources_toolqa/data/derived", "data/processed/energy/derived"),
        ("datasets/ocean_carbon_toolqa/data/derived", "data/processed/ocean/derived"),
        ("_dataset_work/habitable_earth_dataset/processed", "data/processed/habitable"),
        ("KG/geotoolkg", "data/indexes/geotoolkg"),
        ("data/models/fastembed", "data/models/fastembed"),
    ]
    if not args.skip_assets:
        groups.append(("data/benchmarks", "data/benchmarks"))
    previous = root / "data/processed/resource_manifest.json"
    manifest = json.loads(previous.read_text()) if previous.exists() else {
        "version": 1, "records": [], "aliases": {}, "kind": "frozen_reference_inputs"
    }
    records = {row["path"]: row for row in manifest["records"]}
    aliases = manifest["aliases"]
    basenames = defaultdict(list)
    roots = [source, args.source.absolute()]

    def add_aliases(original, target):
        relative = str(target.relative_to(root))
        aliases[str(original)] = relative
        aliases[windows_alias(original)] = relative
        original_relative = original.relative_to(source)
        aliases[str(original_relative)] = relative
        for prefix in roots:
            aliases[str(prefix / original_relative)] = relative
            aliases[windows_alias(prefix / original_relative)] = relative

    count = 0
    for origin, destination in groups:
        folder = source / origin
        if not folder.exists():
            raise FileNotFoundError(folder)
        target_root = root / destination
        target_root.mkdir(parents=True, exist_ok=True)
        add_aliases(folder, target_root)
        for original in sorted(folder.rglob("*")):
            rel = original.relative_to(folder)
            if original.is_dir():
                add_aliases(original, target_root / rel)
                continue
            if original.suffix.lower() == ".zip" or any(
                part.casefold() in {"questions", "answers", "reports", "evaluation", "logs"}
                for part in rel.parts
            ) or original.name.casefold().startswith(("questions", "answers", "predictions")):
                continue
            if original.name.casefold() == "question.json":
                continue
            target = target_root / rel
            fingerprint = digest(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != fingerprint:
                    raise ValueError(f"Existing resource differs; preserve both versions: {target}")
            else:
                shutil.copy2(original, target)
                if digest(target) != fingerprint:
                    raise ValueError(f"Imported resource checksum differs: {target}")
            key = str(target.relative_to(root))
            records[key] = {"path": key, "sha256": fingerprint, "bytes": target.stat().st_size,
                            "source": str(original), "role": "model" if "models" in destination else "index" if "indexes" in destination else "input"}
            add_aliases(original, target)
            basenames[original.name].append((key, fingerprint))
            count += 1
            if count % 250 == 0:
                print(json.dumps({"imported_or_verified": count}), flush=True)
    # Online readers require the original full GEM/MRDS/SNAPO parsers, not their question generators.
    for original, target in [
        (source / "dataset_work/extracted/gem-data/Global-Integrated-Power-March-2026-II.xlsx", root / "data/processed/raw_sources/gem-data/Global-Integrated-Power-March-2026-II.xlsx"),
        (source / "dataset_work/extracted/mrds-csv/mrds.csv", root / "data/processed/raw_sources/mrds-csv/mrds.csv"),
    ]:
        target.parent.mkdir(parents=True, exist_ok=True)
        sha = digest(original)
        if not target.exists():
            shutil.copy2(original, target)
        if digest(target) != sha:
            raise ValueError(f"Raw input checksum differs: {target}")
        key = str(target.relative_to(root))
        records[key] = {"path": key, "sha256": sha, "bytes": target.stat().st_size, "source": str(original), "role": "input"}
        add_aliases(original, target)
    for key, row in records.items():
        basenames[Path(key).name].append((key, row["sha256"]))
        if key.startswith("data/benchmarks/earthbench_geotoolagent/") and "/question" in key:
            tail = key[key.index("question"):]
            aliases["benchmark/data/" + tail] = key
            parts = Path(tail).parts
            for count in range(1, len(parts)):
                source_dir = "benchmark/data/" + "/".join(parts[:count])
                aliases[source_dir] = str(Path(key).parents[len(parts) - count - 1])
    for name, candidates in basenames.items():
        if len({sha for _, sha in candidates}) == 1:
            aliases[name] = sorted({key for key, _ in candidates})[0]
    manifest.update(records=list(records.values()), aliases=aliases)
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"records": len(records), "manifest": str(previous)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
