"""Prepare raw inputs, recording the actual upstream version used."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
MANIFESTS = Path(__file__).with_name("manifests")
USED_SOURCES = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_manifest(relative: str):
    return json.loads((MANIFESTS / relative).read_text("utf-8"))


def require_source(path: Path, digest: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required original input is missing: {path}")
    if sha256(path) != digest:
        raise ValueError(f"Source hash mismatch: {path}; refusing to substitute another version")
    return path


def fetch(key: str, cache: Path, offline: bool, catalog=None) -> Path:
    spec = (read_manifest("sources.json") if catalog is None else catalog)[key]
    target = cache / key
    if not target.exists() and spec.get("local_glob"):
        candidates = sorted(
            cache.glob(spec["local_glob"]), key=lambda p: (p.stat().st_mtime, p.name)
        )
        if candidates:
            target = candidates[-1]
    if target.is_file():
        record_source(key, target, spec)
        return target
    if offline or not (spec["urls"] or spec.get("latest_urls")):
        raise FileNotFoundError(f"Original source is not cached: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    errors = []
    for url in dict.fromkeys([*spec["urls"], *spec.get("latest_urls", [])]):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AgentGS-prepare/1"})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, 1024 * 1024)
            if not temporary.stat().st_size:
                raise ValueError("Downloaded source is empty")
            os.replace(temporary, target)
            record_source(key, target, spec, url)
            return target
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
    raise RuntimeError("\n".join(errors))


def record_source(key: str, path: Path, spec=None, url=None):
    spec = spec or {}
    actual = sha256(path)
    USED_SOURCES[key] = {
        "sha256": actual,
        "size": path.stat().st_size,
        "filename": path.name,
        "preferred_sha256": spec.get("sha256"),
        "matches_preferred_version": actual == spec.get("sha256") if spec.get("sha256") else None,
        "url": url,
    }
    provenance = path.with_name(path.name + ".source.json")
    if url is None and provenance.exists():
        cached = json.loads(provenance.read_text("utf-8"))
        if cached.get("sha256") == actual:
            USED_SOURCES[key]["url"] = cached.get("url")
    if url is not None:
        write_json(provenance, USED_SOURCES[key])
    return path


def existing_assets(domain: str, source_root: Path, output_root: Path) -> list[Path]:
    """Import the explicitly retained, prepared inputs for a dataset."""
    from _formats import normalize_file

    from agentgs.datasets.layout import RETAINED_FILES

    inputs = [(relative, source_root / relative) for relative in RETAINED_FILES[domain]]
    for relative, source in inputs:
        if not source.is_file():
            raise FileNotFoundError(f"Retained dataset input is missing: {source}")
    outputs = []
    for relative, source in inputs:
        record_source("retained/" + relative, source)
        outputs.append(normalize_file(source, output_root / "data" / relative))
    return outputs


def main(domain: str) -> None:
    from _domains import build

    parser = argparse.ArgumentParser(description=f"Prepare the fixed {domain} scientific inputs")
    parser.add_argument("--output-root", type=Path, default=REPO)
    parser.add_argument("--cache-root", type=Path, default=REPO / "data/raw")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Retained prepared datasets; default: this repository data/",
    )
    parser.add_argument("--offline", action="store_true")
    if domain == "gem":
        parser.add_argument(
            "--workbook",
            type=Path,
            help="One-time import of a supplied GEM XLSX into data/gem.jsonl",
        )
    parser.add_argument(
        "--archive", type=Path, help="Extract a prepared dataset ZIP instead of rebuilding it"
    )
    parser.add_argument(
        "--archive-threshold-mb",
        type=float,
        default=64,
        help="Compress multi-file datasets larger than this size; default: 64 MiB",
    )
    args = parser.parse_args()
    USED_SOURCES.clear()
    output_root = args.output_root.resolve()
    cache = args.cache_root.resolve()
    source_root = (args.source_root or REPO / "data").resolve()
    from agentgs.datasets.references import REFERENCE_FILES, load_reference

    reference_hashes = {}
    for relative in REFERENCE_FILES:
        target = output_root / "data" / relative
        write_json(target, load_reference(relative, root=source_root.parent))
        reference_hashes[relative] = sha256(target)
    aliases_path = output_root / "data/metadata/resources.json"
    aliases = (
        json.loads(aliases_path.read_text("utf-8")) if aliases_path.exists() else {"aliases": {}}
    )
    from _formats import pack_dataset, unpack_dataset

    from agentgs.datasets.layout import DATASET_DIRECTORIES, prepared_locator

    aliases["aliases"].update(
        {
            key: prepared_locator(value)
            for key, value in read_manifest("resource_aliases.json")["aliases"].items()
        }
    )
    write_json(aliases_path, aliases)
    try:
        retained_archive = source_root / (DATASET_DIRECTORIES.get(domain, domain) + ".zip")
        archive_source = args.archive or (
            retained_archive
            if domain in {"deep_earth", "geotoolkg", "biodeeptime", "ccs"}
            and retained_archive.exists()
            and not (source_root / domain).exists()
            else None
        )
        if domain == "gem" and args.workbook:
            from _gem import convert_workbook

            workbook = args.workbook.resolve()
            record_source("import/gem/" + workbook.name, workbook)
            target = convert_workbook(workbook, output_root / "data/gem.jsonl")
            outputs, missing, notes = (
                [target],
                [],
                ["All workbook sheets, cell values and cached formulas exported."],
            )
        elif archive_source:
            if domain not in DATASET_DIRECTORIES:
                parser.error("A single-file dataset does not need archive extraction")
            folder = output_root / "data" / DATASET_DIRECTORIES[domain]
            archive = archive_source.resolve()
            record_source("archive/" + archive.name, archive)
            unpack_dataset(archive, folder)
            retained = output_root / "data" / (DATASET_DIRECTORIES[domain] + ".zip")
            retained.parent.mkdir(parents=True, exist_ok=True)
            if archive != retained:
                shutil.copyfile(archive, retained)
            outputs, missing, notes = (
                [retained],
                [],
                ["Prepared archive expanded into its dataset directory under data/."],
            )
        else:
            outputs, missing, notes = build(domain, output_root, cache, args.offline, source_root)
        if not missing and domain in DATASET_DIRECTORIES and not archive_source:
            folder = output_root / "data" / DATASET_DIRECTORIES[domain]
            dataset_files = [p for p in outputs if p.is_relative_to(folder)]
            size = sum(p.stat().st_size for p in dataset_files)
            if size > args.archive_threshold_mb * 1024 * 1024:
                archive = output_root / "data" / (DATASET_DIRECTORIES[domain] + ".zip")
                pack_dataset(
                    folder, archive, {str(p.relative_to(folder)): sha256(p) for p in dataset_files}
                )
                outputs = [p for p in outputs if p not in dataset_files] + [archive]
                notes.append(
                    "Prepared ZIP retained; ready-to-use files remain in the dataset directory under data/."
                )
    except FileNotFoundError as exc:
        outputs, missing, notes = [], [str(exc)], []
    for prefix, label in sorted(
        [
            (str(source_root), "<source-root>"),
            (str(cache), "<cache-root>"),
            (str(output_root), "<output-root>"),
        ],
        key=lambda item: -len(item[0]),
    ):
        missing = [item.replace(prefix, label) for item in missing]
    manifest = {
        "domain": domain,
        "mode": "archive" if archive_source else "raw",
        "sources": dict(sorted(USED_SOURCES.items())),
        "recipe_files": {
            str(p.relative_to(MANIFESTS)): sha256(p) for p in sorted(MANIFESTS.rglob("*.json"))
        },
        "reference_files": reference_hashes,
        "preparation_code": {
            name: sha256(Path(__file__).with_name(name)) for name in ["_common.py", "_domains.py"]
        },
        "status": "partial" if missing else "complete",
        "missing": missing,
        "notes": notes,
        "files": {str(p.relative_to(output_root)): sha256(p) for p in sorted(outputs)},
    }
    write_json(output_root / f"data/metadata/prepare_{domain}.json", manifest)
    print(
        json.dumps(
            {
                "domain": domain,
                "status": manifest["status"],
                "file_count": len(outputs),
                "missing": missing,
            },
            ensure_ascii=False,
        )
    )
    if missing:
        raise SystemExit(2)
