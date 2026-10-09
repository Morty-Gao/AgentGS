"""Portable, lossless serialization of prepared tables and spatial assets."""

import csv
import gzip
import json
import shutil
from pathlib import Path


def normalize_file(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".prepare")
    try:
        if target.suffix == ".csv":
            opener = gzip.open if source.suffix == ".gz" else open
            with opener(source, "rt", encoding="utf-8-sig", newline="") as incoming:
                with temporary.open("w", encoding="utf-8", newline="") as output:
                    writer = csv.writer(output, lineterminator="\n")
                    writer.writerows(csv.reader(incoming))
        elif target.suffix in {".json", ".geojson"}:
            value = json.loads(source.read_text("utf-8-sig"))
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        elif target.suffix == ".jsonl":
            opener = gzip.open if source.suffix == ".gz" else open
            with opener(source, "rt", encoding="utf-8-sig") as incoming:
                with temporary.open("w", encoding="utf-8") as output:
                    for line in incoming:
                        if line.strip():
                            output.write(
                                json.dumps(json.loads(line), ensure_ascii=False, allow_nan=False)
                                + "\n"
                            )
        else:
            shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def pack_dataset(folder: Path, archive: Path, fingerprints: dict[str, str]) -> Path:
    """Write a deterministic ZIP and a per-file integrity manifest."""
    import zipfile

    files = {
        name: {"sha256": digest, "size": (folder / name).stat().st_size}
        for name, digest in sorted(fingerprints.items())
    }
    metadata = {"schema_version": 1, "files": files}
    if archive.exists():
        with zipfile.ZipFile(archive) as existing:
            if (
                "_archive.json" in existing.namelist()
                and json.loads(existing.read("_archive.json")) == metadata
            ):
                return archive
    temporary = archive.with_name(archive.name + ".prepare")
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as bundle:
            for name in ["_archive.json", *files]:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = 1
                info.external_attr = 0o100644 << 16
                with bundle.open(info, "w", force_zip64=True) as output:
                    if name == "_archive.json":
                        output.write(
                            (
                                json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n"
                            ).encode()
                        )
                    else:
                        with (folder / name).open("rb") as source:
                            shutil.copyfileobj(source, output, 1024 * 1024)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def unpack_dataset(archive: Path, folder: Path) -> list[Path]:
    """Extract ZIP inputs with path, symlink and content-integrity checks."""
    import hashlib
    import zipfile
    from pathlib import PurePosixPath
    from uuid import uuid4

    outputs = []
    with zipfile.ZipFile(archive) as bundle:
        entries = [
            item
            for item in bundle.infolist()
            if not item.is_dir() and item.filename != "_archive.json"
        ]
        seen = set()
        for item in entries:
            relative = PurePosixPath(item.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in item.filename
                or (item.external_attr >> 16) & 0o170000 == 0o120000
                or item.filename in seen
            ):
                raise ValueError(f"Unsafe or duplicate archive entry: {item.filename}")
            if not (folder / item.filename).resolve().is_relative_to(folder.resolve()):
                raise ValueError(f"Archive entry escapes its dataset: {item.filename}")
            seen.add(item.filename)
        expected = (
            json.loads(bundle.read("_archive.json"))["files"]
            if "_archive.json" in bundle.namelist()
            else None
        )
        if expected is not None and set(expected) != seen:
            raise ValueError("Archive file list differs from its integrity manifest")
        for item in entries:
            target = folder / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + "." + uuid4().hex + ".prepare")
            digest = hashlib.sha256()
            try:
                with bundle.open(item) as source, temporary.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        output.write(block)
                if expected is not None and (
                    digest.hexdigest() != expected[item.filename]["sha256"]
                    or temporary.stat().st_size != expected[item.filename]["size"]
                ):
                    raise ValueError(f"Archive content hash mismatch: {item.filename}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(target)
    return outputs
