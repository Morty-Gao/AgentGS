"""Download original bytes from versioned manifests, without preprocessing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.request


def repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Cannot find repository root containing pyproject.toml")


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()


def target_path(directory: Path, name: str) -> Path:
    target = (directory / name).resolve()
    if not target.is_relative_to(directory.resolve()) or target == directory.resolve():
        raise ValueError(f"Unsafe manifest path: {name}")
    return target


def validate(path: Path, entry: dict) -> None:
    if "size" in entry and path.stat().st_size != entry["size"]:
        raise ValueError(f"Size mismatch: {path}")
    for algorithm in ("sha256", "md5"):
        if algorithm in entry:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, algorithm).hexdigest()
            if actual != entry[algorithm]:
                raise ValueError(f"{algorithm} mismatch: {path}; expected {entry[algorithm]}, got {actual}")
    if not any(key in entry for key in ("sha256", "md5")):
        raise ValueError(f"Manifest has no content checksum: {entry['path']}")


def acquire(target: Path, entry: dict, source: Path | None, timeout: float, retries: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    urls = entry.get("urls", [])
    if source is None and not urls:
        raise ValueError(entry.get("note", "No public download URL; supply --source-dir"))
    attempts = 1 if source is not None else retries + 1
    for attempt in range(attempts):
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
        os.close(fd)
        part = Path(temporary)
        try:
            if source is not None:
                shutil.copyfile(source, part)
            else:
                url = urls[attempt % len(urls)]
                request = urllib.request.Request(url, headers={"User-Agent": "AgentGS-data-download/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                    length = response.headers.get("Content-Length")
                if length is not None and part.stat().st_size != int(length):
                    raise ValueError(f"Incomplete HTTP response for {entry['path']}")
            validate(part, entry)
            # Another downloader must not silently replace an existing file.
            if target.exists():
                validate(target, entry)
            else:
                os.link(part, target)
            return
        except (OSError, ValueError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError(f"Acquisition failed for {entry['path']}: {error}") from error
            print(f"RETRY {entry['path']}: {error}", file=sys.stderr)
            time.sleep(min(attempt + 1, 3))
        finally:
            part.unlink(missing_ok=True)


def main(dataset: str) -> int:
    manifest_path = Path(__file__).parent / "manifests" / f"{dataset}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=manifest["description"])
    parser.add_argument("--output-dir", default=f"data/{dataset}", help="Output path, relative to repo root by default")
    parser.add_argument("--source-dir", help="Import original files from this directory, using manifest-relative names")
    parser.add_argument("--verify-only", action="store_true", help="Check all files without downloading or importing")
    parser.add_argument("--list", action="store_true", help="Print source manifest without accessing data or network")
    parser.add_argument("--timeout", type=float, default=60, help="HTTP socket timeout in seconds")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0:
        parser.error("--timeout must be positive; --retries must be nonnegative")
    if args.list:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    destination = resolve_path(args.output_dir)
    source_dir = resolve_path(args.source_dir) if args.source_dir else None
    failures = 0
    for entry in manifest["files"]:
        try:
            target = target_path(destination, entry["path"])
            if target.exists():
                validate(target, entry)
                print(f"OK {dataset}/{entry['path']}")
            elif args.verify_only:
                raise FileNotFoundError(f"Missing {target}")
            else:
                source = target_path(source_dir, entry["path"]) if source_dir else None
                acquire(target, entry, source, args.timeout, args.retries)
                print(f"{'IMPORTED' if source else 'DOWNLOADED'} {dataset}/{entry['path']}")
        except (OSError, ValueError, RuntimeError) as error:
            failures += 1
            print(f"ERROR {dataset}/{entry['path']}: {error}", file=sys.stderr)
    print(f"{dataset}: {len(manifest['files']) - failures}/{len(manifest['files'])} files ready")
    return 1 if failures else 0
