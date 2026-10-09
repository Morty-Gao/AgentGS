"""Read-only data paths visible inside a data_only sandbox.

Guest paths keep the stored relative layout. Answer keys, the tool graph and raw
dumps that contain those keys stay on the host.
"""

from pathlib import Path

DATA_DIRECTORIES = (
    "biodeeptime",
    "ccs",
    "climate",
    "deep_earth",
    "energy",
    "ocean",
    "reference",
    "earthbench",
    "geoquestions",
)
DATA_FILES = ("mineral.csv", "hazard.jsonl", "gem.jsonl", "gem.jsonl.gz")
OUTPUT_LIMIT = 16 * 1024


def data_mounts(data_root: Path) -> list[tuple[Path, str]]:
    """Bind existing allowlisted files without copying them."""
    root = Path(data_root).resolve()
    found = []
    for name in DATA_DIRECTORIES:
        host = root / name
        if host.is_symlink():
            raise ValueError(f"Data mount must not be a symlink: {host}")
        if host.is_dir():
            found.append((host.resolve(), f"/data/{name}"))
    for name in DATA_FILES:
        host = root / name
        if host.is_symlink():
            raise ValueError(f"Data mount must not be a symlink: {host}")
        if host.is_file():
            found.append((host.resolve(), f"/data/{name}"))
    return found


def mount_manifest(data_root: Path) -> list[str]:
    return [guest for _, guest in data_mounts(data_root)]
