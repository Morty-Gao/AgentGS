"""Expand the retained GEM gzip input without installing project dependencies."""

import argparse
import gzip
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def decompress(source: Path, target: Path) -> Path:
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("Source and output paths must differ")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with gzip.open(source, "rb") as incoming:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", suffix=".prepare", delete=False
            ) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(incoming, output, 1024 * 1024)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "data/gem.jsonl.gz")
    parser.add_argument("--output", type=Path, default=REPO / "data/gem.jsonl")
    args = parser.parse_args()
    try:
        target = decompress(args.source, args.output)
    except (OSError, EOFError, ValueError) as error:
        parser.exit(2, f"GEM decompression failed: {error}\n")
    print(f"Expanded GEM dataset: {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
