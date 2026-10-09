"""Run all scientific input preparation commands and retain a complete/partial status report."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _common import REPO, write_json

DOMAINS = (
    "gem",
    "deep_earth",
    "geotoolkg",
    "answer_formats",
    "geoquestions",
    "mineral",
    "climate",
    "energy",
    "ocean",
    "biodeeptime",
    "ccs",
    "remote_sensing",
    "hazard",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO)
    parser.add_argument("--cache-root", type=Path, default=REPO / "data/raw")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--archive-threshold-mb", type=float, default=64)
    args = parser.parse_args()
    results = {}
    for domain in DOMAINS:
        manifest = args.output_root / f"data/metadata/prepare_{domain}.json"
        manifest.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name(f"prepare_{domain}.py")),
            "--output-root",
            str(args.output_root),
            "--cache-root",
            str(args.cache_root),
        ]
        if args.source_root:
            command.extend(["--source-root", str(args.source_root)])
        command.extend(["--archive-threshold-mb", str(args.archive_threshold_mb)])
        if args.offline:
            command.append("--offline")
        result = subprocess.run(command, check=False)
        results[domain] = {
            "exit_code": result.returncode,
            "result": json.loads(manifest.read_text()) if manifest.exists() else None,
        }
    write_json(args.output_root / "data/metadata/prepare_all.json", results)
    raise SystemExit(0 if all(r["exit_code"] == 0 for r in results.values()) else 2)


if __name__ == "__main__":
    main()
