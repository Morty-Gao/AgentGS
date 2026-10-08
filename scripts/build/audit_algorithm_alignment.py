"""Audit pure-component parity and report pending production algorithm stages."""

import argparse
import hashlib
import json
from pathlib import Path

from agentgs.evaluation.alignment import alignment_report


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save a JSON report; relative to project root")
    parser.add_argument(
        "--require-runtime-parity",
        action="store_true",
        help="Fail if any production stage is unaligned",
    )
    args = parser.parse_args()
    contract = json.loads((root / "docs/algorithm_alignment.json").read_text("utf-8"))
    selection = contract["baseline"]["selection"]
    selection_path = (root / selection["manifest"]).resolve()
    if (
        not selection_path.is_relative_to(root)
        or hashlib.sha256(selection_path.read_bytes()).hexdigest() != selection["sha256"]
    ):
        parser.error("Tool scope differs from the approved selection")
    identity = contract["acceptance"]["snapshot"]
    path = (root / identity["path"]).resolve()
    if (
        not path.is_relative_to(root)
        or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]
    ):
        parser.error("Snapshot bytes differ from the approved algorithm oracle")
    report = alignment_report(contract, json.loads(path.read_text("utf-8")))
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        output = (root / args.output).resolve()
        # Keep audit output out of source, fixtures and the frozen contract.
        if not output.is_relative_to(root / "outputs"):
            parser.error("Audit reports must be written under outputs/")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    failed = not report["component_parity"] or (
        args.require_runtime_parity and not report["runtime_equivalent"]
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
