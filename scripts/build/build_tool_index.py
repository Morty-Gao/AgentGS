"""Export tool schemas from the authoritative executable registry."""

import argparse
import json
from pathlib import Path

from agentgs.runtime.tools.registry import builtin_tools


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/indexes/tools.json")
    args = parser.parse_args()
    path = root / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [t.spec.model_dump() for t in builtin_tools().values()], ensure_ascii=False, indent=2
        )
    )
    print(path)


if __name__ == "__main__":
    main()
