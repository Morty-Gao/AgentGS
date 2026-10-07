"""Export descriptors from the executable Skill registry."""

import argparse
import json
from pathlib import Path

from agentgs.skills.registry import builtin_skills


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/indexes/skills.json")
    args = parser.parse_args()
    path = root / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [s.spec.model_dump() for s in builtin_skills().values()], ensure_ascii=False, indent=2
        )
    )
    print(path)


if __name__ == "__main__":
    main()
