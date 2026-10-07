"""Run a versioned JSONL suite through the same task runtime used by the CLI/UI."""

import argparse
import asyncio
from pathlib import Path

from agentgs.evaluation.runner import evaluate, load_cases
from agentgs.runtime.service import TaskService


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    args = parser.parse_args()

    async def execute():
        service = TaskService(root)
        try:
            print(
                await evaluate(
                    service, load_cases(root / args.cases), root / "outputs" / "evaluations"
                )
            )
        finally:
            await service.close()

    asyncio.run(execute())


if __name__ == "__main__":
    main()
