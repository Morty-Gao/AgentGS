"""Evaluate GeoToolEval with the Earth-Agent ReAct runner.

Model, endpoint, and request timeout come from llm.toml. The tool loop,
prompts, and tools stay in forks/Earth-Agent.
"""

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bridge import launch  # noqa: E402


def main(
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(
        Path("outputs/evaluations/earthagent/predictions.jsonl"),
        "--output",
    ),
    dataset: Path | None = typer.Option(
        None, "--dataset", help="Dataset file. Omit to use the prepared GeoToolEval JSONL"
    ),
    llm: Path | None = typer.Option(
        None, "--llm", help="llm.toml path. Omit to use the project file"
    ),
    model: str | None = typer.Option(
        None, "--model", help="llm.toml model key; omit to use the only configured model"
    ),
    ids: str = typer.Option("", "--ids", help="Comma-separated question ids. Omit to run all"),
    concurrency: int | None = typer.Option(
        None,
        "--concurrency",
        min=1,
        max=128,
        help="Parallel questions. Omit to use llm.toml max_concurrent_requests, otherwise 4",
    ),
    max_steps: int = typer.Option(
        200, "--max-steps", min=1, help="Tool-call turns for one question"
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Retry failed questions and retain prior attempts"
    ),
    retries: int = typer.Option(
        3,
        "--retries",
        min=0,
        max=10,
        help="Automatic retries per failed question; 3 means at most 4 attempts",
    ),
) -> None:
    """Run Earth-Agent over GeoToolEval and keep one JSONL record per question."""
    launch(
        agent="earthagent",
        fork="Earth-Agent",
        module="run_geotooleval",
        project=project,
        output=output,
        dataset=dataset,
        llm=llm,
        model=model,
        ids=ids,
        concurrency=concurrency,
        extra=[
            "--max-steps",
            str(max_steps),
            "--retries",
            str(retries),
            *(["--retry-failed"] if retry_failed else []),
        ],
    )


if __name__ == "__main__":
    typer.run(main)
