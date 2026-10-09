"""Translate an AgentGS evaluation command into one forked agent runner.

The runner keeps its own loop, prompts, and tools. This module only resolves
llm.toml and the dataset path, then calls that runner.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import typer

from agentgs.cli.parsing import parse_ids, under_project
from agentgs.client.config import load_llm
from agentgs.config.paths import project_root
from agentgs.evaluation.datasets.version import benchmark_dataset


def endpoint(root: Path, config: Path | None) -> dict[str, str | float | int | None]:
    """Read the single configured model. The API key is not included in reports."""
    loaded = load_llm(root, config)
    name, model = loaded.configured_model()
    provider = loaded.providers[model.provider]
    return {
        "model_key": name,
        "model": model.model,
        "base_url": provider.base_url.rstrip("/"),
        "api_key": provider.api_key.get_secret_value(),
        "request_timeout": provider.timeout_seconds,
        "concurrency": provider.max_concurrent_requests,
        "max_output_tokens": model.max_output_tokens,
        "output_token_parameter": provider.output_token_parameter,
    }


def invoke(root: Path, fork: str, module: str, argv: list[str]) -> int:
    source = root / "forks" / fork
    if not (source / f"{module}.py").is_file():
        raise typer.BadParameter(f"Missing agent runner: {source / module}.py")
    spec = importlib.util.spec_from_file_location(
        f"benchmark_{fork.replace('-', '_')}_{module}", source / f"{module}.py"
    )
    runner = importlib.util.module_from_spec(spec)
    previous_path = list(sys.path)
    previous_bridge = sys.modules.pop("geotool_bridge", None)
    sys.path.insert(0, str(source))
    sys.modules[spec.name] = runner
    try:
        spec.loader.exec_module(runner)
        return int(runner.main(argv))
    finally:
        sys.path[:] = previous_path
        sys.modules.pop("geotool_bridge", None)
        if previous_bridge is not None:
            sys.modules["geotool_bridge"] = previous_bridge


def launch(
    *,
    agent: str,
    fork: str,
    module: str,
    project: Path,
    output: Path,
    dataset: Path | None,
    llm: Path | None,
    ids: str,
    concurrency: int | None,
    extra: list[str],
) -> None:
    root = project_root(project)
    settings = endpoint(root, under_project(root, llm) if llm else None)
    questions = benchmark_dataset(root) if dataset is None else under_project(root, dataset)
    destination = under_project(root, output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    workers = concurrency or settings["concurrency"] or 4
    argv = [
        "--questions",
        str(questions),
        "--output",
        str(destination),
        "--concurrency",
        str(workers),
        "--base-url",
        str(settings["base_url"]),
        "--model",
        str(settings["model"]),
        "--api-key",
        str(settings["api_key"]),
        "--request-timeout",
        str(settings["request_timeout"]),
        "--max-output-tokens",
        str(settings["max_output_tokens"]),
        "--output-token-parameter",
        str(settings["output_token_parameter"]),
        *extra,
    ]
    selected = parse_ids(ids)
    if selected:
        argv.append("--ids")
        argv.extend(str(item) for item in sorted(selected))
    code = invoke(root, fork, module, argv)
    written = 0
    if destination.is_file():
        written = sum(
            1 for line in destination.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    typer.echo(
        json.dumps(
            {
                "agent": agent,
                "model_key": settings["model_key"],
                "model": settings["model"],
                "questions": str(questions),
                "output": str(destination),
                "records": written,
                "concurrency": int(workers),
                "exit_code": code,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise typer.Exit(code)
