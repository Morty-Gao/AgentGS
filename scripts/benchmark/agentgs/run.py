"""Evaluate GeoToolEval with the project AgentGS agent.

This directory is a script entry, not a Python package, so it does not shadow
the ``agentgs`` import. Model, endpoint, and algorithm defaults come from the
project ``llm.toml`` and ``config.toml``. Planning and tool execution stay in
``src/agentgs``.

Generation writes ``predictions.jsonl`` and does not calculate FAS. Score the
file afterwards with ``agentgs evaluate score``.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _agentgs import (  # noqa: E402
    build_identity,
    classify_result,
    contract_digest,
    run_batch,
    select_tasks,
    task_request,
)
from _runtime import atomic_json  # noqa: E402


def main(
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(
        Path("outputs/evaluations/agentgs/predictions.jsonl"),
        "--output",
    ),
    dataset: Path | None = typer.Option(
        None, "--dataset", help="Dataset file. Omit to use the prepared GeoToolEval JSONL"
    ),
    dataset_name: str | None = typer.Option(
        None, "--dataset-name", help="Registered dataset name. A path alone is GeoToolEval"
    ),
    ids: str = typer.Option("", "--ids", help="Comma-separated question ids. Omit to run all"),
    model: str | None = typer.Option(
        None, "--model", help="llm.toml model key; omit to use the only configured model"
    ),
    concurrency: int | None = typer.Option(
        None,
        "--concurrency",
        min=1,
        max=128,
        help="Parallel questions. Omit to use the runtime default",
    ),
    planning_mode: str = typer.Option("skill_guided", "--planning-mode"),
    modules: str | None = typer.Option(None, "--modules", help="Module overrides as JSON or @file"),
    ablation: list[str] | None = typer.Option(
        None, "--ablation", help="Repeatable A1–A6 or BIPARTITE preset"
    ),
    allow_rule_planning: bool = typer.Option(
        False,
        "--allow-rule-planning/--no-rule-planning",
        help="Allow heuristic plan routing, fallback and rewriting",
    ),
    plan_attempts: int | None = typer.Option(None, "--plan-attempts", min=1),
    tool_attempts: int | None = typer.Option(None, "--tool-attempts", min=1),
    command_attempts: int | None = typer.Option(None, "--command-attempts", min=1),
    answer_contracts: bool = typer.Option(True, "--answer-contracts/--no-answer-contracts"),
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
    timeout: float | None = typer.Option(
        None, "--timeout", min=0.001, help="Task timeout in seconds"
    ),
    rss_limit_mb: int | None = typer.Option(None, "--rss-limit-mb", min=64),
    max_total_rss_mb: int | None = typer.Option(None, "--max-total-rss-mb", min=64),
    code_timeout: float | None = typer.Option(None, "--code-timeout", min=0.001, max=300),
    sandbox: str = typer.Option("auto", "--sandbox"),
    allow_proot: bool = typer.Option(False, "--allow-proot/--no-allow-proot"),
    worker_python: Path | None = typer.Option(None, "--worker-python"),
    model_concurrency: int | None = typer.Option(None, "--model-concurrency", min=1, max=128),
    tool_concurrency: int | None = typer.Option(None, "--tool-concurrency", min=1, max=128),
    sandbox_concurrency: int | None = typer.Option(None, "--sandbox-concurrency", min=1, max=128),
    sandbox_disk_limit_mb: int | None = typer.Option(None, "--sandbox-disk-limit-mb", min=1),
    queue_capacity: int | None = typer.Option(None, "--queue-capacity", min=1),
    model_timeout: float | None = typer.Option(None, "--model-timeout", min=0.001),
    tool_timeout: float | None = typer.Option(None, "--tool-timeout", min=0.001),
    tool_backend: str | None = typer.Option(None, "--tool-backend"),
) -> None:
    """Run AgentGS over GeoToolEval and keep one JSONL record per question."""
    from agentgs.cli.parsing import check_planning_mode, parse_object, under_project
    from agentgs.cli.runtime import SandboxMode, ToolBackend, runtime_settings
    from agentgs.client.config import load_llm
    from agentgs.config.paths import project_root
    from agentgs.contracts.async_utils import invoke
    from agentgs.runtime.answer_contracts import AnswerContracts
    from agentgs.runtime.service import TaskService

    root = project_root(project)
    mode = check_planning_mode(planning_mode)
    try:
        parsed_modules = parse_object(modules, "--modules") if modules else None
        source, rows, scope = select_tasks(root, dataset, dataset_name, ids, mode)
        loaded = load_llm(root)
        model_key, selected = loaded.configured_model(model)
        provider = loaded.providers[selected.provider]
        backend = None if tool_backend is None else ToolBackend(tool_backend)
        settings = runtime_settings(
            concurrency,
            timeout,
            rss_limit_mb,
            max_total_rss_mb,
            code_timeout,
            SandboxMode(sandbox),
            allow_proot,
            worker_python,
            model_concurrency,
            tool_concurrency,
            queue_capacity,
            model_timeout,
            tool_timeout,
            None if backend is None else backend.value,
            sandbox_concurrency=sandbox_concurrency,
            sandbox_disk_limit_mb=sandbox_disk_limit_mb,
        )
        contracts = AnswerContracts(root) if answer_contracts else None
        identity = build_identity(
            root,
            source.source,
            planning_mode=mode,
            allow_rule_planning=allow_rule_planning,
            plan_attempts=plan_attempts,
            tool_attempts=tool_attempts,
            command_attempts=command_attempts,
            modules=parsed_modules,
            ablations=ablation,
            answer_contracts=answer_contracts,
            contract_sha256=contract_digest(contracts),
            retries=retries,
            runtime=settings.model_dump(mode="json"),
            model_key=model_key,
            model_id=selected.model,
            base_url=provider.base_url,
            benchmark_scope=scope,
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from None

    @asynccontextmanager
    async def open_runner():
        async with TaskService(root, settings=settings) as service:

            async def execute(row, attempt: Path):
                request = task_request(
                    row,
                    planning_mode=mode,
                    allow_rule_planning=allow_rule_planning,
                    plan_attempts=plan_attempts,
                    tool_attempts=tool_attempts,
                    command_attempts=command_attempts,
                    modules=parsed_modules,
                    ablations=ablation,
                    contract_provider=contracts,
                    model=model_key,
                )
                await invoke(
                    atomic_json, attempt / "public_input.json", request.model_dump(mode="json")
                )
                submitted = await service.submit(request)
                await invoke(
                    atomic_json,
                    attempt / "submission.json",
                    {"question_id": row.question_id, "task_id": submitted.spec.id},
                )
                result = await service.wait(submitted.spec.id)
                await invoke(
                    atomic_json, attempt / "task_record.json", result.model_dump(mode="json")
                )
                events = await service.events(submitted.spec.id)

                def write_trace():
                    (attempt / "events.jsonl").write_text(
                        "".join(event.model_dump_json() + "\n" for event in events),
                        encoding="utf-8",
                    )
                    stderr = root / "outputs/tasks" / submitted.spec.id / "stderr.log"
                    if stderr.is_file():
                        target = attempt / "stderr.log"
                        target.write_bytes(stderr.read_bytes())

                await invoke(write_trace)
                outcome = classify_result(result)
                outcome["task_id"] = submitted.spec.id
                return outcome

            yield execute

    code = run_batch(
        rows=rows,
        output=under_project(root, output),
        identity=identity,
        retries=retries,
        retry_failed=retry_failed,
        concurrency=settings.max_concurrent_tasks,
        opener=open_runner,
    )
    raise typer.Exit(code)


if __name__ == "__main__":
    typer.run(main)
