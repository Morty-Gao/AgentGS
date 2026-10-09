import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from agentgs.cli.runtime import (
    AllowProot,
    CodeTimeout,
    Concurrency,
    ModelConcurrency,
    ModelTimeout,
    QueueCapacity,
    RssLimit,
    Sandbox,
    SandboxMode,
    Timeout,
    ToolConcurrency,
    ToolTimeout,
    TotalRssLimit,
    WorkerPython,
    runtime_settings,
)
from agentgs.config.paths import project_root
from agentgs.contracts.models import TaskInput, TaskStatus, ToolCall
from agentgs.messages import Conversation
from agentgs.runtime.service import TaskService
from agentgs.runtime.storage.store import TaskStore

app = typer.Typer(no_args_is_help=True, help="AgentGS task-oriented geoscience agent")
task_app = typer.Typer()
skill_app = typer.Typer()
tool_app = typer.Typer()
config_app = typer.Typer()
eval_app = typer.Typer()
ui_app = typer.Typer()
for name, sub in [
    ("task", task_app),
    ("skills", skill_app),
    ("tools", tool_app),
    ("config", config_app),
    ("evaluate", eval_app),
    ("ui", ui_app),
]:
    app.add_typer(sub, name=name)
console = Console(stderr=True)


def emit(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


@app.command()
def run(
    question: Annotated[str, typer.Argument()] = "",
    project: Path = typer.Option(Path.cwd(), "--project"),
    tool: str | None = typer.Option(None, "--tool"),
    arguments: str = typer.Option("{}", "--arguments"),
    model: str | None = typer.Option(None, "--model"),
    effort: str | None = typer.Option(None, "--effort"),
    messages: Path | None = typer.Option(
        None, "--messages", help="Conversation JSON or JSONL history"
    ),
    concurrency: Concurrency = None,
    timeout: Timeout = None,
    rss_limit_mb: RssLimit = None,
    max_total_rss_mb: TotalRssLimit = None,
    code_timeout: CodeTimeout = None,
    sandbox: Sandbox = SandboxMode.AUTO,
    allow_proot: AllowProot = False,
    worker_python: WorkerPython = None,
    model_concurrency: ModelConcurrency = None,
    tool_concurrency: ToolConcurrency = None,
    queue_capacity: QueueCapacity = None,
    model_timeout: ModelTimeout = None,
    tool_timeout: ToolTimeout = None,
):
    """Run a question or a typed tool call. Explicit calls require no LLM credentials."""
    root = project_root(project)
    calls = [ToolCall(tool=tool, arguments=json.loads(arguments))] if tool else []
    history = []
    if messages:
        if messages.suffix == ".jsonl":
            history = Conversation.from_jsonl(messages.read_text(encoding="utf-8")).messages
        else:
            data = json.loads(messages.read_text(encoding="utf-8"))
            history = (
                Conversation(messages=data).messages
                if isinstance(data, list)
                else Conversation.model_validate(data).messages
            )
    task = TaskInput(
        question=question,
        calls=calls,
        model=model,
        effort=effort,
        messages=history,
    )

    async def execute():
        service = TaskService(
            root,
            settings=runtime_settings(
                concurrency,
                timeout,
                rss_limit_mb,
                max_total_rss_mb,
                code_timeout,
                sandbox,
                allow_proot,
                worker_python,
                model_concurrency,
                tool_concurrency,
                queue_capacity,
                model_timeout,
                tool_timeout,
            ),
        )
        try:
            record = await service.submit(task)
            return await service.wait(record.spec.id)
        finally:
            await service.close()

    record = asyncio.run(execute())
    emit(record)
    if record.status != TaskStatus.SUCCEEDED:
        raise typer.Exit(1)


@task_app.command("list")
def list_tasks(project: Path = typer.Option(Path.cwd(), "--project")):
    rows = TaskStore(project_root(project) / "outputs" / "runtime").list_sync()
    table = Table("Task", "Status", "Question")
    for row in rows:
        table.add_row(row.spec.id, row.status.value, row.spec.input.question[:70])
    console.print(table)


@task_app.command("show")
def show_task(task_id: str, project: Path = typer.Option(Path.cwd(), "--project")):
    emit(TaskStore(project_root(project) / "outputs" / "runtime").get_sync(task_id))


@task_app.command("cancel")
def cancel_task(task_id: str, url: str = typer.Option("http://127.0.0.1:8000", "--url")):
    """Cancel through the owning running UI/backend service."""
    import httpx

    response = httpx.post(url.rstrip("/") + "/api/tasks/" + task_id + "/cancel")
    response.raise_for_status()
    emit(response.json())


@skill_app.command("list")
def list_skills():
    from agentgs.skills.registry import builtin_skills

    emit([s.spec.model_dump() for s in builtin_skills().values()])


@skill_app.command("plan")
def plan_skills(
    question: str = typer.Argument(...),
    inferred_tool: list[str] | None = typer.Option(None, "--inferred-tool"),
    mapping_extensions: bool = typer.Option(True, "--mapping-extensions/--no-mapping-extensions"),
    summary: bool = typer.Option(False, "--summary"),
):
    """Inspect the frozen Skill algorithm. This does not call models or execute tools."""
    from agentgs.agent.planning.grounding import ground_plan
    from agentgs.agent.planning.skills import builtin_planner
    from agentgs.runtime.tools.registry import builtin_tools

    planner = builtin_planner(mapping_extensions=mapping_extensions)
    plan = planner.plan(question, inferred_tool or [])
    available = list(builtin_tools())
    grounded = ground_plan(planner.graph, plan, available)
    if summary:
        emit(
            {
                "execution_status": "not_executed",
                "baseline_id": "geotoolagent-geotooleval-101-snapshot-20261008",
                "selected_skills": [step["skill_id"] for step in plan["steps"]],
                "candidates": plan["candidates"],
                "capabilities": plan["capability_expansion"]["capability_ids"],
                "available_tools": available,
                "grounded_tools": grounded["tool_chain"],
                "constraints": [
                    {"skill_id": row["skill_id"], "status": row["status"]}
                    for row in plan["constraints"]
                ],
            }
        )
    else:
        emit({"execution_status": "not_executed", "skill_plan": plan, "grounding": grounded})


@tool_app.command("list")
def list_tools():
    from agentgs.runtime.tools.registry import builtin_tools

    emit([t.spec.model_dump() for t in builtin_tools().values()])


@tool_app.command("doctor")
def doctor(project: Path = typer.Option(Path.cwd(), "--project")):
    import os
    import shutil
    import subprocess

    root = project_root(project)
    rows = {}
    for name in ["bwrap", "proot", "node"]:
        local = root / "tools" / "bin" / name
        path = str(local) if local.exists() else shutil.which(name)
        rows[name] = {"path": path, "available": bool(path)}
    if rows["bwrap"]["path"] and os.name != "nt":
        probe = subprocess.run(
            [
                rows["bwrap"]["path"],
                "--unshare-all",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "/usr/bin/true",
            ],
            capture_output=True,
            timeout=5,
        )
        rows["bwrap"]["functional"] = probe.returncode == 0
        rows["bwrap"]["diagnostic"] = probe.stderr.decode(errors="replace")[:500]
    rows["memory"] = {"enforcement": "RSS watchdog", "cgroup_hard_limit": False}
    emit(rows)


@config_app.command("validate")
def validate_config(project: Path = typer.Option(Path.cwd(), "--project")):
    from agentgs.client.config import load_llm
    from agentgs.config.algorithm import load_algorithm

    root = project_root(project)
    llm = load_llm(root)
    emit(
        {"models": list(llm.models), "roles": llm.roles, "agent": load_algorithm(root).model_dump()}
    )


@eval_app.command("run")
def evaluate_command(
    cases: Path,
    project: Path = typer.Option(Path.cwd(), "--project"),
    concurrency: Concurrency = None,
    timeout: Timeout = None,
    rss_limit_mb: RssLimit = None,
    max_total_rss_mb: TotalRssLimit = None,
    code_timeout: CodeTimeout = None,
    sandbox: Sandbox = SandboxMode.AUTO,
    allow_proot: AllowProot = False,
    worker_python: WorkerPython = None,
    model_concurrency: ModelConcurrency = None,
    tool_concurrency: ToolConcurrency = None,
    queue_capacity: QueueCapacity = None,
    model_timeout: ModelTimeout = None,
    tool_timeout: ToolTimeout = None,
):
    from agentgs.evaluation.runner import evaluate, load_cases

    root = project_root(project)

    async def execute():
        service = TaskService(
            root,
            settings=runtime_settings(
                concurrency,
                timeout,
                rss_limit_mb,
                max_total_rss_mb,
                code_timeout,
                sandbox,
                allow_proot,
                worker_python,
                model_concurrency,
                tool_concurrency,
                queue_capacity,
                model_timeout,
                tool_timeout,
            ),
        )
        try:
            return await evaluate(service, load_cases(cases), root / "outputs" / "evaluations")
        finally:
            await service.close()

    typer.echo(str(asyncio.run(execute())))


@eval_app.command("benchmark")
def benchmark_command(
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(Path("outputs/evaluations/geotooleval"), "--output"),
    ids: str = typer.Option("", "--ids"),
    retry_failed: bool = typer.Option(False, "--retry-failed"),
    planning_mode: str = typer.Option("skill_guided", "--planning-mode"),
    allow_rule_planning: bool = typer.Option(
        False,
        "--allow-rule-planning/--no-rule-planning",
        help="Allow heuristic plan routing, fallback and rewriting; model calls still occur",
    ),
    plan_attempts: int | None = typer.Option(
        None,
        "--plan-attempts",
        min=1,
        help="Total planning attempts, including replanning",
    ),
    tool_attempts: int | None = typer.Option(
        None,
        "--tool-attempts",
        min=1,
        help="Argument/execution attempts per tool instance",
    ),
    concurrency: Concurrency = None,
    timeout: Timeout = None,
    rss_limit_mb: RssLimit = None,
    max_total_rss_mb: TotalRssLimit = None,
    code_timeout: CodeTimeout = None,
    sandbox: Sandbox = SandboxMode.AUTO,
    allow_proot: AllowProot = False,
    worker_python: WorkerPython = None,
    model_concurrency: ModelConcurrency = None,
    tool_concurrency: ToolConcurrency = None,
    queue_capacity: QueueCapacity = None,
    model_timeout: ModelTimeout = None,
    tool_timeout: ToolTimeout = None,
):
    """Run actual Agent requests with conversation metrics; preserve every attempt."""
    from agentgs.evaluation.benchmark import run_benchmark
    from agentgs.evaluation.datasets.version import benchmark_dataset
    from agentgs.runtime.answer_contracts import AnswerContracts

    root = project_root(project)
    chosen = {int(value) for value in ids.split(",") if value.strip()} if ids else None

    async def execute():
        service = TaskService(
            root,
            settings=runtime_settings(
                concurrency,
                timeout,
                rss_limit_mb,
                max_total_rss_mb,
                code_timeout,
                sandbox,
                allow_proot,
                worker_python,
                model_concurrency,
                tool_concurrency,
                queue_capacity,
                model_timeout,
                tool_timeout,
            ),
        )
        try:
            return await run_benchmark(
                service,
                benchmark_dataset(root),
                root / output,
                ids=chosen,
                retry_failed=retry_failed,
                planning_mode=planning_mode,
                allow_rule_planning=allow_rule_planning,
                plan_attempts=plan_attempts,
                tool_attempts=tool_attempts,
                contract_provider=AnswerContracts(root),
            )
        finally:
            await service.close()

    report = asyncio.run(execute())
    emit({key: value for key, value in report.items() if key != "results"})


@eval_app.command("score")
def score_benchmark(
    predictions: Path,
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(Path("outputs/evaluations/scored.json"), "--output"),
    allow_partial: bool = typer.Option(False, "--allow-partial"),
):
    """Score saved original model answers without any new model calls."""
    from agentgs.evaluation.benchmark import score_predictions
    from agentgs.evaluation.datasets.version import benchmark_dataset

    root = project_root(project)
    report = score_predictions(
        benchmark_dataset(root),
        predictions,
        root / output,
        allow_partial=allow_partial,
        project=root,
    )
    emit({key: value for key, value in report.items() if key != "results"})


@ui_app.command("serve")
def serve(
    project: Path = typer.Option(Path.cwd(), "--project"),
    host: str = "127.0.0.1",
    port: int = 8000,
    concurrency: Concurrency = None,
    timeout: Timeout = None,
    rss_limit_mb: RssLimit = None,
    max_total_rss_mb: TotalRssLimit = None,
    code_timeout: CodeTimeout = None,
    sandbox: Sandbox = SandboxMode.AUTO,
    allow_proot: AllowProot = False,
    worker_python: WorkerPython = None,
    model_concurrency: ModelConcurrency = None,
    tool_concurrency: ToolConcurrency = None,
    queue_capacity: QueueCapacity = None,
    model_timeout: ModelTimeout = None,
    tool_timeout: ToolTimeout = None,
):
    import uvicorn

    from agentgs.visualization.backend.app import create_app

    settings = runtime_settings(
        concurrency,
        timeout,
        rss_limit_mb,
        max_total_rss_mb,
        code_timeout,
        sandbox,
        allow_proot,
        worker_python,
        model_concurrency,
        tool_concurrency,
        queue_capacity,
        model_timeout,
        tool_timeout,
    )
    uvicorn.run(create_app(project_root(project), settings=settings), host=host, port=port)


@app.command("wsl")
def wsl_run(
    arguments: list[str] = typer.Argument(...),
    project: str = typer.Option("/home/gaosong/Code/Project/AgentGS", "--project"),
    distribution: str = typer.Option("Ubuntu-24.04", "--distribution"),
):
    """From Windows, delegate a command after -- to the project's WSL runtime."""
    from agentgs.runtime.backends.wsl import run

    raise typer.Exit(run(project, arguments, distribution))


def main():
    try:
        app()
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(2) from None
