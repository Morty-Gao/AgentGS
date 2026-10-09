import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from agentgs.cli.parsing import (
    build_task_input,
    check_planning_mode,
    parse_ids,
    parse_object,
    select_dataset,
    under_project,
)
from agentgs.cli.runtime import (
    AllowProot,
    CodeTimeout,
    Concurrency,
    ModelConcurrency,
    ModelTimeout,
    QueueCapacity,
    RssLimit,
    Sandbox,
    SandboxConcurrency,
    SandboxDiskLimit,
    SandboxMode,
    Timeout,
    ToolBackendOption,
    ToolConcurrency,
    ToolTimeout,
    TotalRssLimit,
    WorkerPython,
    runtime_settings,
)
from agentgs.config.paths import project_root
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskStatus
from agentgs.runtime.service import TaskService
from agentgs.runtime.storage.store import TaskStore

app = typer.Typer(no_args_is_help=True, help="AgentGS task-oriented geoscience agent")
task_app = typer.Typer(no_args_is_help=True, help="Inspect and control stored tasks")
skill_app = typer.Typer(no_args_is_help=True, help="Inspect the frozen skill library")
tool_app = typer.Typer(no_args_is_help=True, help="Inspect registered tools and the sandbox")
dataset_app = typer.Typer(no_args_is_help=True, help="Resolve evaluation datasets")
config_app = typer.Typer(no_args_is_help=True, help="Validate project configuration")
eval_app = typer.Typer(no_args_is_help=True, help="Run cases, benchmarks and offline scoring")
ui_app = typer.Typer(no_args_is_help=True, help="Serve the task UI")
for name, sub in [
    ("task", task_app),
    ("skills", skill_app),
    ("tools", tool_app),
    ("dataset", dataset_app),
    ("config", config_app),
    ("evaluate", eval_app),
    ("ui", ui_app),
]:
    app.add_typer(sub, name=name)
console = Console(stderr=True)


def emit(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        value = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in value
        ]
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


def make_runtime(
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
    tool_backend,
    sandbox_concurrency=None,
    sandbox_disk_limit_mb=None,
) -> RuntimeSettings:
    return runtime_settings(
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
        None if tool_backend is None else tool_backend.value,
        sandbox_concurrency=sandbox_concurrency,
        sandbox_disk_limit_mb=sandbox_disk_limit_mb,
    )


def require_task(store: TaskStore, task_id: str):
    try:
        return store.get_sync(task_id)
    except KeyError:
        raise typer.BadParameter(f"Unknown task: {task_id}") from None


@app.command()
def run(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument()] = "",
    project: Path = typer.Option(Path.cwd(), "--project"),
    tool: str | None = typer.Option(None, "--tool", help="Run this tool directly"),
    arguments: str = typer.Option("{}", "--arguments", help="JSON object of tool arguments"),
    call: list[str] | None = typer.Option(
        None, "--call", help="ToolCall JSON object or list. Repeatable. @file reads JSON"
    ),
    call_id: str | None = typer.Option(None, "--call-id", help="Id for the --tool call"),
    depends_on: list[str] | None = typer.Option(
        None, "--depends-on", help="Call id that the --tool call waits for"
    ),
    task_file: Path | None = typer.Option(
        None, "--task-file", help="TaskInput JSON. Explicit flags override its fields"
    ),
    model: str | None = typer.Option(
        None, "--model", help="llm.toml model key; omit to use the only configured model"
    ),
    effort: str | None = typer.Option(None, "--effort"),
    messages: Path | None = typer.Option(
        None, "--messages", help="Conversation JSON or JSONL history"
    ),
    parallel_tools: bool = typer.Option(False, "--parallel-tools/--no-parallel-tools"),
    skill_id: list[str] | None = typer.Option(None, "--skill-id", help="Restrict skill retrieval"),
    max_model_calls: int | None = typer.Option(None, "--max-model-calls", min=1, max=10000),
    input_resource: str | None = typer.Option(
        None, "--input-resource", help="JSON value, or @file, passed to the task"
    ),
    file: list[str] | None = typer.Option(None, "--file", help="File path available to the task"),
    output_contract: str | None = typer.Option(
        None, "--output-contract", help="Public answer-contract JSON object, or @file"
    ),
    dataset_instructions: str | None = typer.Option(
        None, "--dataset-instructions", help="Supplementary dataset instructions, or @text-file"
    ),
    planning_mode: str | None = typer.Option(
        None,
        "--planning-mode",
        help="tool_only, skill_guided, or data_only. Omit to use config.toml",
    ),
    allow_rule_planning: bool | None = typer.Option(
        None,
        "--allow-rule-planning/--no-rule-planning",
        help="Heuristic plan routing, fallback and rewriting. Omit to use config.toml",
    ),
    plan_attempts: int | None = typer.Option(None, "--plan-attempts", min=1),
    tool_attempts: int | None = typer.Option(None, "--tool-attempts", min=1),
    command_attempts: int | None = typer.Option(
        None, "--command-attempts", min=1, help="Bash commands allowed for data_only"
    ),
    input_domain: str | None = typer.Option(None, "--input-domain"),
    modules: str | None = typer.Option(
        None, "--modules", help="Module overrides as JSON or @file; skill_guided only"
    ),
    ablation: list[str] | None = typer.Option(
        None, "--ablation", help="Repeatable A1–A6 or BIPARTITE preset; skill_guided only"
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
    tool_backend: ToolBackendOption = None,
    sandbox_concurrency: SandboxConcurrency = None,
    sandbox_disk_limit_mb: SandboxDiskLimit = None,
):
    """Run a question or explicit tool calls. Explicit calls need no model credentials."""
    root = project_root(project)
    task = build_task_input(
        ctx,
        task_file=task_file,
        question=question,
        tool=tool,
        arguments=arguments,
        call=call,
        call_id=call_id,
        depends_on=depends_on,
        model=model,
        effort=effort,
        messages=messages,
        parallel_tools=parallel_tools,
        skill_id=skill_id,
        max_model_calls=max_model_calls,
        input_resource=input_resource,
        file=file,
        output_contract=output_contract,
        dataset_instructions=dataset_instructions,
        planning_mode=planning_mode,
        allow_rule_planning=allow_rule_planning,
        plan_attempts=plan_attempts,
        tool_attempts=tool_attempts,
        command_attempts=command_attempts,
        input_domain=input_domain,
        modules=modules,
        ablation=ablation,
    )
    settings = make_runtime(
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
        tool_backend,
        sandbox_concurrency=sandbox_concurrency,
        sandbox_disk_limit_mb=sandbox_disk_limit_mb,
    )

    async def execute():
        async with TaskService(root, settings=settings) as service:
            record = await service.submit(task)
            return await service.wait(record.spec.id)

    record = asyncio.run(execute())
    emit(record)
    if record.status != TaskStatus.SUCCEEDED:
        raise typer.Exit(1)


@task_app.command("list")
def list_tasks(project: Path = typer.Option(Path.cwd(), "--project")):
    """List tasks stored under outputs/runtime."""
    rows = TaskStore(project_root(project) / "outputs" / "runtime").list_sync()
    table = Table("Task", "Status", "Question")
    for row in rows:
        table.add_row(row.spec.id, row.status.value, row.spec.input.question[:70])
    console.print(table)


@task_app.command("show")
def show_task(task_id: str, project: Path = typer.Option(Path.cwd(), "--project")):
    """Print one stored task record."""
    emit(require_task(TaskStore(project_root(project) / "outputs" / "runtime"), task_id))


@task_app.command("events")
def task_events(
    task_id: str,
    project: Path = typer.Option(Path.cwd(), "--project"),
    after: int = typer.Option(0, "--after", min=0, help="Return events with a later sequence"),
):
    """Print stored task events."""
    store = TaskStore(project_root(project) / "outputs" / "runtime")
    require_task(store, task_id)
    emit(store.events_sync(task_id, after))


@task_app.command("wait")
def wait_task(task_id: str, project: Path = typer.Option(Path.cwd(), "--project")):
    """Wait until a stored task reaches a terminal status."""
    root = project_root(project)

    async def execute():
        async with TaskService(root) as service:
            try:
                return await service.wait(task_id)
            except KeyError:
                raise typer.BadParameter(f"Unknown task: {task_id}") from None

    emit(asyncio.run(execute()))


@task_app.command("cancel")
def cancel_task(
    task_id: str,
    project: Path = typer.Option(Path.cwd(), "--project"),
    url: str = typer.Option("http://127.0.0.1:8000", "--url"),
    local: bool = typer.Option(
        False, "--local", help="Cancel through the local task store instead of the UI"
    ),
):
    """Cancel a task. The default sends the request to the running UI."""
    if not local:
        import httpx

        response = httpx.post(url.rstrip("/") + "/api/tasks/" + task_id + "/cancel")
        response.raise_for_status()
        emit(response.json())
        return
    root = project_root(project)

    async def execute():
        async with TaskService(root) as service:
            try:
                return await service.cancel(task_id)
            except KeyError:
                raise typer.BadParameter(f"Unknown task: {task_id}") from None

    emit(asyncio.run(execute()))


@skill_app.command("list")
def list_skills():
    """Print the builtin skill specs."""
    from agentgs.skills.registry import builtin_skills

    emit([skill.spec.model_dump() for skill in builtin_skills().values()])


@skill_app.command("show")
def show_skill(skill_id: str):
    """Print one builtin skill spec."""
    from agentgs.skills.registry import builtin_skills

    try:
        skill = builtin_skills()[skill_id]
    except KeyError:
        raise typer.BadParameter(f"Unknown skill: {skill_id}") from None
    emit(skill.spec)


@skill_app.command("plan")
def plan_skills(
    question: str = typer.Argument(...),
    inferred_tool: list[str] | None = typer.Option(None, "--inferred-tool"),
    proposed_tool: list[str] | None = typer.Option(
        None, "--proposed-tool", help="Tool chain to ground instead of selecting one"
    ),
    mapping_extensions: bool = typer.Option(True, "--mapping-extensions/--no-mapping-extensions"),
    rs_local_contracts: bool = typer.Option(
        True,
        "--rs-local-contracts/--no-rs-local-contracts",
        help="Attach remote-sensing skill contracts",
    ),
    summary: bool = typer.Option(False, "--summary"),
):
    """Inspect the frozen skill algorithm. This does not call models or execute tools."""
    from agentgs.agent.planning.grounding import ground_plan
    from agentgs.agent.planning.skills import builtin_planner
    from agentgs.runtime.tools.registry import builtin_tools

    planner = builtin_planner(
        mapping_extensions=mapping_extensions, rs_local_contracts=rs_local_contracts
    )
    inferred = inferred_tool or []
    plan = planner.plan(question, inferred)
    available = list(builtin_tools())
    grounded = ground_plan(
        planner.graph,
        plan,
        available,
        inferred,
        proposed_chain=list(proposed_tool) if proposed_tool else None,
    )
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
                "rs_local_contracts": rs_local_contracts,
                "constraints": [
                    {"skill_id": row["skill_id"], "status": row["status"]}
                    for row in plan["constraints"]
                ],
            }
        )
    else:
        emit(
            {
                "execution_status": "not_executed",
                "rs_local_contracts": rs_local_contracts,
                "skill_plan": plan,
                "grounding": grounded,
            }
        )


def _tools(internal: bool):
    from agentgs.runtime.tools.registry import builtin_tools, internal_operations

    tools = dict(builtin_tools())
    if internal:
        tools.update(internal_operations())
    return tools


@tool_app.command("list")
def list_tools(
    internal: bool = typer.Option(
        False, "--internal/--no-internal", help="Include bash and ExecutePython"
    ),
):
    """Print registered tool specs."""
    emit([tool.spec.model_dump() for tool in _tools(internal).values()])


@tool_app.command("show")
def show_tool(tool_id: str):
    """Print one tool spec, including internal operations."""
    try:
        tool = _tools(True)[tool_id]
    except KeyError:
        raise typer.BadParameter(f"Unknown tool: {tool_id}") from None
    emit(tool.spec)


@tool_app.command("doctor")
def doctor(
    project: Path = typer.Option(Path.cwd(), "--project"),
    code_timeout: CodeTimeout = None,
    sandbox: Sandbox = SandboxMode.AUTO,
    allow_proot: AllowProot = False,
    worker_python: WorkerPython = None,
    tool_backend: ToolBackendOption = None,
    sandbox_concurrency: SandboxConcurrency = None,
    sandbox_disk_limit_mb: SandboxDiskLimit = None,
):
    """Report sandbox binaries and the backend the current settings would select."""
    from agentgs.runtime.sandbox.base import CodeExecutor

    root = project_root(project)
    settings = make_runtime(
        None,
        None,
        None,
        None,
        code_timeout,
        sandbox,
        allow_proot,
        worker_python,
        None,
        None,
        None,
        None,
        None,
        tool_backend,
        sandbox_concurrency=sandbox_concurrency,
        sandbox_disk_limit_mb=sandbox_disk_limit_mb,
    )

    async def probe():
        executor = CodeExecutor(root, root / "outputs" / "doctor", settings)
        rows = {}
        for name in ("bwrap", "proot", "node"):
            path = executor.locate(name)
            rows[name] = {"path": path, "available": bool(path)}
        report = {
            "configured": settings.sandbox,
            "allow_proot": settings.allow_proot,
            "worker_python": settings.worker_python,
            "tool_backend": settings.tool_backend,
        }
        try:
            described = await executor.describe_shell(root / "data")
        except Exception as exc:
            message = str(exc)[:500]
            report.update(functional=False, diagnostic=message)
            rows["bwrap"]["functional"] = False
            rows["bwrap"]["diagnostic"] = message
        else:
            report.update(
                functional=True,
                selected=described["kind"],
                binary=None if described["binary"] is None else str(described["binary"]),
                rootfs=None if described["rootfs"] is None else str(described["rootfs"]),
                isolation=described["isolation"],
                data_mount=described["data_mount"],
                network_isolated=described["network_isolated"],
                private_proc=described["private_proc"],
                mounts=[str(item) for item in described["mounts"]],
            )
            if described["kind"] == "bwrap":
                rows["bwrap"]["functional"] = True
        rows["sandbox"] = report
        rows["memory"] = {"enforcement": "RSS watchdog", "cgroup_hard_limit": False}
        return rows

    emit(asyncio.run(probe()))


def config_report(root: Path) -> dict:
    from agentgs.client.config import load_llm
    from agentgs.config.algorithm import load_algorithm

    llm = load_llm(root)
    return {
        "models": list(llm.models),
        "llm": {
            "schema_version": llm.schema_version,
            "providers": [
                {
                    "name": name,
                    "protocol": provider.protocol,
                    "base_url": provider.base_url,
                    "timeout_seconds": provider.timeout_seconds,
                    "max_attempts": provider.max_attempts,
                    "output_token_parameter": provider.output_token_parameter,
                    "max_concurrent_requests": provider.max_concurrent_requests,
                    "requests_per_minute": provider.requests_per_minute,
                    "tokens_per_minute": provider.tokens_per_minute,
                }
                for name, provider in llm.providers.items()
            ],
            "models": [
                {
                    "name": name,
                    "provider": model.provider,
                    "model": model.model,
                    "context_window": model.context_window,
                    "max_output_tokens": model.max_output_tokens,
                    "default_effort": model.default_effort,
                    "supported_efforts": model.supported_efforts,
                    "capabilities": model.capabilities.model_dump(mode="json"),
                }
                for name, model in llm.models.items()
            ],
        },
        "agent": load_algorithm(root).model_dump(mode="json"),
        "runtime": RuntimeSettings().model_dump(mode="json"),
    }


@config_app.command("validate")
def validate_config(project: Path = typer.Option(Path.cwd(), "--project")):
    """Load llm.toml and config.toml and print the effective defaults."""
    emit(config_report(project_root(project)))


@config_app.command("show")
def show_config(project: Path = typer.Option(Path.cwd(), "--project")):
    """Print the same configuration report as validate."""
    emit(config_report(project_root(project)))


@dataset_app.command("list")
def list_datasets():
    """List registered evaluation datasets."""
    from agentgs.evaluation.datasets import Dataset

    emit(
        [
            {"name": name, "domain_count": kind.domain_count}
            for name, kind in sorted(Dataset.registry.items())
        ]
    )


@dataset_app.command("path")
def dataset_path(project: Path = typer.Option(Path.cwd(), "--project")):
    """Print the prepared GeoToolEval dataset path."""
    from agentgs.evaluation.datasets.version import benchmark_dataset

    typer.echo(str(benchmark_dataset(project_root(project))))


@eval_app.command("run")
def evaluate_command(
    cases: Path,
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(Path("outputs/evaluations"), "--output"),
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
    tool_backend: ToolBackendOption = None,
    sandbox_concurrency: SandboxConcurrency = None,
    sandbox_disk_limit_mb: SandboxDiskLimit = None,
):
    """Run caller-supplied EvaluationCase rows. Gold stays out of the runtime request."""
    from agentgs.benchmark.cases import evaluate, load_cases

    root = project_root(project)
    settings = make_runtime(
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
        tool_backend,
        sandbox_concurrency=sandbox_concurrency,
        sandbox_disk_limit_mb=sandbox_disk_limit_mb,
    )

    async def execute():
        async with TaskService(root, settings=settings) as service:
            return await evaluate(service, load_cases(cases), under_project(root, output))

    typer.echo(str(asyncio.run(execute())))


@eval_app.command("benchmark")
def benchmark_command(
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(Path("outputs/evaluations/geotooleval"), "--output"),
    dataset: Path | None = typer.Option(
        None, "--dataset", help="Dataset file. Omit to use the prepared GeoToolEval JSONL"
    ),
    dataset_name: str | None = typer.Option(
        None, "--dataset-name", help="Registered dataset name. A path alone is GeoToolEval"
    ),
    ids: str = typer.Option("", "--ids", help="Comma-separated example ids. Omit to run all"),
    model: str | None = typer.Option(
        None, "--model", help="llm.toml model key; omit to use the only configured model"
    ),
    retry_failed: bool = typer.Option(False, "--retry-failed/--no-retry-failed"),
    benchmark_scope: str | None = typer.Option(
        None,
        "--benchmark-scope",
        help="full or common_data; use the same scope for every compared mode",
    ),
    planning_mode: str = typer.Option("skill_guided", "--planning-mode"),
    modules: str | None = typer.Option(None, "--modules", help="Module overrides as JSON or @file"),
    ablation: list[str] | None = typer.Option(
        None, "--ablation", help="Repeatable A1–A6 or BIPARTITE preset"
    ),
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
    command_attempts: int | None = typer.Option(
        None,
        "--command-attempts",
        min=1,
        help="Bash commands allowed during data_only exploration",
    ),
    answer_contracts: bool = typer.Option(
        True,
        "--answer-contracts/--no-answer-contracts",
        help="Attach public answer-format contracts from the prepared dataset",
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
    tool_backend: ToolBackendOption = None,
    sandbox_concurrency: SandboxConcurrency = None,
    sandbox_disk_limit_mb: SandboxDiskLimit = None,
):
    """Run a dataset on the agent and keep every attempt."""
    from agentgs.benchmark.runner import run_benchmark
    from agentgs.runtime.answer_contracts import AnswerContracts

    root = project_root(project)
    mode = check_planning_mode(planning_mode)
    source = select_dataset(root, dataset, dataset_name)
    settings = make_runtime(
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
        tool_backend,
        sandbox_concurrency=sandbox_concurrency,
        sandbox_disk_limit_mb=sandbox_disk_limit_mb,
    )

    async def execute():
        async with TaskService(root, settings=settings) as service:
            return await run_benchmark(
                service,
                source,
                under_project(root, output),
                ids=parse_ids(ids),
                model=model,
                retry_failed=retry_failed,
                benchmark_scope=benchmark_scope,
                planning_mode=mode,
                modules=parse_object(modules, "--modules") if modules else None,
                ablations=ablation or [],
                allow_rule_planning=allow_rule_planning,
                plan_attempts=plan_attempts,
                tool_attempts=tool_attempts,
                command_attempts=command_attempts,
                contract_provider=AnswerContracts(root) if answer_contracts else None,
            )

    report = asyncio.run(execute())
    emit({key: value for key, value in report.items() if key != "results"})


@eval_app.command("score")
def score_benchmark(
    predictions: Path,
    benchmark_scope: str = typer.Option("full", "--benchmark-scope"),
    project: Path = typer.Option(Path.cwd(), "--project"),
    output: Path = typer.Option(Path("outputs/evaluations/scored.json"), "--output"),
    dataset: Path | None = typer.Option(None, "--dataset"),
    dataset_name: str | None = typer.Option(None, "--dataset-name"),
    allow_partial: bool = typer.Option(False, "--allow-partial/--no-allow-partial"),
):
    """Score saved model answers. This does not call the model."""
    from agentgs.benchmark.score import score_predictions

    root = project_root(project)
    report = score_predictions(
        select_dataset(root, dataset, dataset_name),
        predictions,
        under_project(root, output),
        allow_partial=allow_partial,
        benchmark_scope=benchmark_scope,
        project=root,
    )
    emit({key: value for key, value in report.items() if key != "results"})


@ui_app.command("serve")
def serve(
    project: Path = typer.Option(Path.cwd(), "--project"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    log_level: str = typer.Option("info", "--log-level"),
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
    tool_backend: ToolBackendOption = None,
    sandbox_concurrency: SandboxConcurrency = None,
    sandbox_disk_limit_mb: SandboxDiskLimit = None,
):
    """Serve the task API and frontend."""
    import uvicorn

    from agentgs.visualization.backend.app import create_app

    settings = make_runtime(
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
        tool_backend,
        sandbox_concurrency=sandbox_concurrency,
        sandbox_disk_limit_mb=sandbox_disk_limit_mb,
    )
    uvicorn.run(
        create_app(project_root(project), settings=settings),
        host=host,
        port=port,
        log_level=log_level,
    )


@app.command("wsl")
def wsl_run(
    arguments: list[str] = typer.Argument(...),
    project: str = typer.Option(
        ..., "--project", help="Absolute project path inside the WSL distribution"
    ),
    distribution: str = typer.Option("Ubuntu-24.04", "--distribution"),
):
    """From Windows, delegate a command after -- to the project's WSL runtime."""
    from agentgs.runtime.backends.wsl import run

    raise typer.Exit(run(project, arguments, distribution))


def main():
    try:
        app()
    except (ValueError, FileNotFoundError, ValidationError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(2) from None
