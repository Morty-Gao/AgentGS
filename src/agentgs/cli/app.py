import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from agentgs.config.paths import project_root
from agentgs.contracts.models import TaskInput, TaskStatus, ToolCall
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
):
    """Run a question or a typed tool call. Explicit calls require no LLM credentials."""
    root = project_root(project)
    calls = [ToolCall(tool=tool, arguments=json.loads(arguments))] if tool else []
    task = TaskInput(question=question, calls=calls, model=model, effort=effort)

    async def execute():
        service = TaskService(root)
        try:
            record = service.submit(task)
            return await service.wait(record.spec.id)
        finally:
            await service.close()

    record = asyncio.run(execute())
    emit(record)
    if record.status != TaskStatus.SUCCEEDED:
        raise typer.Exit(1)


@task_app.command("list")
def list_tasks(project: Path = typer.Option(Path.cwd(), "--project")):
    rows = TaskStore(project_root(project) / "outputs" / "runtime").list()
    table = Table("Task", "Status", "Question")
    for row in rows:
        table.add_row(row.spec.id, row.status.value, row.spec.input.question[:70])
    console.print(table)


@task_app.command("show")
def show_task(task_id: str, project: Path = typer.Option(Path.cwd(), "--project")):
    emit(TaskStore(project_root(project) / "outputs" / "runtime").get(task_id))


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
    from agentgs.config.settings import load_runtime

    root = project_root(project)
    llm = load_llm(root)
    emit(
        {"models": list(llm.models), "roles": llm.roles, "runtime": load_runtime(root).model_dump()}
    )


@eval_app.command("run")
def evaluate_command(cases: Path, project: Path = typer.Option(Path.cwd(), "--project")):
    from agentgs.evaluation.runner import evaluate, load_cases

    root = project_root(project)

    async def execute():
        service = TaskService(root)
        try:
            return await evaluate(service, load_cases(cases), root / "outputs" / "evaluations")
        finally:
            await service.close()

    typer.echo(str(asyncio.run(execute())))


@ui_app.command("serve")
def serve(
    project: Path = typer.Option(Path.cwd(), "--project"), host: str = "127.0.0.1", port: int = 8000
):
    import uvicorn

    from agentgs.visualization.backend.app import create_app

    uvicorn.run(create_app(project_root(project)), host=host, port=port)


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
