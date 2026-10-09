"""Shared CLI options for runtime settings; no project configuration file is read."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from agentgs.config.settings import RuntimeSettings


class SandboxMode(StrEnum):
    AUTO = "auto"
    BWRAP = "bwrap"
    PROOT = "proot"
    DISABLED = "disabled"


class ToolBackend(StrEnum):
    PROCESS = "process"
    THREAD = "thread"


Concurrency = Annotated[
    int | None, typer.Option("--concurrency", min=1, max=128, help="Maximum concurrent tasks")
]
ModelConcurrency = Annotated[int | None, typer.Option("--model-concurrency", min=1, max=128)]
ToolConcurrency = Annotated[int | None, typer.Option("--tool-concurrency", min=1, max=128)]
SandboxConcurrency = Annotated[int | None, typer.Option("--sandbox-concurrency", min=1, max=128)]
SandboxDiskLimit = Annotated[int | None, typer.Option("--sandbox-disk-limit-mb", min=1)]
QueueCapacity = Annotated[int | None, typer.Option("--queue-capacity", min=1)]
ModelTimeout = Annotated[float | None, typer.Option("--model-timeout", min=0.001)]
ToolTimeout = Annotated[float | None, typer.Option("--tool-timeout", min=0.001)]


Timeout = Annotated[
    float | None, typer.Option("--timeout", min=0.001, help="Task timeout in seconds")
]
RssLimit = Annotated[
    int | None, typer.Option("--rss-limit-mb", min=64, help="RSS limit per task in MB")
]
TotalRssLimit = Annotated[
    int | None, typer.Option("--max-total-rss-mb", min=64, help="Total RSS budget in MB")
]
CodeTimeout = Annotated[
    float | None,
    typer.Option("--code-timeout", min=0.001, max=300, help="Code step timeout in seconds"),
]
Sandbox = Annotated[SandboxMode, typer.Option("--sandbox", help="Code sandbox backend")]
AllowProot = Annotated[
    bool,
    typer.Option(
        "--allow-proot/--no-allow-proot", help="Compatibility flag; model code requires isolation"
    ),
]
WorkerPython = Annotated[
    Path | None,
    typer.Option("--worker-python", help="Worker Python path, relative to project or absolute"),
]
ToolBackendOption = Annotated[
    ToolBackend | None,
    typer.Option("--tool-backend", help="Scientific tool execution backend"),
]


def runtime_settings(
    concurrency: int | None,
    timeout: float | None,
    rss_limit_mb: int | None,
    max_total_rss_mb: int | None,
    code_timeout: float | None,
    sandbox: SandboxMode,
    allow_proot: bool,
    worker_python: Path | None,
    model_concurrency: int | None = None,
    tool_concurrency: int | None = None,
    queue_capacity: int | None = None,
    model_timeout: float | None = None,
    tool_timeout: float | None = None,
    tool_backend: str | None = None,
    sandbox_concurrency: int | None = None,
    sandbox_disk_limit_mb: int | None = None,
) -> RuntimeSettings:
    values = RuntimeSettings().model_dump()
    overrides = {
        "max_concurrent_model_calls": model_concurrency,
        "max_concurrent_tool_calls": tool_concurrency,
        "max_concurrent_sandbox_calls": sandbox_concurrency,
        "sandbox_disk_limit_mb": sandbox_disk_limit_mb,
        "max_pending_tasks": queue_capacity,
        "model_timeout_seconds": model_timeout,
        "tool_timeout_seconds": tool_timeout,
        "tool_backend": tool_backend,
        "max_concurrent_tasks": concurrency,
        "default_timeout_seconds": timeout,
        "default_rss_limit_mb": rss_limit_mb,
        "max_total_rss_mb": max_total_rss_mb,
        "code_timeout_seconds": code_timeout,
        "sandbox": sandbox.value,
        "allow_proot": allow_proot,
        "worker_python": str(worker_python) if worker_python is not None else None,
    }
    values.update({name: value for name, value in overrides.items() if value is not None})
    if max_total_rss_mb is None and (concurrency is not None or rss_limit_mb is not None):
        values["max_total_rss_mb"] = max(
            values["max_total_rss_mb"],
            values["max_concurrent_tasks"] * values["default_rss_limit_mb"],
        )
    return RuntimeSettings.model_validate(values)
