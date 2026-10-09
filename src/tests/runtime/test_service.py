import asyncio
import sys
from types import SimpleNamespace

import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskInput, TaskStatus, ToolCall
from agentgs.runtime.service import TaskService
from agentgs.skills.definitions import load_tool_scope


def catalog_query():
    return TaskInput(
        calls=[ToolCall(tool="DeepEarthQueryUSGSCatalog", arguments={"min_magnitude": 8})]
    )


async def test_real_workers_queue_artifacts_and_events(project):
    service = TaskService(project, RuntimeSettings(max_concurrent_tasks=1))
    try:
        first = service.submit(catalog_query())
        second = service.submit(catalog_query())
        a = await service.wait(first.spec.id)
        b = await service.wait(second.spec.id)
        assert a.status == b.status == TaskStatus.SUCCEEDED
        assert a.result.tool_results[0].data["count"] == 2
        ref = a.result.tool_results[0].artifacts[0]
        assert (project / "outputs" / "tasks" / a.spec.id / "artifacts" / ref.path).exists()
        events = service.events(a.spec.id)
        assert [e.sequence for e in events] == list(range(1, len(events) + 1))
        assert service.events(b.spec.id)[1].time >= events[-1].time
    finally:
        await service.close()


async def test_cancel_queued_task_and_timeout(project):
    service = TaskService(project, RuntimeSettings(max_concurrent_tasks=1))
    try:
        task = service.submit(catalog_query())
        cancelled = await service.cancel(task.spec.id)
        assert cancelled.status == TaskStatus.CANCELLED
        task = service.submit(catalog_query(), timeout_seconds=0.001)
        result = await service.wait(task.spec.id)
        assert result.status == TaskStatus.FAILED and result.error.code == "timeout"
    finally:
        await service.close()


@pytest.mark.parametrize("tool", ["RemovedPlaceholder", *load_tool_scope()["excluded_tool_ids"]])
async def test_unknown_and_excluded_tools_are_explicit(project, tool):
    service = TaskService(project)
    try:
        task = service.submit(TaskInput(calls=[ToolCall(tool=tool)]))
        result = await service.wait(task.spec.id)
        assert result.status == TaskStatus.FAILED
        assert "unknown_tool" in result.error.message
    finally:
        await service.close()


async def test_close_finalizes_not_yet_started_tasks(project):
    service = TaskService(project)
    task = service.submit(catalog_query())
    await service.close()
    assert service.get(task.spec.id).status == TaskStatus.CANCELLED


def test_watchdogs_share_rss_scans_and_drop_finished_workers(project, monkeypatch):
    scans = []
    monkeypatch.setattr("agentgs.runtime.service.time.monotonic", lambda: 1.0)
    monkeypatch.setattr("agentgs.runtime.service.tree_rss", lambda pid: scans.append(pid) or pid)
    service = TaskService(project)
    service.processes = {"a": SimpleNamespace(pid=10), "b": SimpleNamespace(pid=20)}
    assert service._memory_usage(10) == (10, 30)
    assert service._memory_usage(20) == (20, 30)
    assert scans == [10, 20]
    service.processes.pop("a")
    assert service._memory_usage(20) == (20, 20)
    service.processes["c"] = SimpleNamespace(pid=30)
    assert service._memory_usage(30) == (30, 50)
    assert scans == [10, 20, 30]


async def test_rss_watchdog_enforces_limit_without_code_sandbox(project, monkeypatch):
    spawn = asyncio.create_subprocess_exec

    async def memory_worker(*args, **kwargs):
        return await spawn(
            sys.executable,
            "-c",
            "import time; data=bytearray(150*1024*1024); time.sleep(5)",
            **kwargs,
        )

    monkeypatch.setattr("agentgs.runtime.service.asyncio.create_subprocess_exec", memory_worker)
    service = TaskService(project)
    try:
        record = service.submit(catalog_query(), rss_limit_mb=64)
        finished = await service.wait(record.spec.id)
        assert finished.status == TaskStatus.FAILED
        assert finished.error.code == "resource_limit"
    finally:
        await service.close()
