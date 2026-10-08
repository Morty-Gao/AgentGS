import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskInput, TaskStatus, ToolCall
from agentgs.runtime.service import TaskService
from agentgs.skills.definitions import load_tool_scope


def catalog_query():
    return TaskInput(
        calls=[
            ToolCall(
                tool="DeepEarthQueryUSGSCatalog", arguments={"min_magnitude": 8}
            )
        ]
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
