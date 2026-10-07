import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskInput, TaskStatus, ToolCall
from agentgs.runtime.service import TaskService


def distance():
    return TaskInput(
        calls=[
            ToolCall(
                tool="GeoHaversineDistance", arguments={"lat1": 0, "lon1": 0, "lat2": 0, "lon2": 1}
            )
        ]
    )


async def test_real_workers_queue_artifacts_and_events(project):
    service = TaskService(project, RuntimeSettings(max_concurrent_tasks=1))
    try:
        first = service.submit(distance())
        second = service.submit(distance())
        a = await service.wait(first.spec.id)
        b = await service.wait(second.spec.id)
        assert a.status == b.status == TaskStatus.SUCCEEDED
        assert a.result.tool_results[0].data["distance_km"] == pytest.approx(111.1950802)
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
        task = service.submit(distance())
        cancelled = await service.cancel(task.spec.id)
        assert cancelled.status == TaskStatus.CANCELLED
        task = service.submit(distance(), timeout_seconds=0.001)
        result = await service.wait(task.spec.id)
        assert result.status == TaskStatus.FAILED and result.error.code == "timeout"
    finally:
        await service.close()


async def test_missing_data_and_unknown_tool_are_explicit(project):
    service = TaskService(project)
    try:
        task = service.submit(TaskInput(calls=[ToolCall(tool="RemovedPlaceholder")]))
        result = await service.wait(task.spec.id)
        assert result.status == TaskStatus.FAILED
        assert "unknown_tool" in result.error.message
    finally:
        await service.close()


async def test_close_finalizes_not_yet_started_tasks(project):
    service = TaskService(project)
    task = service.submit(distance())
    await service.close()
    assert service.get(task.spec.id).status == TaskStatus.CANCELLED
