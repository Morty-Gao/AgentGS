"""Concurrency contracts tested through real loops, files and child processes."""

import asyncio
import json
import sqlite3
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentgs.agent.state import RunState
from agentgs.client.models import ChatReply, LLMConfig
from agentgs.client.openai_compatible import OpenAICompatibleClient
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.async_utils import invoke
from agentgs.contracts.models import TaskInput, TaskStatus, ToolCall
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.concurrency import RuntimeCoordinator
from agentgs.runtime.models import ModelExecutor
from agentgs.runtime.sandbox.base import CodeExecutor
from agentgs.runtime.service import TaskService
from agentgs.runtime.storage.store import QueueFull, TaskStore
from agentgs.runtime.subprocesses import spawn
from agentgs.runtime.tools.context import paths
from agentgs.runtime.tools.executor import LocalToolExecutor


class Events:
    def __init__(self):
        self.rows = []

    async def emit(self, kind, payload):
        self.rows.append((kind, payload))


def call(value=1, **kwargs):
    return ToolCall(
        tool="CCSHaversineDistance",
        arguments={"lat1": value, "lon1": 0, "lat2": 0, "lon2": 1},
        **kwargs,
    )


def executor(root, *, capacity=2, backend="thread"):
    return LocalToolExecutor(
        DatasetRegistry(root),
        LocalArtifactStore(root / "artifacts"),
        Events(),
        settings=RuntimeSettings(max_concurrent_tool_calls=capacity, tool_backend=backend),
    )


@pytest.mark.parametrize("async_handler", [False, True])
async def test_handlers_overlap_without_blocking_and_own_their_artifacts(project, async_handler):
    tools = executor(project)
    lock = threading.Lock()
    active = peak = 0
    timer = asyncio.create_task(asyncio.sleep(0.01))
    observations = []

    def enter():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)

    def finish(arguments):
        nonlocal active
        observations.append(timer.done())
        output = paths().outputs / "product.txt"
        output.write_text(str(arguments["lat1"]))
        with lock:
            active -= 1
        return {"value": arguments["lat1"], "file": str(output)}

    if async_handler:

        async def handler(arguments):
            enter()
            await asyncio.sleep(0.08)
            return await invoke(finish, arguments)
    else:

        def handler(arguments):
            enter()
            time.sleep(0.08)
            return finish(arguments)

    tools.registry["CCSHaversineDistance"] = replace(
        tools.registry["CCSHaversineDistance"], handler=handler, process_safe=False
    )
    calls = [call(1), call(2)]
    results = await tools.execute_many(calls)
    await timer
    assert peak == 2 and all(observations)
    assert all(result.succeeded for result in results)
    for invocation, result in zip(calls, results):
        files = [ref for ref in result.artifacts if ref.path.endswith("product.txt")]
        assert len(files) == 1
        assert files[0].path == f"calls/{invocation.id}/product.txt"
        assert (tools.artifacts.root / files[0].path).read_text() == str(
            float(invocation.arguments["lat1"])
        )
    await tools.aclose()


async def test_dependencies_gate_execution_and_validate_before_running(project):
    tools = executor(project)
    completed = []

    async def handler(arguments):
        value = arguments["lat1"]
        if value == 1:
            await asyncio.sleep(0.02)
        else:
            assert completed == [1]
        completed.append(value)
        return {"value": value}

    tools.registry["CCSHaversineDistance"] = replace(
        tools.registry["CCSHaversineDistance"], handler=handler, process_safe=False
    )
    a = call(1)
    b = call(2, depends_on=[a.id])
    results = await tools.execute_many([b, a])
    assert [r.data["value"] for r in results] == [2, 1]
    invalid = call(depends_on=["missing"])
    with pytest.raises(ValueError, match="Unknown tool dependency"):
        await tools.execute_many([invalid])
    a.depends_on = [b.id]
    with pytest.raises(ValueError, match="Cyclic"):
        await tools.execute_many([a, b])
    assert completed == [1, 2]


async def test_failed_dependency_prevents_downstream_side_effects(project):
    tools = executor(project)
    a = ToolCall(tool="missing")
    b = call(depends_on=[a.id])
    results = await tools.execute_many([a, b])
    assert results[0].error.code == "unknown_tool"
    assert results[1].error.code == "dependency_failed"
    assert not (tools.artifacts.root / "calls" / b.id).exists()


def config():
    return LLMConfig.model_validate(
        {
            "providers": {
                "p": {
                    "base_url": "https://fixture/v1",
                    "api_key": "fixture",
                    "max_concurrent_requests": 1,
                }
            },
            "models": {
                "m": {
                    "provider": "p",
                    "model": "remote",
                    "context_window": 1000,
                    "max_output_tokens": 10,
                }
            },
            "roles": {"planner": "m"},
        }
    )


async def test_model_capacity_shared_across_clients_and_cancelled_waiter(project):
    active = peak = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        try:
            await release.wait()
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        finally:
            active -= 1

    settings = RuntimeSettings(max_concurrent_model_calls=1)
    clients = [
        ModelExecutor(
            OpenAICompatibleClient(config(), httpx.MockTransport(handler)),
            settings,
            RuntimeCoordinator(project),
        )
        for _ in range(2)
    ]
    first = asyncio.create_task(clients[0].chat([]))
    await entered.wait()
    waiting = asyncio.create_task(clients[1].chat([]))
    await asyncio.sleep(0.04)
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    release.set()
    assert (await first).message.content == "ok"
    assert (await clients[1].chat([])).message.content == "ok"
    assert peak == 1

    def leases():
        with sqlite3.connect(clients[0].coordinator.path) as db:
            return db.execute("SELECT COUNT(*) FROM leases").fetchone()[0]

    assert await invoke(leases) == 0
    await asyncio.gather(*(client.aclose() for client in clients))


async def test_rate_limits_are_atomic_and_usage_is_settled(project, monkeypatch):
    coordinator = RuntimeCoordinator(project)
    await coordinator.initialize()
    now = [100.0]
    monkeypatch.setattr("agentgs.runtime.concurrency.time.time", lambda: now[0])
    assert await invoke(coordinator._rate, "a", "p", 8, 2, 10)
    assert not await invoke(coordinator._rate, "b", "p", 3, 2, 10)
    await coordinator.settle("a", 5)
    assert await invoke(coordinator._rate, "b", "p", 3, 2, 10)
    assert not await invoke(coordinator._rate, "c", "p", 1, 2, 10)
    now[0] += 61
    assert await invoke(coordinator._rate, "c", "p", 1, 2, 10)


async def test_each_transport_retry_is_rate_limited(project):
    cfg = config()
    cfg.providers["p"].requests_per_minute = 10
    seen = []

    async def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}}
        )

    coordinator = RuntimeCoordinator(project)
    client = ModelExecutor(
        OpenAICompatibleClient(cfg, httpx.MockTransport(handler)), RuntimeSettings(), coordinator
    )
    try:
        assert (await client.chat([])).attempts == 2

        def rows():
            with sqlite3.connect(coordinator.path) as db:
                return db.execute("SELECT tokens FROM rates ORDER BY started").fetchall()

        reservations = await invoke(rows)
        assert len(reservations) == 2 and reservations[-1][0] == 3
    finally:
        await client.aclose()


async def test_sqlite_contention_does_not_block_loop(project):
    store = TaskStore(project / "storage")
    ready = threading.Event()

    def lock_database():
        with store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            ready.set()
            time.sleep(0.12)

    thread = threading.Thread(target=lock_database)
    thread.start()
    await asyncio.to_thread(ready.wait)
    writing = asyncio.create_task(store.emit("a", "test", {}))
    await asyncio.sleep(0.02)
    assert not writing.done()
    assert (await writing).sequence == 1
    await asyncio.to_thread(thread.join)


async def sleeping_worker(*args, **kwargs):
    return await spawn(sys.executable, "-c", "import time; time.sleep(5)", **kwargs)


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_task_limit_is_shared_and_foreign_instance_can_cancel(project, monkeypatch):
    monkeypatch.setattr("agentgs.runtime.service.spawn", sleeping_worker)
    settings = RuntimeSettings(max_concurrent_tasks=1)
    a, b = TaskService(project, settings), TaskService(project, settings)
    request = TaskInput(calls=[call()])
    try:
        first = await a.submit(request)
        await wait_until(lambda: bool(a.processes))
        second = await b.submit(request)
        await asyncio.sleep(0.06)
        assert len(a.processes) + len(b.processes) == 1
        record = await b.cancel(first.spec.id)
        assert record.status == TaskStatus.CANCELLED
        assert not a.processes
        await wait_until(lambda: bool(b.processes))
        assert (await a.cancel(second.spec.id)).status == TaskStatus.CANCELLED
    finally:
        await asyncio.gather(a.close(), b.close())
    assert not a.jobs and not b.jobs


async def test_queue_bound_and_admission_deadline(project, monkeypatch):
    monkeypatch.setattr("agentgs.runtime.service.spawn", sleeping_worker)
    service = TaskService(project, RuntimeSettings(max_concurrent_tasks=1, max_pending_tasks=2))
    request = TaskInput(calls=[call()])
    try:
        first = await service.submit(request)
        await wait_until(lambda: bool(service.processes))
        second = await service.submit(request)
        with pytest.raises(QueueFull):
            await service.submit(request)
        await service.cancel(second.spec.id)
        expired = await service.submit(request, timeout_seconds=0.05)
        record = await service.wait(expired.spec.id)
        assert record.error.code == "timeout"
        assert not (project / "outputs/tasks" / expired.spec.id).exists()
        waiter = asyncio.create_task(service.wait(first.spec.id))
        await asyncio.sleep(0.01)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        assert first.spec.id in service.processes
    finally:
        await service.close()


async def test_compute_process_cancel_reaps_child_and_releases_capacity(project, monkeypatch):
    tools = executor(project, backend="process")
    entered = asyncio.Event()
    processes = []

    async def slow_spawn(*args, **kwargs):
        proc = await sleeping_worker(*args, **kwargs)
        processes.append(proc)
        entered.set()
        return proc

    monkeypatch.setattr("agentgs.runtime.tools.process.spawn", slow_spawn)
    task = asyncio.create_task(tools.execute(call()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode is not None
    assert any(kind == "tool.cancelled" for kind, _ in tools.events.rows)
    assert not tools._active and not tools._tasks


async def test_code_calls_have_distinct_scripts_and_reap_processes(project, monkeypatch):
    code = CodeExecutor(project, project / "code", RuntimeSettings(sandbox="bwrap"))
    monkeypatch.setattr(code, "locate", lambda _: "/fixture/bwrap")
    workdirs = []

    async def fake_sandbox(*args, **kwargs):
        if args[-1] == "/usr/bin/true":
            return await spawn(sys.executable, "-c", "pass", **kwargs)
        workdirs.append(Path(kwargs["cwd"]))
        return await spawn(sys.executable, str(Path(kwargs["cwd"]) / "main.py"), **kwargs)

    monkeypatch.setattr("agentgs.runtime.sandbox.base.spawn", fake_sandbox)
    results = await asyncio.gather(code.run_python("print('A')"), code.run_python("print('B')"))
    assert [result["stdout"].strip() for result in results] == ["A", "B"]
    assert len(set(workdirs)) == 2


async def test_model_dialogue_serialization_and_cancellation_tracking():
    entered = asyncio.Event()
    payloads = []

    class Client:
        async def chat(self, messages, **kwargs):
            payloads.append(messages)
            entered.set()
            await asyncio.sleep(0.04)
            return ChatReply(message={"role": "assistant", "content": "ok"})

    events = Events()
    state = RunState(
        SimpleNamespace(
            client=Client(), options=AlgorithmSettings(), events=events, checkpoint=None
        ),
        TaskInput(question="q"),
    )
    await asyncio.gather(state.model("A", stage="A"), state.model("B", stage="B"))
    assert any(message["role"] == "assistant" for message in payloads[1])
    entered.clear()
    task = asyncio.create_task(state.model("C", stage="C"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.usage[-1]["status"] == "cancelled"
    assert events.rows[-1][0] == "model.cancelled"


async def test_lazy_retrieval_initializes_once_without_blocking_loop():
    from agentgs.agent.retrieval.tools import KGToolRetriever, LazyToolRetriever

    initialized = []
    observations = []
    timer = asyncio.create_task(asyncio.sleep(0.01))

    class Embedding:
        def get_text_embedding(self, text):
            time.sleep(0.05)
            observations.append(timer.done())
            return [1.0, 0.0]

    def factory():
        initialized.append(True)
        time.sleep(0.05)
        return KGToolRetriever({}, {}, Embedding())

    retriever = LazyToolRetriever(factory)
    results = await asyncio.gather(
        *(retriever.retrieve(SimpleNamespace(query_str="query")) for _ in range(2))
    )
    await timer
    assert results == [[], []] and initialized == [True] and all(observations)


async def test_cancelled_batch_preserves_completed_results_and_answers_all_calls(project):
    tools = executor(project)
    completed = asyncio.Event()

    async def handler(arguments):
        if arguments["lat1"] == 1:
            return {"value": 1}
        await asyncio.sleep(10)

    tools.registry["CCSHaversineDistance"] = replace(
        tools.registry["CCSHaversineDistance"], handler=handler, process_safe=False
    )

    async def checkpoint(result):
        if result.tool_results:
            completed.set()

    state = RunState(
        SimpleNamespace(tools=tools, events=Events(), checkpoint=checkpoint), TaskInput()
    )
    task = asyncio.create_task(state.execute_many([call(1), call(2)]))
    await completed.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(state.results) == 1 and state.results[0].succeeded
    assert not state.conversation.pending_tool_calls()
    assert {m.metadata.get("status") for m in state.conversation.messages if m.role == "tool"} == {
        "success",
        "cancelled",
    }


async def test_cross_process_capacity_recovers_dead_owner(project):
    code = """import asyncio,sys
from pathlib import Path
from agentgs.runtime.concurrency import RuntimeCoordinator
async def run():
    async with RuntimeCoordinator(Path(sys.argv[1])).slot("test-cross-process",1):
        print("ready",flush=True)
        await asyncio.sleep(10)
asyncio.run(run())
"""
    child = await spawn(
        sys.executable,
        "-c",
        code,
        str(project),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    from agentgs.runtime.subprocesses import stop

    try:
        assert await child.stdout.readline() == b"ready\n"
        coordinator = RuntimeCoordinator(project)
        acquired = asyncio.Event()

        async def waiting():
            async with coordinator.slot("test-cross-process", 1):
                acquired.set()

        waiter = asyncio.create_task(waiting())
        await asyncio.sleep(0.05)
        assert not acquired.is_set()
        child.kill()
        await child.wait()
        async with asyncio.timeout(2):
            await waiter
        assert acquired.is_set()
    finally:
        await stop(child)


async def test_dead_supervisor_records_do_not_fill_queue(project):
    from agentgs.contracts.models import TaskRecord, TaskSpec

    store = TaskStore(project / "outputs/runtime")
    stale = TaskRecord(spec=TaskSpec(input=TaskInput(calls=[call()])))
    await store.reserve(stale, 1)

    def mark_owner_dead():
        with store.connection() as db:
            db.execute("UPDATE controls SET owner_pid=999999999 WHERE task_id=?", (stale.spec.id,))

    await invoke(mark_owner_dead)
    fresh = TaskRecord(spec=TaskSpec(input=TaskInput(calls=[call()])))
    await store.reserve(fresh, 1)
    assert (await store.get(stale.spec.id)).error.code == "owner_lost"
    assert (await store.get(fresh.spec.id)).status == TaskStatus.QUEUED


async def test_parallel_explicit_calls_run_in_real_worker_without_model(project):
    service = TaskService(project, RuntimeSettings(max_concurrent_tool_calls=2))
    request = TaskInput(
        parallel_tools=True,
        calls=[
            ToolCall(id="mean", tool="RemoteSensingMean", arguments={"x": [1, 3]}),
            ToolCall(
                id="kelvin", tool="RemoteSensingKelvinToCelsius", arguments={"kelvin": 273.15}
            ),
        ],
    )
    try:
        submitted = await service.submit(request)
        record = await service.wait(submitted.spec.id)
        assert record.status == TaskStatus.SUCCEEDED, record.error
        assert [result.call_id for result in record.result.tool_results] == ["mean", "kelvin"]
        assert not record.result.usage
        assert not any(
            m.role == "tool" and m.metadata.get("status") != "success"
            for m in record.result.messages
        )
        events = await service.events(record.spec.id)
        kinds = [event.type for event in events if event.type in {"tool.started", "tool.completed"}]
        assert kinds[:2] == ["tool.started", "tool.started"]
    finally:
        await service.close()


async def test_provider_waiters_do_not_hold_global_model_capacity(project):
    cfg = config()
    cfg.providers["q"] = cfg.providers["p"].model_copy()
    cfg.models["fast"] = cfg.models["m"].model_copy(update={"provider": "q", "model": "fast"})
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        if json.loads(request.content)["model"] != "fast":
            entered.set()
            await release.wait()
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = ModelExecutor(
        OpenAICompatibleClient(cfg, httpx.MockTransport(handler)),
        RuntimeSettings(max_concurrent_model_calls=2),
        RuntimeCoordinator(project),
    )
    first = asyncio.create_task(client.chat([]))
    await entered.wait()
    waiting = asyncio.create_task(client.chat([]))
    await asyncio.sleep(0.04)
    try:
        async with asyncio.timeout(1):
            assert (await client.chat([], model="fast")).message.content == "ok"
        assert not waiting.done()
    finally:
        release.set()
        await asyncio.gather(first, waiting)
        await client.aclose()


async def test_transport_close_cancels_inflight_requests_and_rejects_new_calls():
    entered = asyncio.Event()
    active = 0

    async def handler(request):
        nonlocal active
        active += 1
        if active == 2:
            entered.set()
        try:
            await asyncio.sleep(10)
        finally:
            active -= 1

    client = OpenAICompatibleClient(config(), httpx.MockTransport(handler))
    jobs = [asyncio.create_task(client.chat([])) for _ in range(2)]
    await entered.wait()
    await client.aclose()
    assert all(job.cancelled() for job in jobs)
    assert active == 0 and not client._tasks
    with pytest.raises(RuntimeError, match="closed"):
        await client.chat([])


async def test_engine_cancel_during_native_reply_checkpoint_closes_pending_calls(project):
    from agentgs.agent.base import AgentContext
    from agentgs.agent.engine import AgentEngine
    from agentgs.messages import Conversation

    entered = asyncio.Event()
    saved = []

    class Client:
        async def chat(self, *args, **kwargs):
            return ChatReply(
                message={
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "plan",
                            "type": "function",
                            "function": {
                                "name": "submit_tool_plan",
                                "arguments": '{"tools":["CCSHaversineDistance"]}',
                            },
                        }
                    ],
                }
            )

    class Retriever:
        async def retrieve(self, query):
            return [["CCSHaversineDistance", "has the functionality that", "Compute a distance"]]

    async def checkpoint(result):
        saved.append(result)
        if Conversation(messages=result.messages).pending_tool_calls():
            entered.set()
            await asyncio.sleep(10)

    tools = executor(project)
    context = AgentContext(
        tools,
        tools.artifacts,
        Events(),
        client=Client(),
        tool_retriever=Retriever(),
        checkpoint=checkpoint,
    )
    task = asyncio.create_task(AgentEngine().run(TaskInput(question="distance"), context))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not Conversation(messages=saved[-1].messages).pending_tool_calls()
    assert saved[-1].completion_status == "Not Completed"
    assert any(
        message.role == "tool" and message.metadata.get("status") == "cancelled"
        for message in saved[-1].messages
    )
