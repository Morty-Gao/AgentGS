"""Sandbox boundaries and resource admission under competing calls."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
from test_shell_runtime import shell_tools, wait_file

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskInput, TaskRecord, TaskSpec, ToolCall
from agentgs.runtime.sandbox.base import SandboxUnavailable, shell_runtime_paths
from agentgs.runtime.storage.store import TaskStore
from agentgs.runtime.subprocesses import spawn, stop


def distance_call():
    return ToolCall(
        tool="CCSHaversineDistance", arguments={"lat1": 0, "lon1": 0, "lat2": 0, "lon2": 1}
    )


def test_interpreter_alias_never_mounts_host_root(tmp_path, monkeypatch):
    alias = tmp_path / "python"
    alias.symlink_to("/bin/sh")
    monkeypatch.setattr(sys, "executable", str(alias))
    assert Path("/") not in shell_runtime_paths()
    monkeypatch.setattr(sys, "base_prefix", "/")
    with pytest.raises(SandboxUnavailable, match="host root"):
        shell_runtime_paths()


async def test_same_workspace_waiter_does_not_consume_tool_capacity(sandbox_project):
    tools = shell_tools(sandbox_project, max_concurrent_tool_calls=2)
    first = asyncio.create_task(
        tools.execute(
            ToolCall(
                tool="bash",
                arguments={
                    "command": "echo first > state; touch entered; sleep .3; test $(cat state) = first"
                },
            )
        )
    )
    await wait_file(tools.code_executor.work / "shell/entered")
    second = asyncio.create_task(
        tools.execute(ToolCall(tool="bash", arguments={"command": "echo second > state"}))
    )
    await asyncio.sleep(0.03)
    async with asyncio.timeout(0.2):
        direct = await tools.execute(distance_call())
    assert direct.succeeded
    assert (await first).succeeded and (await second).succeeded
    assert (tools.code_executor.work / "shell/state").read_text().strip() == "second"
    await tools.aclose()


async def test_sandbox_capacity_reserves_room_for_direct_tools(sandbox_project):
    first = shell_tools(
        sandbox_project, max_concurrent_tool_calls=2, max_concurrent_sandbox_calls=1
    )
    second = shell_tools(
        sandbox_project, max_concurrent_tool_calls=2, max_concurrent_sandbox_calls=1
    )
    second.code_executor.work = sandbox_project / "second-code"
    running = asyncio.create_task(
        first.execute(ToolCall(tool="bash", arguments={"command": "touch entered; sleep .3"}))
    )
    await wait_file(first.code_executor.work / "shell/entered")
    queued = asyncio.create_task(
        second.execute(ToolCall(tool="bash", arguments={"command": "touch entered"}))
    )
    await asyncio.sleep(0.03)
    assert not (second.code_executor.work / "shell/entered").exists()
    async with asyncio.timeout(0.2):
        assert (await second.execute(distance_call())).succeeded
    assert (await running).succeeded and (await queued).succeeded
    await first.aclose()
    await second.aclose()


async def test_result_publication_holds_no_compute_lease(project, monkeypatch):
    tools = shell_tools(project, max_concurrent_tool_calls=1)
    publishing, finish = asyncio.Event(), asyncio.Event()
    register = tools._register_products
    first_id = distance_call()

    async def slow_publication(call, data):
        if call.id == first_id.id:
            publishing.set()
            await finish.wait()
        return await register(call, data)

    monkeypatch.setattr(tools, "_register_products", slow_publication)
    pending = asyncio.create_task(tools.execute(first_id))
    await publishing.wait()
    try:
        async with asyncio.timeout(1):
            assert (await tools.execute(distance_call())).succeeded
    finally:
        finish.set()
        await pending
        await tools.aclose()


async def test_waiting_shell_cancel_releases_atomic_reservation(sandbox_project):
    tools = shell_tools(
        sandbox_project, max_concurrent_tool_calls=2, max_concurrent_sandbox_calls=1
    )
    async with tools.coordinator.tool_invocation(tools.settings, 5, sandbox=True):
        queued = asyncio.create_task(
            tools.execute(ToolCall(tool="bash", arguments={"command": "touch ran"}))
        )
        await asyncio.sleep(0.03)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
    assert not (tools.code_executor.work / "shell/ran").exists()
    async with asyncio.timeout(1):
        assert (
            await tools.execute(ToolCall(tool="bash", arguments={"command": "echo ready"}))
        ).succeeded
    await tools.aclose()


async def test_foreign_worker_records_do_not_affect_recovery_or_rss(project, monkeypatch):
    a = TaskStore(project / "storage")
    b = TaskStore(project / "storage")
    a.pool, b.pool = "worker-a", "worker-b"
    ra, rb = (TaskRecord(spec=TaskSpec(input=TaskInput(question=q))) for q in ("a", "b"))
    await a.reserve(ra, 1)
    with a.connection() as db:
        db.execute("UPDATE controls SET owner_pid=999999999 WHERE task_id=?", (ra.spec.id,))
    await b.recover()
    assert (await b.get(ra.spec.id)).status.value == "queued"
    await b.reserve(rb, 1)
    assert await a.worker(ra.spec.id, os.getpid(), 777) == 777
    assert await b.worker(rb.spec.id, os.getpid(), 222) == 222
    assert await a.worker(ra.spec.id, os.getpid(), 777) == 777


async def test_unclassified_owner_is_not_recovered_using_local_pid(project):
    store = TaskStore(project / "storage")
    record = TaskRecord(spec=TaskSpec(input=TaskInput(question="unknown owner")))
    await store.reserve(record, 1)
    with store.connection() as db:
        db.execute(
            "UPDATE controls SET owner_pool=NULL,owner_pid=999999999 WHERE task_id=?",
            (record.spec.id,),
        )
    await store.recover()
    assert (await store.get(record.spec.id)).status.value == "queued"


async def test_creation_failure_does_not_replace_cancellation(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def failed_creation(*args, **kwargs):
        entered.set()
        await release.wait()
        raise FileNotFoundError("missing")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", failed_creation)
    pending = asyncio.create_task(spawn("missing"))
    await entered.wait()
    pending.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending


async def test_stop_kills_a_child_without_its_own_process_group():
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(10)", start_new_session=False
    )
    try:
        async with asyncio.timeout(1):
            await stop(process)
        assert process.returncode is not None
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_database_contention_does_not_extend_cancelled_admission_by_seconds(project):
    import sqlite3
    import time

    from agentgs.runtime.concurrency import RuntimeCoordinator

    coordinator = RuntimeCoordinator(project)
    await coordinator.initialize()
    with sqlite3.connect(coordinator.path) as db:
        db.execute("BEGIN IMMEDIATE")
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                async with coordinator.slot("tools", 1):
                    pass
        assert time.monotonic() - start < 0.5
        db.rollback()
    async with coordinator.slot("tools", 1):
        pass


async def test_completed_dead_worker_does_not_keep_rss_reserved(project):
    from agentgs.contracts.models import TaskStatus

    store = TaskStore(project / "storage")
    record = TaskRecord(spec=TaskSpec(input=TaskInput(question="completed")))
    await store.reserve(record, 1)
    record.status = TaskStatus.SUCCEEDED
    await store.save(record)
    with store.connection() as db:
        db.execute(
            "UPDATE controls SET worker_pid=999999999,worker_identity=1,rss=999999 WHERE task_id=?",
            (record.spec.id,),
        )
    await store.recover()
    control = await store.control(record.spec.id)
    assert control["worker_pid"] is None and control["rss"] == 0
    assert (await store.get(record.spec.id)).status == TaskStatus.SUCCEEDED


@pytest.mark.parametrize("target", ["/tmp", "/dev/shm", "/work"])
async def test_storage_limit_is_enforced_and_scratch_is_reaped(sandbox_project, target):
    tools = shell_tools(sandbox_project, sandbox_disk_limit_mb=1)
    command = "python -c " + repr(
        "from pathlib import Path; Path("
        + repr(target + "/large")
        + ").write_bytes(b'x'*(2*1024*1024))"
    )
    result = await tools.execute(ToolCall(tool="bash", arguments={"command": command}))
    assert not result.succeeded and result.error.code == "resource_limit"
    assert not list(tools.code_executor.work.glob(".scratch-*"))
    await tools.aclose()


async def test_posix_shared_memory_works_with_disk_backed_scratch(sandbox_project):
    tools = shell_tools(sandbox_project)
    command = "python - <<'CODE'\nfrom multiprocessing import shared_memory\ns=shared_memory.SharedMemory(create=True, size=8)\ns.buf[0]=42\nassert s.buf[0]==42\ns.close(); s.unlink()\nprint('shared')\nCODE"
    result = await tools.execute(ToolCall(tool="bash", arguments={"command": command}))
    assert result.succeeded and result.data["stdout"] == "shared\n", result
    assert not list(tools.code_executor.work.glob(".scratch-*"))
    await tools.aclose()


def test_storage_accounting_does_not_follow_external_links_or_open_fifos(tmp_path):
    from agentgs.runtime.sandbox.storage import check_storage

    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with (outside / "large").open("wb") as stream:
        stream.truncate(4 * 1024 * 1024)
    (work / "outside").symlink_to(outside, target_is_directory=True)
    os.mkfifo(work / "pipe")
    check_storage([work], RuntimeSettings(sandbox_disk_limit_mb=1))


async def test_failed_sandbox_start_cleans_scratch_and_capacity(sandbox_project, monkeypatch):
    tools = shell_tools(sandbox_project)
    await tools.code_executor.describe_shell(sandbox_project / "data")

    async def fail(*args, **kwargs):
        raise FileNotFoundError("missing sandbox binary")

    monkeypatch.setattr("agentgs.runtime.sandbox.base.spawn", fail)
    result = await tools.execute(ToolCall(tool="bash", arguments={"command": "echo test"}))
    assert not result.succeeded and result.error.code == "dependency_missing"
    assert not list(tools.code_executor.work.glob(".scratch-*"))
    async with tools.coordinator.tool_invocation(tools.settings, 1, sandbox=True):
        pass
    await tools.aclose()


async def test_unreadable_scratch_does_not_prevent_cleanup(sandbox_project):
    tools = shell_tools(sandbox_project)
    result = await tools.execute(
        ToolCall(
            tool="bash",
            arguments={"command": "mkdir -p /tmp/locked; chmod 000 /tmp/locked; sleep .5"},
        )
    )
    assert not result.succeeded and result.error.code == "resource_limit", result
    assert not list(tools.code_executor.work.glob(".scratch-*"))
    await tools.aclose()


async def test_direct_sandbox_execution_enforces_rss_without_task_service(sandbox_project):
    tools = shell_tools(sandbox_project, default_rss_limit_mb=64)
    result = await tools.execute(
        ToolCall(
            tool="bash",
            arguments={
                "command": 'python -c "import time; x=bytearray(128*1024*1024); time.sleep(1)"'
            },
        )
    )
    assert not result.succeeded and result.error.code == "resource_limit", result
    assert not list(tools.code_executor.work.glob(".scratch-*"))
    await tools.aclose()


@pytest.mark.parametrize("target", ["python", "../bin/python"])
def test_cyclic_interpreter_alias_is_rejected_without_unbounded_traversal(
    tmp_path, monkeypatch, target
):
    folder = tmp_path / "bin"
    folder.mkdir()
    alias = folder / "python"
    alias.symlink_to(target)
    monkeypatch.setattr(sys, "executable", str(alias))
    with pytest.raises(SandboxUnavailable, match="alias chain"):
        shell_runtime_paths()


def test_owned_scratch_cleanup_handles_depth_beyond_python_stack(tmp_path):
    from agentgs.runtime.sandbox.storage import remove_scratch

    root = tmp_path / "scratch"
    root.mkdir()
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(1100):
            os.mkdir("d", dir_fd=fd)
            child = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        os.fchmod(fd, 0)
    finally:
        os.close(fd)
    remove_scratch(root)
    assert not root.exists()


async def test_command_assembly_failure_cleans_prepared_scratch(sandbox_project, monkeypatch):
    tools = shell_tools(sandbox_project)
    await tools.code_executor.describe_shell(sandbox_project / "data")

    def fail(*args, **kwargs):
        raise SandboxUnavailable("invalid interpreter")

    monkeypatch.setattr("agentgs.runtime.sandbox.base.build_shell_command", fail)
    result = await tools.execute(ToolCall(tool="bash", arguments={"command": "echo test"}))
    assert not result.succeeded and result.error.code == "sandbox_unavailable"
    assert not list(tools.code_executor.work.glob(".scratch-*"))
    await tools.aclose()
