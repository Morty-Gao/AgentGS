"""Real shell isolation, lifecycle and shared runtime admission."""

import asyncio
import socket
from dataclasses import replace

import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskInput, ToolCall
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.sandbox.base import CodeExecutor, SandboxUnavailable
from agentgs.runtime.sandbox.mounts import OUTPUT_LIMIT, data_mounts
from agentgs.runtime.service import TaskService
from agentgs.runtime.tools.executor import LocalToolExecutor


class Events:
    def __init__(self):
        self.rows = []

    async def emit(self, kind, payload):
        self.rows.append((kind, payload))


def shell_tools(root, **settings):
    settings = RuntimeSettings(**settings)
    code = CodeExecutor(root, root / "code", settings)
    return LocalToolExecutor(
        DatasetRegistry(root),
        LocalArtifactStore(root / "artifacts"),
        Events(),
        code,
        settings=settings,
        coordinator=code.coordinator,
    )


async def wait_file(path):
    async with asyncio.timeout(5):
        while not path.exists():
            await asyncio.sleep(0.01)


async def test_shell_uses_venv_and_preserves_work_after_output_truncation(sandbox_project):
    tools = shell_tools(sandbox_project)
    command = (
        "python - <<'CODE'\n"
        "import numpy, scipy, pathlib, sys\n"
        "assert sys.prefix != sys.base_prefix\n"
        "sys.stdout.write('x'*50000); sys.stderr.write('y'*50000)\n"
        "pathlib.Path('value.txt').write_text(str(int(numpy.sum([20,22]))))\n"
        "CODE"
    )
    first = await tools.execute(ToolCall(tool="bash", arguments={"command": command}))
    assert first.succeeded, first
    assert first.data["truncated"]
    assert len(first.data["stdout"].encode()) == OUTPUT_LIMIT
    assert len(first.data["stderr"].encode()) == OUTPUT_LIMIT
    second = await tools.execute(ToolCall(tool="bash", arguments={"command": "cat value.txt"}))
    assert second.succeeded and second.data["stdout"] == "42"
    assert second.artifacts
    assert [kind for kind, _ in tools.events.rows] == [
        "tool.queued",
        "tool.started",
        "tool.completed",
        "tool.queued",
        "tool.started",
        "tool.completed",
    ]
    await tools.aclose()


async def test_shell_denies_host_loopback_network_and_hides_host_proc(sandbox_project):
    tools = shell_tools(sandbox_project)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        command = (
            "python - <<'CODE'\n"
            "import socket, pathlib, os\n"
            "for address in [('127.0.0.1', " + str(port) + "), ('192.0.2.1', 80)]:\n"
            "    s=socket.socket(); s.settimeout(0.3)\n"
            "    try: s.connect(address)\n"
            "    except OSError: pass\n"
            "    else: raise AssertionError('network reachable')\n"
            "    finally: s.close()\n"
            "assert not pathlib.Path('/proc/" + str(os_pid()) + "/environ').exists()\n"
            "print('isolated')\nCODE"
        )
        result = await tools.execute(ToolCall(tool="bash", arguments={"command": command}))
    assert result.succeeded, result
    assert result.data["stdout"] == "isolated\n"
    assert result.data["network_isolated"]
    await tools.aclose()


def os_pid():
    import os

    return os.getpid()


async def test_shell_failure_retains_observations_and_validation_prevents_execution(
    sandbox_project,
):
    tools = shell_tools(sandbox_project)
    result = await tools.execute(
        ToolCall(tool="bash", arguments={"command": "echo observed; echo diagnostic >&2; exit 7"})
    )
    assert not result.succeeded and result.error.code == "shell_exit"
    assert result.data["stdout"] == "observed\n" and result.data["exit_code"] == 7
    assert result.data["stderr"] == "diagnostic\n" and result.artifacts
    for arguments in (
        {"command": " "},
        {"command": "x\x00y"},
        {"command": 3},
        {"command": "touch forbidden", "extra": True},
        {"command": "x" * 4001},
    ):
        bad = await tools.execute(ToolCall(tool="bash", arguments=arguments))
        assert not bad.succeeded and bad.error.code == "validation"
    assert not (tools.code_executor.work / "shell/forbidden").exists()
    await tools.aclose()


async def test_shell_and_direct_python_tool_share_admission_without_nested_deadlock(
    sandbox_project,
):
    tools = shell_tools(sandbox_project, max_concurrent_tool_calls=1)
    marker = tools.code_executor.work / "shell/entered"
    shell = asyncio.create_task(
        tools.execute(
            ToolCall(tool="bash", arguments={"command": "touch entered; sleep 0.3; echo done"})
        )
    )
    await wait_file(marker)
    entered = asyncio.Event()

    async def handler(arguments):
        entered.set()
        return {"value": 42}

    tools.registry["CCSHaversineDistance"] = replace(
        tools.registry["CCSHaversineDistance"], handler=handler, process_safe=False
    )
    direct = asyncio.create_task(
        tools.execute(
            ToolCall(
                tool="CCSHaversineDistance", arguments={"lat1": 0, "lon1": 0, "lat2": 0, "lon2": 1}
            )
        )
    )
    await asyncio.sleep(0.05)
    assert not entered.is_set()
    assert (await shell).succeeded
    assert (await direct).succeeded and entered.is_set()
    await tools.aclose()


@pytest.mark.parametrize("cancel", [False, True])
async def test_shell_timeout_and_close_reap_background_children(sandbox_project, cancel):
    tools = shell_tools(sandbox_project, tool_timeout_seconds=5 if cancel else 0.5)
    await tools.code_executor.describe_shell(sandbox_project / "data")
    await tools.coordinator.initialize()
    work = tools.code_executor.work / "shell"
    job = asyncio.create_task(
        tools.execute(
            ToolCall(
                tool="bash",
                arguments={
                    "command": "touch entered; (sleep 1.2; touch leaked) >/dev/null 2>&1 & wait"
                },
            )
        )
    )
    await wait_file(work / "entered")
    if cancel:
        await tools.aclose()
        with pytest.raises(asyncio.CancelledError):
            await job
    else:
        result = await job
        assert not result.succeeded and result.error.code == "timeout"
    await asyncio.sleep(1.3)
    assert not (work / "leaked").exists()
    async with tools.coordinator.slot("tools", 1):
        pass
    await tools.aclose()


async def test_task_service_runs_bash_in_worker_and_records_result(sandbox_project):
    service = TaskService(sandbox_project, RuntimeSettings(sandbox="bwrap"))
    try:
        task = await service.submit(
            TaskInput(
                calls=[
                    ToolCall(
                        tool="bash", arguments={"command": "cat /data/deep_earth/usgs_source.csv"}
                    )
                ]
            )
        )
        record = await service.wait(task.spec.id)
        assert record.error is None, record
        result = record.result.tool_results[0]
        assert result.succeeded and result.data["network_isolated"]
        assert "8.5,C" in result.data["stdout"]
        events = await service.events(task.spec.id)
        assert any(row.type == "tool.completed" for row in events)
    finally:
        await service.close()


@pytest.mark.parametrize("backend", ["disabled", "proot"])
async def test_linux_never_falls_back_to_host_shell(project, backend):
    code = CodeExecutor(
        project, project / "code", RuntimeSettings(sandbox=backend, allow_proot=True)
    )
    with pytest.raises(SandboxUnavailable):
        await code.run_shell(
            "touch leaked", work=project / "scratch", data_root=project / "data", timeout=5
        )
    assert not (project / "scratch/leaked").exists()


def test_mount_refuses_symlink_to_answer_directory(project):
    (project / "data/metadata").mkdir()
    (project / "data/climate").symlink_to(project / "data/metadata", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        data_mounts(project / "data")


async def test_normal_shell_exit_does_not_leave_background_writers(sandbox_project):
    tools = shell_tools(sandbox_project)
    result = await tools.execute(
        ToolCall(
            tool="bash",
            arguments={"command": "(sleep 0.3; touch leaked) >/dev/null 2>&1 & echo returned"},
        )
    )
    assert result.succeeded and result.data["stdout"] == "returned\n"
    await asyncio.sleep(0.4)
    assert not (tools.code_executor.work / "shell/leaked").exists()
    await tools.aclose()


async def test_shared_memory_scratch_is_writable_without_exposing_host_files(sandbox_project):
    tools = shell_tools(sandbox_project)
    result = await tools.execute(
        ToolCall(
            tool="bash",
            arguments={"command": "printf scratch > /dev/shm/probe; cat /dev/shm/probe"},
        )
    )
    assert result.succeeded and result.data["stdout"] == "scratch"
    await tools.aclose()
