import subprocess
from pathlib import Path

import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskInput, ToolCall
from agentgs.runtime.sandbox.base import CodeExecutor, SandboxUnavailable
from agentgs.runtime.service import TaskService


@pytest.fixture
def sandbox_project(project):
    binary = Path(__file__).parents[3] / "tools" / "bin" / "bwrap"
    if not binary.exists():
        pytest.skip("Local bwrap integration dependency missing")
    probe = subprocess.run(
        [str(binary), "--unshare-all", "--ro-bind", "/", "/", "/usr/bin/true"],
        capture_output=True,
        timeout=5,
    )
    if probe.returncode:
        pytest.skip("Host does not permit the bwrap integration sandbox")
    (project / "tools" / "bin").mkdir(parents=True)
    (project / "tools" / "bin" / "bwrap").symlink_to(binary.resolve())
    return project


async def test_code_does_not_inherit_secrets_or_host_files(sandbox_project, monkeypatch):
    root = sandbox_project
    monkeypatch.setenv("AGENTGS_SECRET_TEST", "secret")
    secret = root / "secret.txt"
    secret.write_text("secret")
    code = f"import os,pathlib; assert 'AGENTGS_SECRET_TEST' not in os.environ; assert not pathlib.Path({str(secret)!r}).exists(); print('isolated')"
    executor = CodeExecutor(root, root / "code", RuntimeSettings())
    result = await executor.run_python(code)
    assert result["stdout"] == "isolated\n" and result["network_isolated"]


async def test_no_sandbox_never_runs_unprotected(project):
    executor = CodeExecutor(project, project / "code", RuntimeSettings(sandbox="disabled"))
    with pytest.raises(SandboxUnavailable):
        await executor.run_python("print(1)")


async def test_rss_limit_is_enforced(sandbox_project):
    service = TaskService(sandbox_project, RuntimeSettings())
    try:
        task = await service.submit(
            TaskInput(
                calls=[
                    ToolCall(
                        tool="ExecutePython",
                        arguments={
                            "code": "import time; data=bytearray(150*1024*1024); time.sleep(5)"
                        },
                    )
                ]
            ),
            rss_limit_mb=64,
        )
        record = await service.wait(task.spec.id)
        assert record.error.code == "resource_limit"
    finally:
        await service.close()
