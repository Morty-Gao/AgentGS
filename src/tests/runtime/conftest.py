import sys

import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.runtime.sandbox.base import CodeExecutor, SandboxUnavailable


@pytest.fixture
async def sandbox_project(project):
    if sys.platform != "linux":
        pytest.skip("Linux namespace integration test")
    executor = CodeExecutor(project, project / "code", RuntimeSettings(sandbox="bwrap"))
    try:
        await executor.describe_shell(project / "data")
    except SandboxUnavailable as exc:
        pytest.skip(str(exc))
    return project
