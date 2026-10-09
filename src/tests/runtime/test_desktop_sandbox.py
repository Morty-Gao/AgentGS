"""Desktop demo policy and paths; native OS execution needs platform CI."""

import json
import sys

import pytest

from agentgs.config.settings import RuntimeSettings
from agentgs.runtime.sandbox.base import CodeExecutor, SandboxUnavailable
from agentgs.runtime.sandbox.native import desktop_command, desktop_path, seatbelt_profile


@pytest.mark.parametrize("platform", ["darwin", "win32"])
async def test_desktop_fallback_reports_actual_paths_and_no_isolation(
    project, monkeypatch, platform
):
    from agentgs.runtime.sandbox import base

    bash = base.shutil.which("bash")
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(base.shutil, "which", lambda name: bash if name == "bash" else None)
    code = CodeExecutor(project, project / "code", RuntimeSettings())
    described = await code.describe_shell(project / "data")
    assert described["isolation"] == "native_demo"
    assert described["network_isolated"] is False
    assert described["data_mount"] == "native_unprotected"
    assert str(project) in described["work_path"]
    result = await code.run_shell(
        "printf demo > saved; cat saved",
        work=code.work / "shell",
        data_root=project / "data",
        timeout=5,
    )
    assert result["stdout"] == "demo" and result["exit_code"] == 0


async def test_macos_selects_seatbelt_and_explicit_bwrap_needs_linux(project, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        "agentgs.runtime.sandbox.base.shutil.which", lambda _: "/usr/bin/sandbox-exec"
    )
    code = CodeExecutor(project, project / "code", RuntimeSettings())

    async def available(_):
        return True

    monkeypatch.setattr(code, "_probe_seatbelt", available)
    described = await code.describe_shell(project / "data")
    assert described["isolation"] == "sandbox_exec" and described["network_isolated"]
    forced = CodeExecutor(project, project / "code", RuntimeSettings(sandbox="bwrap"))
    with pytest.raises(SandboxUnavailable, match="WSL"):
        await forced.describe_shell(project / "data")


def test_seatbelt_profile_limits_write_paths_and_denies_network(tmp_path):
    data = tmp_path / 'data with "quotes"'
    work = tmp_path / "work"
    profile = seatbelt_profile(work, [(data, "/data/climate")])
    assert "(deny network*)" in profile
    write_policy = profile.split("(allow file-write*", 1)[1]
    assert json.dumps(str(work.resolve())) in write_policy
    assert json.dumps(str(data.resolve())) not in write_policy
    command = desktop_command(
        "sandbox_exec", "/usr/bin/sandbox-exec", ["/bin/bash", "-c", "echo hi"], work, []
    )
    assert command[-3:] == ["/bin/bash", "-c", "echo hi"]


def test_windows_git_bash_drive_and_unc_paths():
    assert desktop_path(r"C:\Users\Demo User\data", windows=True) == "/c/Users/Demo User/data"
    assert desktop_path(r"\\server\share\data", windows=True) == "//server/share/data"
