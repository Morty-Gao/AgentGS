import asyncio
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from agentgs.contracts.async_utils import drain, invoke
from agentgs.runtime.subprocesses import collect, spawn, stop


class SandboxUnavailable(RuntimeError):
    pass


class CodeExecutor:
    def __init__(self, root, work, settings):
        self.root, self.work, self.settings = root, work, settings

    def locate(self, name):
        local = self.root / "tools" / "bin" / name
        return str(local) if local.exists() else shutil.which(name)

    async def run_python(self, code):
        async with asyncio.timeout(self.settings.code_timeout_seconds):
            return await self._run_python(code)

    async def _run_python(self, code):
        if os.name == "nt":
            raise SandboxUnavailable(
                "Native Windows code isolation is not implemented; run this task in WSL"
            )
        work = self.work / uuid4().hex
        await invoke(work.mkdir, parents=True, exist_ok=False)
        script = work / "main.py"
        await invoke(script.write_text, code)
        python = str(Path(sys.executable).resolve())
        backend = self.settings.sandbox
        bwrap = await invoke(self.locate, "bwrap")
        if bwrap and backend in {"auto", "bwrap"}:
            probe = await spawn(
                bwrap,
                "--unshare-all",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "/usr/bin/true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(probe.wait(), 5)
            finally:
                await drain(asyncio.create_task(stop(probe)))
            if probe.returncode != 0:
                if backend == "bwrap":
                    raise SandboxUnavailable("bwrap functional probe failed")
                bwrap = None
        if backend in {"auto", "bwrap"} and bwrap:
            command = [
                bwrap,
                "--die-with-parent",
                "--new-session",
                "--unshare-all",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
            ]
            mounts = [Path("/usr"), Path("/lib"), Path("/lib64"), Path(sys.base_prefix)]
            for path in dict.fromkeys(mounts):
                if path.exists():
                    command.extend(["--ro-bind", str(path), str(path)])
            command.extend(
                [
                    "--bind",
                    str(work),
                    "/work",
                    "--chdir",
                    "/work",
                    python,
                    "-I",
                    "/work/main.py",
                ]
            )
            isolation = "bwrap"
        elif backend in {"auto", "proot"} and self.settings.allow_proot and self.locate("proot"):
            # Opt-in compatibility only: PRoot is not a kernel/network isolation boundary.
            rootfs = self.root / "tools" / "share" / "rootfs"
            if not rootfs.exists():
                raise SandboxUnavailable("PRoot rootfs missing")
            command = [
                self.locate("proot"),
                "-r",
                str(rootfs),
                "-w",
                "/work",
                "-b",
                str(work) + ":/work",
            ]
            command.extend(["-b", "/dev", "-b", "/proc"])
            for path in [Path("/usr"), Path("/lib"), Path("/lib64"), Path(sys.base_prefix)]:
                if path.exists():
                    command.extend(["-b", str(path)])
            command.extend([python, "-I", "/work/main.py"])
            isolation = "proot_compatibility"
        else:
            raise SandboxUnavailable(
                "No configured code sandbox is available; install bwrap (or explicitly opt in to PRoot)"
            )
        env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"}
        process = await spawn(
            *command,
            cwd=work,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        try:
            stdout, stderr = await collect(process)
        finally:
            await drain(asyncio.create_task(stop(process)))
        if process.returncode:
            raise RuntimeError(f"Sandbox exited {process.returncode}: {stderr[:2000]}")
        return {
            "stdout": stdout,
            "stderr": stderr,
            "isolation": isolation,
            "network_isolated": isolation == "bwrap",
        }
