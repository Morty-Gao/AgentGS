"""Namespace isolation and bounded, cancellation-safe code execution."""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from agentgs.contracts.async_utils import drain, invoke
from agentgs.contracts.sandbox import SandboxUnavailable
from agentgs.runtime.concurrency import RuntimeCoordinator
from agentgs.runtime.sandbox.mounts import OUTPUT_LIMIT, data_mounts
from agentgs.runtime.sandbox.native import (
    bash_binary,
    desktop_command,
    desktop_env,
    desktop_path,
)
from agentgs.runtime.subprocesses import collect, spawn, stop


def shell_runtime_paths():
    paths = [Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path(sys.base_prefix)]
    if sys.prefix != sys.base_prefix:
        paths.append(Path(sys.prefix))
    executable = Path(sys.executable).absolute()
    seen = set()
    while executable.is_symlink() and executable not in seen:
        seen.add(executable)
        executable = (executable.parent / executable.readlink()).absolute()
        paths.append(executable.parent.parent)
    return [path for path in dict.fromkeys(paths) if path.exists()]


def shell_env():
    # Keep the venv launcher so Python discovers the installed project dependencies.
    bindir = str(Path(sys.executable).absolute().parent)
    return {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": "/work",
        "LANG": "C.UTF-8",
        "PYTHONSAFEPATH": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def build_bwrap_command(binary, command, work=None, mounts=(), *, mount_proc=False):
    argv = [
        binary,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--unshare-net",
        "--cap-drop",
        "ALL",
        "--proc" if mount_proc else "--dir",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/data",
    ]
    for path in shell_runtime_paths():
        # Image aliases and interpreter prefixes may point through an NFS root.
        argv.extend(["--ro-bind", str(path.resolve()), str(path)])
    for host, guest in mounts:
        argv.extend(["--ro-bind", str(host), guest])
    if work is not None:
        argv.extend(["--bind", str(Path(work).resolve()), "/work", "--chdir", "/work"])
    else:
        argv.extend(["--chdir", "/tmp"])
    argv.extend(["--remount-ro", "/", "--remount-ro", "/dev", *command])
    return argv


def build_shell_command(kind, binary, command, work, mounts, *, mount_proc=False, rootfs=None):
    if kind != "bwrap":
        raise SandboxUnavailable("Model-authored commands require bwrap; PRoot cannot isolate them")
    return build_bwrap_command(
        binary,
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        work,
        mounts,
        mount_proc=mount_proc,
    )


class CodeExecutor:
    def __init__(self, root, work, settings, *, coordinator=None):
        self.root, self.work, self.settings = Path(root), Path(work), settings
        self.coordinator = coordinator or RuntimeCoordinator(self.root)
        self._shell_backend = None
        self._backend_lock = asyncio.Lock()
        self._mount_proc = False

    def locate(self, name):
        local = self.root / "tools" / "bin" / name
        return str(local) if local.exists() else shutil.which(name)

    async def run_python(self, code):
        async with self.coordinator.tool_invocation(
            self.settings, self.settings.code_timeout_seconds
        ):
            return await self._run_python(code)

    async def _run_python(self, code):
        kind, binary, _ = await self._choose_shell_backend()
        work = self.work / uuid4().hex
        await invoke(work.mkdir, parents=True, exist_ok=False)
        await invoke((work / "main.py").write_text, code, encoding="utf-8")
        if kind == "bwrap":
            argv = build_bwrap_command(
                binary,
                [str(Path(sys.executable).absolute()), "-I", "/work/main.py"],
                work,
                mount_proc=self._mount_proc,
            )
            env = shell_env()
        else:
            await invoke((work / ".tmp").mkdir, exist_ok=True)
            argv = desktop_command(
                kind, binary, [sys.executable, "-I", str(work / "main.py")], work, ()
            )
            env = desktop_env(work, sys.executable)
        process = await spawn(
            *argv,
            cwd=work,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
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
            "isolation": kind,
            "network_isolated": kind != "native_demo",
            "private_proc": self._mount_proc,
        }

    async def describe_shell(self, data_root):
        kind, binary, rootfs = await self._choose_shell_backend()
        mounts = await invoke(data_mounts, data_root)
        namespaced = kind == "bwrap"
        return {
            "kind": kind,
            "binary": binary,
            "rootfs": rootfs,
            "mount_pairs": mounts,
            "isolation": kind,
            "data_mount": "bwrap_ro"
            if namespaced
            else "sandbox_exec_ro"
            if kind == "sandbox_exec"
            else "native_unprotected",
            "network_isolated": kind != "native_demo",
            "private_proc": self._mount_proc,
            "mounts": [guest if namespaced else desktop_path(host) for host, guest in mounts],
            "work_path": "/work" if namespaced else desktop_path(self.work / "shell"),
        }

    async def run_shell(self, command, *, work, data_root, timeout):
        # Admission, setup and execution all consume the same command deadline.
        async with self.coordinator.tool_invocation(self.settings, timeout):
            return await self._run_shell(command, work=work, data_root=data_root)

    async def _run_shell(self, command, *, work, data_root):
        described = await self.describe_shell(data_root)
        await invoke(Path(work).mkdir, parents=True, exist_ok=True)
        if described["kind"] == "bwrap":
            argv = build_shell_command(
                described["kind"],
                described["binary"],
                command,
                work,
                described["mount_pairs"],
                mount_proc=described["private_proc"],
            )
            env = shell_env()
        else:
            bash = await invoke(bash_binary)
            if not bash:
                raise SandboxUnavailable("Desktop demo requires Bash; install Git Bash on Windows")
            await invoke((Path(work) / ".tmp").mkdir, exist_ok=True)
            argv = desktop_command(
                described["kind"],
                described["binary"],
                [bash, "--noprofile", "--norc", "-c", command],
                work,
                described["mount_pairs"],
            )
            env = desktop_env(work, bash)
        process = await spawn(
            *argv,
            cwd=work,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr, truncated, code = await _capture_limited(process, OUTPUT_LIMIT)
        finally:
            await drain(asyncio.create_task(stop(process)))
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": code,
            "truncated": truncated,
            **{
                key: described[key]
                for key in ("isolation", "data_mount", "network_isolated", "private_proc", "mounts")
            },
        }

    async def _choose_shell_backend(self):
        async with self._backend_lock:
            if self._shell_backend is not None:
                return self._shell_backend
            if sys.platform in {"darwin", "win32"} or os.name == "nt":
                if self.settings.sandbox not in {"auto", "disabled"}:
                    raise SandboxUnavailable("Desktop demo uses auto; bwrap requires Linux or WSL")
                seatbelt = (
                    await invoke(shutil.which, "sandbox-exec") if sys.platform == "darwin" else None
                )
                kind = (
                    "sandbox_exec"
                    if seatbelt
                    and self.settings.sandbox == "auto"
                    and await self._probe_seatbelt(seatbelt)
                    else "native_demo"
                )
                self._shell_backend = (kind, seatbelt, None)
                return self._shell_backend
            if self.settings.sandbox not in {"auto", "bwrap"}:
                raise SandboxUnavailable(
                    "Model-authored code requires bwrap with network isolation; PRoot is unsupported"
                )
            binary = await invoke(self.locate, "bwrap")
            if not binary:
                raise SandboxUnavailable("No code sandbox is available; install bwrap")
            errors = []
            # Containers can forbid private procfs while permitting all required namespaces.
            for mount_proc in (True, False):
                probe = await spawn(
                    *build_bwrap_command(binary, ["/usr/bin/true"], mount_proc=mount_proc),
                    env=shell_env(),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                try:
                    async with asyncio.timeout(5):
                        _, stderr = await collect(probe)
                except TimeoutError:
                    errors.append("probe timed out")
                    continue
                finally:
                    await drain(asyncio.create_task(stop(probe)))
                if probe.returncode == 0:
                    self._mount_proc = mount_proc
                    self._shell_backend = ("bwrap", binary, None)
                    return self._shell_backend
                errors.append(stderr.strip()[:500])
            raise SandboxUnavailable("bwrap functional probe failed: " + "; ".join(errors))

    async def _probe_seatbelt(self, binary):
        work = self.work / ".probe"
        await invoke(work.mkdir, parents=True, exist_ok=True)
        try:
            probe = await spawn(
                *desktop_command(
                    "sandbox_exec",
                    binary,
                    ["/bin/bash", "--noprofile", "--norc", "-c", "true"],
                    work,
                    (),
                ),
                cwd=work,
                env=desktop_env(work, "/bin/bash"),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError:
            return False
        try:
            async with asyncio.timeout(5):
                await collect(probe)
            return probe.returncode == 0
        except TimeoutError:
            return False
        finally:
            await drain(asyncio.create_task(stop(probe)))


async def _read_limited(stream, limit):
    if stream is None:
        return "", False
    buf = bytearray()
    truncated = False
    while block := await stream.read(8192):
        remaining = limit - len(buf)
        buf.extend(block[:remaining])
        truncated |= len(block) > remaining
        # Continue draining so logging does not abort the command or block its pipes.
    return buf.decode(errors="replace"), truncated


async def _capture_limited(process, limit):
    readers = [
        asyncio.create_task(_read_limited(process.stdout, limit)),
        asyncio.create_task(_read_limited(process.stderr, limit)),
    ]
    try:
        stdout, stderr = await asyncio.gather(*readers)
        await process.wait()
        return stdout[0], stderr[0], stdout[1] or stderr[1], process.returncode
    finally:
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
