"""Namespace isolation and bounded, cancellation-safe code execution."""

import asyncio
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from agentgs.contracts.async_utils import drain, invoke
from agentgs.contracts.sandbox import SandboxResourceLimit, SandboxUnavailable
from agentgs.runtime.concurrency import RuntimeCoordinator
from agentgs.runtime.resources import tree_rss
from agentgs.runtime.sandbox.mounts import OUTPUT_LIMIT, data_mounts
from agentgs.runtime.sandbox.native import (
    bash_binary,
    desktop_command,
    desktop_env,
    desktop_path,
)
from agentgs.runtime.sandbox.storage import check_storage, remove_scratch
from agentgs.runtime.subprocesses import collect, spawn, stop


def shell_runtime_paths():
    paths = [Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path(sys.base_prefix)]
    if sys.prefix != sys.base_prefix:
        paths.append(Path(sys.prefix))
    prefixes = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    if Path("/") in prefixes:
        raise SandboxUnavailable("Interpreter prefix must not expose the host root")
    executable = Path(sys.executable).absolute()
    seen = set()
    while executable.is_symlink() and executable not in seen:
        if len(seen) >= 40:
            raise SandboxUnavailable("Interpreter alias chain is cyclic or too long")
        seen.add(executable)
        executable = (executable.parent / executable.readlink()).absolute()
        alias = executable.parent.parent
        paths.append(alias if alias.resolve() in prefixes else executable.parent)
    if executable in seen:
        raise SandboxUnavailable("Interpreter alias chain is cyclic or too long")
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


def build_bwrap_command(binary, command, work=None, mounts=(), *, mount_proc=False, scratch=None):
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
        "--dir",
        "/data",
    ]
    if scratch is None:
        argv.extend(["--tmpfs", "/tmp", "--tmpfs", "/dev/shm"])
    else:
        argv.extend(
            ["--bind", str(scratch / "tmp"), "/tmp", "--bind", str(scratch / "shm"), "/dev/shm"]
        )
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


def build_shell_command(kind, binary, command, work, mounts, *, mount_proc=False, scratch=None):
    if kind != "bwrap":
        raise SandboxUnavailable("Model-authored commands require bwrap; PRoot cannot isolate them")
    return build_bwrap_command(
        binary,
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        work,
        mounts,
        mount_proc=mount_proc,
        scratch=scratch,
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

    async def run_python(self, code, *, coordinator=None, settings=None, on_started=None):
        settings = settings or self.settings
        coordinator = coordinator or self.coordinator
        async with coordinator.tool_invocation(
            settings, settings.code_timeout_seconds, sandbox=True
        ):
            if on_started is not None:
                await invoke(on_started)
            return await self._run_python(code, settings)

    async def _run_python(self, code, settings):
        kind, binary, _ = await self._choose_shell_backend()
        work = self.work / uuid4().hex
        await invoke(work.mkdir, parents=True, exist_ok=False)
        await invoke((work / "main.py").write_text, code, encoding="utf-8")
        async with self._scratch_for(work, kind == "bwrap") as scratch:
            if kind == "bwrap":
                argv = build_bwrap_command(
                    binary,
                    [str(Path(sys.executable).absolute()), "-I", "/work/main.py"],
                    work,
                    mount_proc=self._mount_proc,
                    scratch=scratch,
                )
                env = shell_env()
            else:
                await invoke((work / ".tmp").mkdir, exist_ok=True)
                argv = desktop_command(
                    kind, binary, [sys.executable, "-I", str(work / "main.py")], work, ()
                )
                env = desktop_env(work, sys.executable)
            stdout, stderr, code = await self._execute(
                argv, work, env, scratch, settings, limited=False
            )
        if code:
            raise RuntimeError(f"Sandbox exited {code}: {stderr[:2000]}")
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

    async def run_shell(
        self, command, *, work, data_root, timeout, coordinator=None, settings=None, on_started=None
    ):
        # Admission, setup and execution all consume the same command deadline.
        coordinator = coordinator or self.coordinator
        settings = settings or self.settings
        async with asyncio.timeout(timeout):
            work = await invoke(Path(work).resolve)
            # A queued command in the same persistent workspace consumes no compute slot.
            async with coordinator.workspace(work):
                async with coordinator.tool_invocation(settings, timeout, sandbox=True):
                    if on_started is not None:
                        await invoke(on_started)
                    return await self._run_shell(
                        command, work=work, data_root=data_root, settings=settings
                    )

    async def _run_shell(self, command, *, work, data_root, settings):
        described = await self.describe_shell(data_root)
        await invoke(Path(work).mkdir, parents=True, exist_ok=True)
        async with self._scratch_for(work, described["kind"] == "bwrap") as scratch:
            if described["kind"] == "bwrap":
                argv = build_shell_command(
                    described["kind"],
                    described["binary"],
                    command,
                    work,
                    described["mount_pairs"],
                    mount_proc=described["private_proc"],
                    scratch=scratch,
                )
                env = shell_env()
            else:
                bash = await invoke(bash_binary)
                if not bash:
                    raise SandboxUnavailable(
                        "Desktop demo requires Bash; install Git Bash on Windows"
                    )
                await invoke((Path(work) / ".tmp").mkdir, exist_ok=True)
                argv = desktop_command(
                    described["kind"],
                    described["binary"],
                    [bash, "--noprofile", "--norc", "-c", command],
                    work,
                    described["mount_pairs"],
                )
                env = desktop_env(work, bash)
            stdout, stderr, truncated, code = await self._execute(
                argv, work, env, scratch, settings, limited=True
            )
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

    @asynccontextmanager
    async def _scratch_for(self, work, enabled):
        scratch = await self._prepare_scratch(work) if enabled else None
        try:
            yield scratch
        finally:
            if scratch is not None:
                await drain(asyncio.create_task(invoke(remove_scratch, scratch)))

    async def _prepare_scratch(self, work):
        scratch = Path(work).parent / (".scratch-" + uuid4().hex)

        def create():
            scratch.mkdir(parents=True, exist_ok=False)
            (scratch / "tmp").mkdir()
            (scratch / "shm").mkdir()

        try:
            await invoke(create)
        except BaseException:
            await drain(asyncio.create_task(invoke(shutil.rmtree, scratch, ignore_errors=True)))
            raise
        return scratch

    async def _execute(self, argv, work, env, scratch, settings, *, limited):
        process = None
        capture = monitor = None
        roots = [work] + ([scratch] if scratch is not None else [])

        async def watch_resources():
            while True:
                rss = await invoke(tree_rss, process.pid)
                if rss > settings.default_rss_limit_mb * 1024 * 1024:
                    raise SandboxResourceLimit(
                        f"Sandbox process tree exceeds {settings.default_rss_limit_mb} MiB RSS"
                    )
                await invoke(check_storage, roots, settings)
                await asyncio.sleep(0.2)

        async def cleanup():
            if process is not None:
                await stop(process)
            for task in (capture, monitor):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(t for t in (capture, monitor) if t is not None), return_exceptions=True
            )

        try:
            await invoke(check_storage, roots, settings)
            process = await spawn(
                *argv,
                cwd=work,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
            )
            capture = asyncio.create_task(
                _capture_limited(process, OUTPUT_LIMIT) if limited else collect(process)
            )
            monitor = asyncio.create_task(watch_resources())
            done, _ = await asyncio.wait((capture, monitor), return_when=asyncio.FIRST_COMPLETED)
            if monitor in done:
                await monitor
            output = await capture
            await invoke(check_storage, roots, settings)
            return output if limited else (*output, process.returncode)
        finally:
            await drain(asyncio.create_task(cleanup()))

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
