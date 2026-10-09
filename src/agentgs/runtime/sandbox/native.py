"""Native desktop demo execution; macOS Seatbelt when available."""

import json
import os
import shutil
import sys
from pathlib import Path, PureWindowsPath


def desktop_path(path, *, windows=None):
    """Paths understood by Git Bash on Windows and by native Bash on macOS."""
    windows = os.name == "nt" if windows is None else windows
    if not windows:
        return str(Path(path).resolve())
    path = PureWindowsPath(path)
    if path.drive and not path.drive.startswith("\\\\"):
        return "/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])
    return path.as_posix()


def bash_binary():
    if os.name == "nt":
        # Git Bash executes on Windows; System32/bash.exe delegates into WSL.
        for variable in ("ProgramFiles", "ProgramW6432", "LOCALAPPDATA"):
            folder = os.environ.get(variable)
            if folder:
                for relative in ("Git/bin/bash.exe", "Programs/Git/bin/bash.exe"):
                    candidate = Path(folder) / relative
                    if candidate.is_file():
                        return str(candidate)
    binary = shutil.which("bash")
    if binary and "system32" not in binary.lower():
        return binary
    return None


def desktop_env(work, binary):
    path_entries = [str(Path(sys.executable).parent), str(Path(binary).parent)]
    utilities = Path(binary).parent.parent / "usr/bin"
    if utilities.is_dir():
        path_entries.append(str(utilities))
    path_entries.append(os.defpath)
    env = {
        "PATH": os.pathsep.join(path_entries),
        "HOME": desktop_path(work),
        "LANG": "C.UTF-8",
        "PYTHONSAFEPATH": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TMPDIR": desktop_path(Path(work) / ".tmp"),
        "TEMP": str((Path(work) / ".tmp").resolve()),
        "TMP": str((Path(work) / ".tmp").resolve()),
    }
    if os.name == "nt":
        # Required by Windows process loading, independent of user/proxy settings.
        for name in ("SystemRoot", "WINDIR", "COMSPEC"):
            if name in os.environ:
                env[name] = os.environ[name]
    return env


def seatbelt_profile(work, mounts):
    readonly = [
        "/System",
        "/Library",
        "/usr",
        "/bin",
        "/sbin",
        "/dev",
        sys.base_prefix,
        sys.prefix,
        *[str(host) for host, _ in mounts],
    ]
    work = str(Path(work).resolve())
    read_rules = " ".join(f"(subpath {json.dumps(str(Path(p).resolve()))})" for p in readonly)
    return (
        "(version 1) (deny default) (deny network*) "
        "(allow process*) (allow sysctl-read) (allow mach-lookup) "
        "(allow signal (target self)) "
        f"(allow file-read* {read_rules} (subpath {json.dumps(work)})) "
        f"(allow file-write* (subpath {json.dumps(work)}) "
        '(literal "/dev/null") (literal "/dev/tty"))'
    )


def desktop_command(kind, binary, argv, work, mounts):
    if kind == "sandbox_exec":
        return [binary, "-p", seatbelt_profile(work, mounts), *argv]
    return argv
