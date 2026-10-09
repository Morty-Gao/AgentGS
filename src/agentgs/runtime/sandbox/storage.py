"""Bounded workspace accounting; links and special files are never followed."""

import errno
import os
import stat
import subprocess

from agentgs.contracts.sandbox import SandboxResourceLimit


def check_storage(roots, settings):
    size = count = 0
    limit = settings.sandbox_disk_limit_mb * 1024 * 1024

    def account(metadata):
        nonlocal size, count
        count += 1
        if count > settings.sandbox_max_files:
            raise SandboxResourceLimit(
                f"Sandbox workspace exceeds {settings.sandbox_max_files} entries"
            )
        if stat.S_ISREG(metadata.st_mode):
            size += metadata.st_size
            if size > limit:
                raise SandboxResourceLimit(
                    f"Sandbox workspace exceeds {settings.sandbox_disk_limit_mb} MiB"
                )

    def walk(root, depth=0):
        if depth > 64:
            raise SandboxResourceLimit("Sandbox workspace exceeds 64 directory levels")
        with os.scandir(root) as entries:
            for entry in entries:
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                account(metadata)
                if not stat.S_ISDIR(metadata.st_mode):
                    continue
                if os.name == "nt":
                    walk(entry.path, depth + 1)
                    continue
                try:
                    child = os.open(
                        entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root
                    )
                except OSError as exc:
                    if exc.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
                        continue
                    raise
                try:
                    walk(child, depth + 1)
                finally:
                    os.close(child)

    for root in roots:
        try:
            if os.name == "nt":
                walk(root)
                continue
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                walk(fd)
            finally:
                os.close(fd)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SandboxResourceLimit("Sandbox workspace could not be inspected") from exc


def remove_scratch(root):
    """Restore directory permissions only inside the stopped child's owned scratch."""
    import shutil

    def restore(fd):
        os.fchmod(fd, 0o700)
        with os.scandir(fd) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                if not stat.S_ISDIR(metadata.st_mode):
                    continue
                os.chmod(entry.name, 0o700, dir_fd=fd, follow_symlinks=False)
                child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    restore(child)
                finally:
                    os.close(child)

    try:
        os.chmod(root, 0o700, follow_symlinks=False)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            restore(fd)
        finally:
            os.close(fd)
        shutil.rmtree(root)
    except RecursionError:
        # Native traversal handles trees deeper than the Python stack without following links.
        options = {
            "check": True,
            "timeout": 30,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        subprocess.run(
            [
                "/usr/bin/find",
                "-P",
                str(root),
                "-xdev",
                "-type",
                "d",
                "-exec",
                "/usr/bin/chmod",
                "u+rwx",
                "--",
                "{}",
                ";",
            ],
            **options,
        )
        subprocess.run(["/usr/bin/rm", "-rf", "--", str(root)], **options)
