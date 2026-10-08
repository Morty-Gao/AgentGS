"""Transfer this workspace and its prepared data to a Linux server over SSH."""

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", help="SSH destination, such as user@server")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--destination", default="AgentGS/", help="Remote project directory")
    parser.add_argument("--dry-run", action="store_true", help="List transfers; still requires SSH")
    parser.add_argument("--include-env", action="store_true", help="Also transfer the local .env")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.host) or args.host.startswith("-"):
        parser.error("Use an SSH hostname or IPv4 address, optionally prefixed with user@")
    if not 1 <= args.port <= 65535:
        parser.error("SSH port must be between 1 and 65535")
    if not args.destination or args.destination.startswith("-"):
        parser.error("Supply a remote directory")
    if not shutil.which("rsync") or not shutil.which("ssh"):
        parser.error("Run from Linux/WSL with rsync and ssh installed")
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    command = [
        "rsync",
        "-aL",
        "--protect-args",
        "--partial",
        "--info=progress2",
        "-e",
        f"ssh -p {args.port}",
    ]
    for pattern in (
        "/.venv/",
        "/tools/",
        "/outputs/",
        "/.git/",
        "/tmp/",
        "/dist/",
        "/forks/",
        "node_modules/",
        "__pycache__/",
        ".pytest_cache/",
        ".ruff_cache/",
        "*.docx",
    ):
        command.append(f"--exclude={pattern}")
    if not args.include_env:
        command.extend(["--include=/.env.example", "--exclude=/.env", "--exclude=/.env.*"])
    if args.dry_run:
        command.append("--dry-run")
    command.extend([str(root) + "/", f"{args.host}:{args.destination.rstrip('/')}/"])
    print("Transferring code and the entire prepared data directory; rerun to resume.", flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
