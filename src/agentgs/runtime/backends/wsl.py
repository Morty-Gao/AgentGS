"""Windows-to-WSL bridge without shell interpolation."""

import subprocess
from pathlib import PurePosixPath


def run(project: str, arguments: list[str], distribution: str = "Ubuntu-24.04") -> int:
    root = PurePosixPath(project)
    if not root.is_absolute():
        raise ValueError("WSL project path must be absolute")
    return subprocess.call(
        [
            "wsl.exe",
            "-d",
            distribution,
            "--cd",
            str(root),
            "--exec",
            str(root / ".venv/bin/python"),
            "-m",
            "agentgs",
            *arguments,
        ]
    )
