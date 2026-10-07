"""Build React frontend using the project-local versioned Node installation."""

import os
import subprocess
from pathlib import Path


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    node = root / "tools" / "bin" / ("node.exe" if os.name == "nt" else "node")
    npm = Path((root / "tools" / "state" / "npm-path.txt").read_text())
    frontend = root / "src" / "agentgs" / "visualization" / "frontend"
    env = {**os.environ, "PATH": str(node.parent) + os.pathsep + os.environ.get("PATH", "")}
    subprocess.run(
        [
            str(node),
            str(npm),
            "ci" if (frontend / "package-lock.json").exists() else "install",
            "--no-audit",
            "--no-fund",
        ],
        cwd=frontend,
        env=env,
        check=True,
    )
    subprocess.run([str(node), str(npm), "run", "build"], cwd=frontend, env=env, check=True)


if __name__ == "__main__":
    main()
