"""Supervise a full benchmark and persist its console log and process exit status."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=128, choices=range(1, 129))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--rss-limit-mb", type=int, default=2048)
    args = parser.parse_args()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "launch.json").exists():
        raise FileExistsError("Use a fresh experiment directory; resume via the benchmark CLI")
    command = [
        sys.executable,
        "-u",
        "-m",
        "agentgs",
        "evaluate",
        "benchmark",
        "--project",
        str(root),
        "--output",
        str(output),
        "--concurrency",
        str(args.concurrency),
        "--timeout",
        str(args.timeout),
        "--rss-limit-mb",
        str(args.rss_limit_mb),
    ]
    started = time.monotonic()
    with (output / "experiment.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        receipt = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "supervisor_pid": os.getpid(),
            "runner_pid": process.pid,
            "command": command,
            "console_log": str(output / "experiment.log"),
        }
        (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
        code = process.wait()
    (output / "exit.json").write_text(
        json.dumps(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "returncode": code,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
