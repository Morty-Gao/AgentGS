"""Terminateable subprocess execution of the registered scientific kernels."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from agentgs.contracts.async_utils import drain, invoke
from agentgs.runtime.subprocesses import collect, spawn, stop


class ProcessToolRunner:
    async def run(self, root, artifacts, call, *, rs_enabled=True):
        work = Path(artifacts).parent / "tool_runs" / call.id
        request = work / "request.json"
        result_path = work / "result.json"

        def prepare():
            work.mkdir(parents=True, exist_ok=False)
            request.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "artifacts": str(artifacts),
                        "call": call.model_dump(mode="json"),
                        "internal": call._runtime_arguments,
                        "rs_enabled": rs_enabled,
                    }
                ),
                encoding="utf-8",
            )

        await invoke(prepare)
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
        process = await spawn(
            sys.executable,
            "-m",
            "agentgs.runtime.tools.process",
            str(request),
            cwd=root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        try:
            _, stderr = await collect(process)
            if not result_path.exists():
                raise RuntimeError(
                    f"Scientific worker exited {process.returncode}: {stderr[:1000]}"
                )
            result = await invoke(lambda: json.loads(result_path.read_text("utf-8")))
            if "error" in result:
                kinds = {
                    "ValueError": ValueError,
                    "FileNotFoundError": FileNotFoundError,
                    "ImportError": ImportError,
                    "KeyError": KeyError,
                }
                raise kinds.get(result["error_type"], RuntimeError)(result["error"])
            return result["data"]
        finally:
            await drain(asyncio.create_task(stop(process)))


def main():
    from agentgs.contracts.models import ToolCall
    from agentgs.contracts.remote_sensing import contract_scope
    from agentgs.datasets.registry import DatasetRegistry
    from agentgs.runtime.tools.compute import compute_tool
    from agentgs.runtime.tools.context import tool_context
    from agentgs.runtime.tools.registry import builtin_tools

    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    request = Path(parser.parse_args().request)
    data = json.loads(request.read_text("utf-8"))
    call = ToolCall.model_validate(data["call"])
    call._runtime_arguments = data["internal"]
    try:
        with (
            contract_scope(data["rs_enabled"]),
            tool_context(
                data["root"], Path(data["artifacts"]) / "calls" / call.id, data["artifacts"]
            ),
        ):
            value = compute_tool(
                call.tool,
                {**call.arguments, **call._runtime_arguments},
                DatasetRegistry(Path(data["root"])),
                builtin_tools(),
            )
        result = {"data": value}
    except Exception as exc:
        result = {"error_type": type(exc).__name__, "error": str(exc)}
    (request.parent / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
