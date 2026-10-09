"""Bash exploration without registered scientific tools or skills."""

import json
from pathlib import Path

from agentgs.contracts.models import ToolCall
from agentgs.contracts.sandbox import BashInput, SandboxUnavailable

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run one shell command. /data is the read-only stored data. "
            "/work is writable and keeps files across calls. The shell process "
            "starts in /work each time. There is no network."
        ),
        "parameters": BashInput.model_json_schema(),
    },
}


def parse_bash_arguments(raw):
    return BashInput.model_validate(json.loads(raw)).command


def _instruction(question, *, first, described):
    if not first:
        return (
            "Continue from the bash results. Call bash again when a value is still "
            "missing. When the observations are sufficient, reply with those values "
            "and do not call bash."
        )
    layout = (
        "/data is a read-only view of the existing files at their stored relative paths."
        if described["data_mount"] == "bwrap_ro"
        else "Use these actual stored paths: " + json.dumps(described["mounts"])
    )
    network = (
        "Network access is blocked."
        if described["network_isolated"]
        else "This desktop demo has no enforced network or filesystem isolation. Do not use network."
    )
    work = described.get("work_path", "/work")
    return f"""
Explore the stored data with the bash tool and collect the measurements the question asks for.
{layout}
{work} is writable and keeps files between calls. Each call starts a new shell there.
{network} There is no registered scientific tool. Read filenames, headers and values yourself.
The evaluation locator below is a hint, not a file listing and not a renamed path.

When the observations are sufficient, stop calling bash and reply with the values you found.
A later step writes the formatted final answer.

Question and public resource locator:
{question}
"""


async def explore(question, state, stages, answering):
    executor = getattr(state.context, "code_executor", None)
    if executor is None:
        raise SandboxUnavailable("data_only requires the configured code sandbox")
    data_root = Path(executor.root) / "data"
    described = await executor.describe_shell(data_root)
    state.provenance_extra = {
        "data_mount": described["data_mount"],
        "data_paths": described["mounts"],
        "isolation": described["isolation"],
        "network_isolated": described["network_isolated"],
        "private_proc": described.get("private_proc", False),
    }
    await state.emit(
        "data_sandbox",
        {
            "data_mount": described["data_mount"],
            "mounts": described["mounts"],
            "isolation": described["isolation"],
            "network_isolated": described["network_isolated"],
        },
    )
    bash_tool = {
        **BASH_TOOL,
        "function": {
            **BASH_TOOL["function"],
            "description": (
                BASH_TOOL["function"]["description"]
                if described["data_mount"] == "bwrap_ro"
                else "Run Bash in "
                + described["work_path"]
                + ". Data paths: "
                + json.dumps(described["mounts"])
                + ". "
                + (
                    "Network blocked; data read-only."
                    if described["network_isolated"]
                    else "Desktop demo: no enforced network or filesystem isolation."
                )
            ),
        },
    }
    limit = state.context.options.command_attempts
    transcript = []
    used = 0
    completed = False
    first = True
    while used < limit:
        message = await state.model(
            _instruction(question, first=first, described=described),
            stage="data_exploration",
            tools=[bash_tool],
            tool_choice="auto",
        )
        first = False
        if not message.tool_calls:
            completed = any(result.succeeded for result in state.results)
            break
        for call in message.tool_calls:
            if used >= limit:
                payload = {"status": "error", "error": "bash command budget exhausted"}
                state.conversation.finish_tool(call.id, json.dumps(payload), status="failed")
                transcript.append(payload)
                continue
            used += 1
            payload = await _run_bash(state, call)
            transcript.append({"command_index": used, **payload})
    evidence = (
        json.dumps(transcript, ensure_ascii=False) if transcript else "No bash commands were run."
    )
    return await answering.generate(
        question,
        evidence,
        completed=completed,
        tool_attempts_exhausted=used >= limit and not completed,
    )


async def _run_bash(state, call):
    if call.function.name != "bash":
        payload = {"status": "error", "error": "Only the bash tool is available"}
        state.conversation.finish_tool(call.id, json.dumps(payload), status="failed")
        await state.save()
        return payload
    try:
        command = parse_bash_arguments(call.function.arguments)
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        payload = {"status": "error", "error": str(exc)}
        state.conversation.finish_tool(call.id, json.dumps(payload), status="failed")
        await state.save()
        return payload
    result = await state.execute(ToolCall(id=call.id, tool="bash", arguments={"command": command}))
    payload = {
        "command": command,
        "status": "ok" if result.succeeded else "error",
        **result.data,
    }
    if result.error:
        payload["error"] = result.error.model_dump(mode="json")
    if result.data.get("truncated"):
        payload["note"] = "Output truncated to 16384 bytes per stream."
    await state.record(
        {
            "type": "tool_call",
            "call_id": call.id,
            "tool_name": "bash",
            "tool_input": {"command": command},
            "tool_output": json.dumps(payload, ensure_ascii=False),
            "status": "success" if result.succeeded else "failed",
        }
    )
    return payload
