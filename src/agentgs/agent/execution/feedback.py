"""Agent argument repair over typed execution feedback."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence
from uuid import uuid4

from agentgs.contracts.async_utils import invoke
from agentgs.contracts.tool_feedback import ExecutionFeedback, assess_tool_execution, stringify
from agentgs.messages import Conversation


@dataclass(frozen=True)
class ToolAttempt:
    attempt: int
    tool_input: str
    tool_output: str
    feedback: ExecutionFeedback

    def to_prompt_context(self) -> dict:
        return {
            "attempt": self.attempt,
            "input": self.tool_input,
            "output": _shorten(self.tool_output),
            "feedback": {
                "status": self.feedback.status,
                "error_type": self.feedback.error_type,
                "summary": self.feedback.summary,
                "suggestion": self.feedback.suggestion,
            },
        }


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    succeeded: bool
    final_input: str
    output: str
    attempts: tuple[ToolAttempt, ...]

    @property
    def final_feedback(self) -> ExecutionFeedback:
        return self.attempts[-1].feedback

    def render_for_agent(self) -> str:
        if self.succeeded:
            return f"\n\nNow we have run the **{self.tool_name}** with input `{self.final_input}` and obtained result\n {self.output}."
        return f"\n\nTool **{self.tool_name}** failed after {len(self.attempts)} attempts.\nExecution feedback: {self.final_feedback.summary}\nLast tool output: {self.output}"


@dataclass(frozen=True)
class PreparedToolCall:
    """An original native call plus the arguments resolved by the framework.

    Missing calls and malformed arguments produce feedback without inventing an
    assistant call. Runtime-only calls use the string adapter below.
    """
    call_id: str | None
    tool_input: str
    error_output: str | None = None


async def run_tool_with_feedback(
    *,
    tool_name: str,
    file_path_list: list,
    input_provider: InputProvider,
    tool_runner: ToolRunner,
    input_format: str = "",
    max_attempts: int = 3,
    messages: Conversation | None = None,
    checkpoint: Callable | None = None,
    producer_step: str = "",
) -> ToolExecutionResult:
    """Execute, assess, and (when possible) repair one tool call."""
    max_attempts = max(1, int(max_attempts))
    attempts: list[ToolAttempt] = []
    for attempt_number in range(1, max_attempts + 1):
        supplied = await input_provider(attempt_number, tuple(attempts))
        native = supplied if isinstance(supplied, PreparedToolCall) else None
        tool_input = native.tool_input if native else str(supplied).strip()
        from contextlib import nullcontext

        from agentgs.agent.execution import scope as migration
        from agentgs.agent.execution.references import binding_error_response

        binding_error = native.error_output if native and native.error_output else binding_error_response(tool_input)
        call_context = (
            migration.call_scope(native.call_id if native else None)
            if migration.active() and binding_error is None
            else nullcontext()
        )
        identity = {}
        produced_records = []
        message_call_id = native.call_id if native else uuid4().hex
        try:
            with call_context as call_id:
                if call_id:
                    identity = {"request_id": migration.current_request_id(), "call_id": call_id}
                    if native is None:
                        message_call_id = call_id
                if messages is not None and native is None:
                    messages.begin_tool(tool_name, tool_input, call_id=message_call_id)
                    if checkpoint is not None:
                        await invoke(checkpoint)
                try:
                    if binding_error is not None:
                        raw_output = binding_error
                    elif native:
                        raw_output = await tool_runner(tool_name, tool_input, file_path_list, call_id=message_call_id)
                    else:
                        raw_output = await tool_runner(tool_name, tool_input, file_path_list)
                except Exception as exc:
                    raw_output = f"Tool function execution error, {type(exc).__name__}: {exc}"
                registry = migration.current_registry()
                if identity and registry is not None:
                    try:
                        produced_records = await invoke(registry.ingest_tool_output, 
                            tool_name,
                            producer_step,
                            raw_output,
                            execution_info={
                                **identity,
                                "attempt": attempt_number,
                                "succeeded": assess_tool_execution(
                                    raw_output,
                                    input_format=input_format,
                                    attempt=attempt_number,
                                    max_attempts=max_attempts,
                                ).status
                                == "passed",
                            },
                        )
                    except Exception as exc:
                        if messages is not None:
                            messages.record_event(
                                "artifact_provenance_error",
                                {
                                    **identity,
                                    "error_type": type(exc).__name__,
                                },
                            )
        except asyncio.CancelledError:
            if messages is not None and message_call_id is not None:
                if any(call.id == message_call_id for call in messages.pending_tool_calls()):
                    messages.finish_tool(message_call_id, json.dumps({"status": "cancelled"}), status="cancelled")
                messages.record_event("tool.cancelled", {"tool_name": tool_name, "call_id": message_call_id})
                if checkpoint is not None:
                    await invoke(checkpoint)
            raise
        except Exception as exc:
            raw_output = f"Tool function execution error, {type(exc).__name__}: {exc}"
        tool_output = _stringify_output(raw_output)
        feedback = assess_tool_execution(
            tool_output,
            input_format=input_format,
            attempt=attempt_number,
            max_attempts=max_attempts,
        )
        record = ToolAttempt(
            attempt=attempt_number,
            tool_input=tool_input,
            tool_output=tool_output,
            feedback=feedback,
        )
        attempts.append(record)
        if messages is not None:
            metadata = {
                "attempt": attempt_number,
                "status": "success" if feedback.status == "passed" else feedback.status,
                **{key: value for key, value in identity.items() if key != "call_id"},
            }
            if message_call_id is not None:
                if native:
                    metadata["tool_input"] = tool_input
                    metadata["argument_source"] = "native_function_call"
                messages.finish_tool(message_call_id, tool_output, **metadata)
            event = feedback.to_event(
                tool_name=tool_name, attempt=attempt_number, max_attempts=max_attempts
            )
            messages.record_event(event.pop("type"), event)
            if produced_records:
                messages.record_event(
                    "artifact_production",
                    {
                        **identity,
                        "artifacts": [record.as_dict() for record in produced_records],
                    },
                )
            if checkpoint is not None:
                await invoke(checkpoint)
        if feedback.status == "passed":
            return ToolExecutionResult(
                tool_name=tool_name,
                succeeded=True,
                final_input=tool_input,
                output=tool_output,
                attempts=tuple(attempts),
            )
    return ToolExecutionResult(
        tool_name=tool_name,
        succeeded=False,
        final_input=attempts[-1].tool_input,
        output=attempts[-1].tool_output,
        attempts=tuple(attempts),
    )


InputProvider = Callable[[int, Sequence[ToolAttempt]], Awaitable[str | PreparedToolCall]]
ToolRunner = Callable[[str, str, list], Awaitable[Any]]

_stringify_output = stringify


def _shorten(text, limit=800):
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"
