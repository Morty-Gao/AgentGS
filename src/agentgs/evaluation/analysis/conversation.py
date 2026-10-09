"""Evaluation views derived from the canonical conversation, never a saved trace."""

from collections.abc import Iterable
from typing import Any

from agentgs.messages import Conversation, Message


def as_conversation(
    value: Conversation | dict[str, Any] | Iterable[Message | dict[str, Any]] | None,
) -> Conversation:
    if isinstance(value, Conversation):
        return value
    if isinstance(value, dict):
        return Conversation.model_validate(value)
    return Conversation(messages=list(value) if value is not None else [])


def tool_executions(value) -> list[dict[str, Any]]:
    """Completed domain invocations, including failed/repaired calls.

    Only the current user turn is scored. Planning calls and nested comparison
    branches are not task executions. Parallel replies use invocation order,
    rather than the order in which tools happen to finish.
    Conversation resolves call IDs and prefers runtime-bound tool_input metadata
    while retaining the original function arguments in the message log.
    """
    conversation = as_conversation(value)
    call_order = {
        call.id: index
        for index, call in enumerate(
            call
            for message in _current_turn(conversation)
            if message.role == "assistant"
            for call in message.tool_calls
        )
        if call.function.name not in {"submit_tool_plan", "select_capabilities"}
    }
    # Event annotations can contain historical tool_call-shaped payloads. Only
    # assistant/tool message pairs establish an actual invocation.
    dialogue = Conversation(messages=[m for m in conversation.messages if m.role != "event"])
    events = [
        event
        for event in dialogue.execution_events()
        if event.get("type") in {"tool_call", "tool_retry"} and event.get("call_id") in call_order
    ]
    return sorted(events, key=lambda event: call_order[event["call_id"]])


def _current_turn(conversation):
    start = next(
        (
            i
            for i in range(len(conversation.messages) - 1, -1, -1)
            if conversation.messages[i].role == "user"
        ),
        0,
    )
    return conversation.messages[start:]


def model_usage(value) -> list[dict[str, Any]]:
    """One usage row per request, preserving failed and unfinished requests."""
    conversation = as_conversation(value)
    requests = {}
    for message in _current_turn(conversation):
        metadata = message.metadata
        request_id = metadata.get("request_id")
        if not request_id:
            continue
        kind = metadata.get("type") if message.role == "event" else None
        if kind == "model.started":
            requests.setdefault(request_id, {"usage": None, "attempts": None})
            requests[request_id].update(status="started")
        elif kind in {
            "model.completed",
            "model.failed",
            "model.cancelled",
            "model.protocol_failed",
        }:
            row = requests.setdefault(request_id, {"usage": None, "attempts": None})
            row.update(metadata)
            row["status"] = kind.removeprefix("model.")
        elif message.role == "assistant":
            # Also supports a checkpoint taken after the reply but before its
            # model.completed annotation has been written.
            row = requests.setdefault(request_id, {"usage": None, "attempts": None})
            row.update(metadata)
    return list(requests.values())


def usage_summary(calls) -> dict[str, Any]:
    """Normalize raw request usage or an already aggregated prediction summary."""
    if isinstance(calls, dict):
        complete = calls.get("usage_complete") is True
        total = calls.get("total_tokens")
        complete = (
            complete and isinstance(total, int) and not isinstance(total, bool) and total >= 0
        )
        return {
            "usage_complete": complete,
            "total_tokens": total if complete else None,
            "reported_tokens": calls.get("reported_tokens", total) or 0,
        }
    calls = list(calls or [])
    totals = []
    complete = bool(calls)
    for call in calls:
        usage = call.get("usage") if isinstance(call, dict) else None
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        valid = isinstance(total, int) and not isinstance(total, bool) and total >= 0
        complete = complete and valid and call.get("attempts", 1) == 1
        complete = complete and call.get("status", "completed") in {"completed", "protocol_failed"}
        if valid:
            totals.append(total)
    return {
        "usage_complete": bool(complete),
        "total_tokens": sum(totals) if complete else None,
        "reported_tokens": sum(totals),
    }


def conversation_usage(value, fallback=None) -> dict[str, Any]:
    """Conversation request evidence takes precedence over legacy usage fields."""
    calls = model_usage(value)
    return usage_summary(calls if calls else fallback)
