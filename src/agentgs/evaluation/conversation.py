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

    Planning calls and nested comparison branches are not task executions.
    Conversation resolves call IDs and prefers runtime-bound tool_input metadata
    while retaining the original function arguments in the message log.
    """
    return [
        event
        for event in as_conversation(value).execution_events()
        if event.get("type") in {"tool_call", "tool_retry"}
    ]


def model_usage(value) -> list[dict[str, Any]]:
    """One usage row per request, preserving failed and unfinished requests."""
    conversation = as_conversation(value)
    requests = {}
    for message in conversation.messages:
        metadata = message.metadata
        request_id = metadata.get("request_id")
        if not request_id:
            continue
        kind = metadata.get("type") if message.role == "event" else None
        if kind == "model.started":
            requests.setdefault(request_id, {"usage": None, "attempts": None})
            requests[request_id].update(status="started")
        elif kind in {"model.completed", "model.failed"}:
            row = requests.setdefault(request_id, {"usage": None, "attempts": None})
            row.update(metadata)
            row["status"] = "completed" if kind == "model.completed" else "failed"
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
        complete = complete and isinstance(total, int) and not isinstance(total, bool) and total >= 0
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
        complete = complete and call.get("status", "completed") == "completed"
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
