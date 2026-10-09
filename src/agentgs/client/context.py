"""Conservative request bounds and explicit, pair-preserving history reduction."""

import json
from copy import deepcopy

from agentgs.client.errors import BudgetExceeded


def token_bound(value):
    # UTF-8 bytes plus framing is a conservative bound, not a tokenizer measurement.
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()) + 64


def fit_context(payload, context_window):
    request = deepcopy(payload)
    output = request.get("max_tokens", request.get("max_completion_tokens", 0))
    limit = context_window - output
    if token_bound(request) <= limit:
        return request, False
    messages = request["messages"]
    user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=0)
    messages[:] = [m for m in messages[:user] if m["role"] == "system"] + messages[user:]
    for message in messages[:-1]:
        if token_bound(request) <= limit:
            break
        if message["role"] == "tool":
            text = message.get("content") or ""
            message["content"] = json.dumps({
                "context_omitted": True, "tool_call_id": message["tool_call_id"],
                "original_chars": len(text),
                "note": "Earlier result omitted for context budget. Do not infer its values.",
            })
    if token_bound(request) > limit:
        raise BudgetExceeded("Required system, task, schemas and current feedback exceed context budget")
    return request, True
