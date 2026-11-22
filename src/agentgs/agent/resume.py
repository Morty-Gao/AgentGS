"""Restore a saved trajectory and re-enter the model call that did not finish."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace

from agentgs.agent.planning.instances import ExecutionPlan
from agentgs.client.context import token_bound
from agentgs.contracts.models import AgentResult
from agentgs.messages import Conversation

_UNFINISHED = {"model.started", "model.failed", "model.cancelled"}


class RestoredPlan(SimpleNamespace):
    def as_dict(self):
        return deepcopy(vars(self))


@dataclass
class ResumeCursor:
    stage: str | None
    messages: list
    usage: list
    tool_results: list
    model_calls: int
    token_charge: int
    http_attempts: int
    physical_tool_calls: int
    diagnostic_tool_calls: int
    resend: bool
    replay: bool
    continue_plan: bool
    answer_only: bool
    skip_skill_layer: bool
    chain: list = field(default_factory=list)
    instances: object | None = None
    restored_outputs: list = field(default_factory=list)
    restored_plan: RestoredPlan | None = None
    bridge_report: dict | None = None
    tool_chain_completed: bool = False
    action_budget_exhausted: bool = False
    evidence: str = ""


def prepare_resume(result) -> ResumeCursor:
    """Drop an unfinished model request and return the charge that was only reserved."""
    if not isinstance(result, AgentResult):
        result = AgentResult.model_validate(result)
    messages = [message.model_copy(deep=True) for message in result.messages]
    request_id = _unfinished_request(messages)
    stage = _request_stage(messages, request_id) if request_id else None
    refund = _unsettled_reserve(messages, request_id) if request_id else 0
    if request_id:
        messages = [
            message
            for message in messages
            if not (
                message.role == "event"
                and message.metadata.get("request_id") == request_id
                and str(message.metadata.get("type", "")).startswith("model.")
            )
        ]
    conversation = Conversation(messages=messages)
    if conversation.pending_tool_calls():
        raise ValueError("Cannot resume a task with unresolved historical tool calls")
    dialogue = [message for message in messages if message.role not in {"system", "event"}]
    if not any(message.role == "user" for message in dialogue):
        raise ValueError("Cannot resume a trajectory that has no user turn")
    if stage is None and dialogue and dialogue[-1].metadata.get("sent_to_model"):
        stage = dialogue[-1].metadata.get("stage")
    provenance = result.provenance or {}
    budget = provenance.get("budget") or {}
    cost = provenance.get("execution_cost") or {}
    model_calls = cost.get("model_requests")
    if not isinstance(model_calls, int):
        model_calls = max((row.get("call") or 0 for row in result.usage), default=0)
    token_charge = budget.get("token_charge") or 0
    if not isinstance(token_charge, int):
        token_charge = 0
    http_attempts = budget.get("http_attempts") or 0
    if not isinstance(http_attempts, int):
        http_attempts = 0
    chain, plan_at = _chain(messages)
    outputs = _completed_outputs(messages, chain, plan_at)
    restored_plan = _restored_plan(messages)
    bridge = _event_content(messages, "shared_capability_bridge_search")
    continue_plan = stage == "tool_argument_generation" and bool(chain)
    answer_only = stage == "final_answer_generation"
    skip_skill_layer = (
        restored_plan is not None
        and stage == "skill_tool_planning"
        and not continue_plan
        and not answer_only
    )
    replay = stage == "capability_completion" or (
        stage == "skill_tool_planning" and restored_plan is None
    )
    resend = bool(
        stage
        and dialogue
        and dialogue[-1].metadata.get("sent_to_model")
        and dialogue[-1].metadata.get("stage") == stage
    )
    instances = _instances(messages, chain) if continue_plan else None
    return ResumeCursor(
        stage=stage if isinstance(stage, str) else None,
        messages=messages,
        usage=deepcopy(result.usage),
        tool_results=[item.model_copy(deep=True) for item in result.tool_results],
        model_calls=model_calls,
        token_charge=max(0, token_charge - refund),
        http_attempts=http_attempts,
        physical_tool_calls=_count(cost.get("tool_executor_requests")),
        diagnostic_tool_calls=_count(cost.get("diagnostic_tool_executor_requests")),
        resend=resend,
        replay=replay,
        continue_plan=continue_plan,
        answer_only=answer_only,
        skip_skill_layer=skip_skill_layer,
        chain=chain,
        instances=instances,
        restored_outputs=outputs,
        restored_plan=restored_plan,
        bridge_report=bridge if isinstance(bridge, dict) else None,
        tool_chain_completed=result.tool_chain_completed,
        action_budget_exhausted=result.action_budget_exhausted,
        evidence=_evidence(messages),
    )


def resumable_failure(exc: BaseException) -> bool:
    """Deadline and budget failures can continue; protocol and transport corruption cannot."""
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        if name == "ModelProtocolError":
            return False
        text = str(current)
        if (
            name == "BudgetExceeded"
            or "Task token budget exhausted" in text
            or "LLM request deadline exceeded" in text
        ):
            return True
        current = current.__cause__
    return False


def _count(value) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _unfinished_request(messages) -> str | None:
    latest = {}
    order = []
    for message in messages:
        if message.role != "event":
            continue
        kind = str(message.metadata.get("type", ""))
        request_id = message.metadata.get("request_id")
        if not kind.startswith("model.") or not request_id:
            continue
        if request_id not in latest:
            order.append(request_id)
        latest[request_id] = kind
    answered = {
        message.metadata.get("request_id")
        for message in messages
        if message.role == "assistant" and message.metadata.get("request_id")
    }
    for request_id in reversed(order):
        if request_id in answered:
            return None
        if latest[request_id] in _UNFINISHED:
            return request_id
        return None
    return None


def _request_stage(messages, request_id):
    stage = None
    for message in messages:
        if message.role != "event" or message.metadata.get("request_id") != request_id:
            continue
        if message.metadata.get("stage"):
            stage = message.metadata["stage"]
    return stage


def _reserve(metadata) -> int:
    request = metadata.get("payload")
    if not isinstance(request, dict):
        return 0
    if "messages" not in request and isinstance(request.get("payload"), dict):
        request = request["payload"]
    limit = request.get("max_tokens", request.get("max_completion_tokens", 0)) or 0
    try:
        return token_bound(request) + int(limit)
    except (TypeError, ValueError):
        return 0


def _response_total(metadata):
    body = metadata.get("body")
    if not isinstance(body, str):
        return None
    try:
        usage = json.loads(body).get("usage") or {}
    except ValueError:
        return None
    total = usage.get("total_tokens")
    if isinstance(total, int) and total >= 0:
        return total
    return None


def _unsettled_reserve(messages, request_id) -> int:
    """A timed-out call keeps its upper bound until a response reports real usage."""
    refund = 0
    pending = 0
    for message in messages:
        if message.role != "event" or message.metadata.get("request_id") != request_id:
            continue
        kind = message.metadata.get("type")
        if kind == "model.transport_request":
            pending = _reserve(message.metadata)
            refund += pending
        elif kind == "model.transport_response":
            if _response_total(message.metadata) is not None:
                refund -= pending
            pending = 0
    return max(0, refund)


def _event_content(messages, kind):
    content = None
    for message in messages:
        if message.role == "event" and message.metadata.get("type") == kind:
            content = message.metadata.get("content", message.metadata.get("requirements"))
    return content


def _restored_plan(messages):
    content = _event_content(messages, "skill_plan")
    if not isinstance(content, dict):
        return None
    return RestoredPlan(**deepcopy(content))


def _chain(messages):
    chain, index = [], None
    for position, message in enumerate(messages):
        if message.role != "event" or message.metadata.get("type") != "plan.created":
            continue
        tools = message.metadata.get("tools")
        if isinstance(tools, list) and tools and all(isinstance(name, str) for name in tools):
            chain, index = list(tools), position
    return chain, index


def _calls(messages):
    calls = {}
    for message in messages:
        if message.role == "assistant":
            for call in message.tool_calls:
                calls[call.id] = call
    return calls


def _raw_output(message):
    raw = message.metadata.get("raw_tool_output", message.content)
    return raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)


def _completed_outputs(messages, chain, plan_at):
    if not chain or plan_at is None:
        return []
    calls = _calls(messages)
    outputs = []
    for message in messages[plan_at + 1 :]:
        if message.role != "tool" or message.metadata.get("status") != "success":
            continue
        if message.metadata.get("operation") in {"planning", "protocol"}:
            continue
        call = calls.get(message.tool_call_id)
        if call is None or len(outputs) >= len(chain) or call.function.name != chain[len(outputs)]:
            break
        outputs.append(_raw_output(message))
    return outputs


def _submitted_steps(messages):
    calls = _calls(messages)
    steps = None
    for message in messages:
        if message.role != "tool" or message.metadata.get("status") != "success":
            continue
        call = calls.get(message.tool_call_id)
        if call is None or call.function.name != "submit_tool_plan":
            continue
        try:
            payload = json.loads(_raw_output(message))
        except ValueError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("steps"), list) and payload["steps"]:
            steps = payload["steps"]
    return steps


def _instances(messages, chain):
    steps = _submitted_steps(messages)
    if steps:
        try:
            return ExecutionPlan(steps).align(chain)
        except (ValueError, TypeError):
            pass
    return ExecutionPlan(
        [
            {"id": f"resume_{index}", "tool": name, "purpose": f"Execute {name}"}
            for index, name in enumerate(chain, 1)
        ]
    )


def _evidence(messages) -> str:
    calls = _calls(messages)
    parts = []
    for message in messages:
        if message.role != "tool" or message.metadata.get("status") != "success":
            continue
        if message.metadata.get("operation") in {"planning", "protocol"}:
            continue
        call = calls.get(message.tool_call_id)
        if call is None:
            continue
        raw = _raw_output(message)
        parts.append(
            f"\n\nNow we have run the **{call.function.name}** with input `{{}}` "
            f"and obtained result\n {raw}."
        )
    return "".join(parts)
