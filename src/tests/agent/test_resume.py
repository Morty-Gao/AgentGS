"""Resume restores a saved trajectory without repeating finished tool steps."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.prompts import system_prompt
from agentgs.agent.resume import prepare_resume
from agentgs.agent.state import AgentExecutionError, ModelProtocolError, RunState
from agentgs.client.context import token_bound
from agentgs.client.errors import BudgetExceeded, TransportFailure
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.models import AgentResult, TaskInput, ToolResult, ToolSpec
from agentgs.messages import Function, FunctionCall, Message, RuntimeEvent


def _event(kind, **payload):
    return RuntimeEvent(metadata={"type": kind, **payload})


def _call(name, arguments, call_id):
    return FunctionCall(
        id=call_id, function=Function(name=name, arguments=json.dumps(arguments))
    )


def test_legacy_agent_result_parses_without_resume_fields():
    result = AgentResult.model_validate(
        {
            "answer": "2",
            "conversation_schema": 2,
            "messages": [{"role": "user", "content": "Find x"}],
        }
    )
    assert result.conversation_schema == 2
    assert result.provenance == {}
    task = TaskInput(question="Find x")
    assert task.resume is None and task.resume_attempts == 0


def test_unfinished_model_request_is_dropped_and_reserve_refunded():
    settled = {"messages": [{"role": "user", "content": "Find the mean"}], "max_tokens": 8}
    pending = {
        "messages": [
            {"role": "user", "content": "Find the mean"},
            {"role": "tool", "content": '{"value": 2}'},
        ],
        "max_tokens": 1000,
    }
    pending_reserve = token_bound(pending) + 1000
    response = json.dumps({"usage": {"total_tokens": 20}})
    done = "req-done"
    failed = "req-failed"
    messages = [
        Message(role="system", content=system_prompt("tool_only")),
        Message(
            role="user",
            content="Find the mean",
            metadata={
                "sent_to_model": True,
                "stage": "tool_planning",
                "original_question": "Find the mean",
            },
        ),
        _event(
            "model.transport_request",
            request_id=done,
            stage="tool_planning",
            payload=settled,
        ),
        _event(
            "model.transport_response",
            request_id=done,
            stage="tool_planning",
            body=response,
        ),
        _event("model.completed", request_id=done, stage="tool_planning"),
        Message(
            role="assistant",
            tool_calls=[_call("RemoteSensingMean", {"x": [1, 2]}, "call-1")],
            metadata={"stage": "tool_planning", "request_id": done, "sent_to_model": True},
        ),
        Message(
            role="tool",
            tool_call_id="call-1",
            content='{"value": 2}',
            metadata={
                "status": "success",
                "sent_to_model": True,
                "stage": "tool_argument_generation",
                "raw_tool_output": '{"value": 2}',
            },
        ),
        _event("model.started", request_id=failed, stage="tool_argument_generation"),
        _event(
            "model.transport_request",
            request_id=failed,
            stage="tool_argument_generation",
            payload=pending,
        ),
        _event(
            "model.failed",
            request_id=failed,
            stage="tool_argument_generation",
            error_type="TransportFailure",
        ),
    ]
    result = AgentResult(
        answer="",
        messages=messages,
        usage=[
            {"call": 1, "stage": "tool_planning", "usage": {"total_tokens": 20}, "status": "completed"},
            {"call": 2, "stage": "tool_argument_generation", "usage": None, "status": "failed"},
        ],
        completion_status="Not Completed",
        tool_chain_completed=False,
        answer_generation_status="not_started",
        provenance={
            "execution_cost": {
                "model_requests": 2,
                "tool_executor_requests": 1,
                "diagnostic_tool_executor_requests": 0,
            },
            "budget": {"token_charge": 20 + pending_reserve, "http_attempts": 2},
        },
    )
    cursor = prepare_resume(result)
    assert cursor.token_charge == 20
    assert cursor.model_calls == 2
    assert cursor.physical_tool_calls == 1
    assert not any(
        message.role == "event" and message.metadata.get("request_id") == failed
        for message in cursor.messages
    )
    state = RunState(
        SimpleNamespace(
            options=AlgorithmSettings(),
            client=None,
            checkpoint=None,
            events=SimpleNamespace(emit=lambda *args: None),
        ),
        TaskInput(question="Find the mean", resume=result),
    )
    dialogue = [message for message in state.conversation.messages if message.role != "event"]
    assert [message.role for message in dialogue].count("user") == 1
    assert dialogue[-1].role == "tool"
    assert state._token_charge == 20
    with pytest.raises(ValueError, match="unresolved historical tool calls"):
        prepare_resume(
            AgentResult(
                answer="",
                messages=[
                    Message(role="system", content=system_prompt("tool_only")),
                    Message(role="user", content="Find the mean"),
                    Message(
                        role="assistant",
                        tool_calls=[_call("RemoteSensingMean", {"x": [1]}, "open-call")],
                    ),
                ],
            )
        )


def function_call(name, arguments, call_id=None):
    if name == "submit_tool_plan" and isinstance(arguments, dict) and set(arguments) == {"tools"}:
        arguments = {
            "steps": [
                {"id": f"s{index}", "tool": tool, "purpose": f"Compute {tool}"}
                for index, tool in enumerate(arguments["tools"], 1)
            ]
        }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id or uuid4().hex,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class Events:
    def emit(self, *args):
        return None


class KG:
    async def retrieve(self, query):
        return [
            [name, "has the functionality that", "Scalar calculation"]
            for name in ["RemoteSensingMean", "RemoteSensingDifference"]
        ]

    def get_tool_subgraph(self, name, **kwargs):
        return [[name, "inputs", "JSON object"], [name, "outputs", "JSON result"]]


class Tools:
    def __init__(self):
        self.calls = []

    def specs(self):
        return [
            ToolSpec(id=name, description="Scalar calculation", input_schema={"type": "object"})
            for name in ["RemoteSensingMean", "RemoteSensingDifference"]
        ]

    async def execute(self, call):
        self.calls.append(call)
        data = {"value": 2}
        return ToolResult(
            call_id=call.id, tool=call.tool, succeeded=True, data=data, raw_output=json.dumps(data)
        )


class Client:
    def __init__(self, failure=None):
        self.failure = failure
        self.failed = False
        self.stages = []
        self.requests = []

    async def chat(self, messages, **kwargs):
        stage = kwargs.get("stage")
        self.stages.append(stage)
        self.requests.append(messages)
        choice = kwargs.get("tool_choice")
        name = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
        if name == "RemoteSensingDifference" and not self.failed and self.failure is not None:
            self.failed = True
            raise self.failure
        if stage in {"skill_tool_planning", "tool_planning"} or name == "submit_tool_plan":
            return ChatReply(
                message=function_call(
                    "submit_tool_plan",
                    {"tools": ["RemoteSensingMean", "RemoteSensingDifference"]},
                )
            )
        if name:
            arguments = {"x": [1, 2, 3]} if name == "RemoteSensingMean" else {"a": 3, "b": 1}
            return ChatReply(message=function_call(name, arguments))
        return ChatReply(
            message={"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:2"}
        )


def context(tools, client, *, argument_attempts=3):
    return AgentContext(
        tools,
        None,
        Events(),
        {},
        None,
        client,
        tool_retriever=KG(),
        options=AlgorithmSettings(
            planning_mode="tool_only",
            allow_rule_planning=False,
            plan_attempts=1,
            argument_attempts=argument_attempts,
        ),
    )


@pytest.mark.parametrize(
    "failure",
    [
        TransportFailure("LLM request deadline exceeded"),
        BudgetExceeded("Task token budget exhausted"),
    ],
)
async def test_completed_tool_step_is_not_executed_again(failure):
    tools = Tools()
    client = Client(failure)
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic.", resume_attempts=1),
        context(tools, client),
    )
    assert [call.tool for call in tools.calls] == ["RemoteSensingMean", "RemoteSensingDifference"]
    assert client.stages.count("tool_planning") == 1
    assert client.stages.count("tool_argument_generation") == 3
    resumed = client.requests[-2]
    assert sum(message["role"] == "user" for message in resumed) == 1
    assert result.answer == "2"


async def test_saved_skill_plan_resumes_at_tool_planning():
    failed = "req-plan"
    messages = [
        Message(role="system", content=system_prompt("skill_guided")),
        Message(
            role="user",
            content="Find the mean",
            metadata={
                "sent_to_model": True,
                "stage": "capability_selection",
                "original_question": "Find the mean",
            },
        ),
        Message(
            role="assistant",
            tool_calls=[_call("select_capabilities", {"requirements": []}, "cap")],
            metadata={"stage": "capability_selection", "request_id": "req-cap", "sent_to_model": True},
        ),
        Message(
            role="tool",
            tool_call_id="cap",
            content='{"status": "selected", "requirements": {}}',
            metadata={"status": "success", "operation": "planning", "sent_to_model": True, "stage": "skill_tool_planning"},
        ),
        _event(
            "skill_plan",
            content={
                "query": "Find the mean",
                "selected_requirements": {},
                "steps": [],
                "declarations": [],
                "capabilities": [],
                "capability_expansion": {},
            },
        ),
        _event("model.started", request_id=failed, stage="skill_tool_planning"),
        _event("model.failed", request_id=failed, stage="skill_tool_planning"),
    ]
    snapshot = AgentResult(
        answer="",
        messages=messages,
        completion_status="Not Completed",
        tool_chain_completed=False,
        answer_generation_status="not_started",
        provenance={"execution_cost": {"model_requests": 2}, "budget": {"token_charge": 30, "http_attempts": 2}},
    )
    cursor = prepare_resume(snapshot)
    assert cursor.stage == "skill_tool_planning" and cursor.skip_skill_layer and not cursor.continue_plan

    class Stop(Client):
        async def chat(self, messages, **kwargs):
            self.stages.append(kwargs.get("stage"))
            raise TransportFailure("LLM request deadline exceeded")

    client = Stop()
    with pytest.raises(AgentExecutionError):
        await AgentEngine().run(
            TaskInput(
                question="Find the mean",
                planning_mode="skill_guided",
                resume=snapshot,
            ),
            AgentContext(
                Tools(),
                None,
                Events(),
                {},
                None,
                client,
                planner=SimpleNamespace(),
                options=AlgorithmSettings(
                    planning_mode="skill_guided",
                    allow_rule_planning=False,
                    plan_attempts=1,
                    candidate_pool="all_registered",
                    argument_attempts=1,
                ),
            ),
        )
    assert client.stages == ["skill_tool_planning"]


async def test_protocol_failure_is_not_resumed():
    class ProtocolClient(Client):
        async def chat(self, messages, **kwargs):
            self.stages.append(kwargs.get("stage"))
            raise ModelProtocolError("Function-call format repair budget exhausted")

    client = ProtocolClient()
    with pytest.raises(AgentExecutionError):
        await AgentEngine().run(
            TaskInput(question="Compute the requested statistic.", resume_attempts=3),
            context(Tools(), client, argument_attempts=1),
        )
    assert client.stages == ["tool_planning"]
