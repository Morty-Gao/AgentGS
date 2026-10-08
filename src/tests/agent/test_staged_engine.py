import json

import pytest

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.state import AgentExecutionError
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.models import TaskInput, ToolError, ToolResult, ToolSpec


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
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def specs(self):
        return [
            ToolSpec(id=name, description="Scalar calculation", input_schema={"type": "object"})
            for name in ["RemoteSensingMean", "RemoteSensingDifference"]
        ]

    async def execute(self, call):
        self.calls.append(call)
        if self.fail:
            return ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="domain", message="bad value"),
                raw_output='{"status":"error","error":"bad value"}',
            )
        values = call.arguments.get("x", [1, 2, 3])
        data = {"value": sum(values) / len(values)}
        return ToolResult(
            call_id=call.id, tool=call.tool, succeeded=True, data=data, raw_output=json.dumps(data)
        )


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def chat(self, messages, **kwargs):
        self.requests.append(messages)
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return ChatReply(message={"role": "assistant", "content": value})


def context(tools, client, strict=True):
    return AgentContext(
        tools,
        None,
        Events(),
        {},
        None,
        client,
        tool_retriever=KG(),
        options=AlgorithmSettings(planning_mode="tool_only", require_llm=strict),
    )


async def test_staged_plan_argument_answer_preserves_model_text():
    tools = Tools()
    client = Client(
        [
            "Plan Chain:\n['RemoteSensingMean']",
            'Input:{"x":[1,2,3]}',
            "IsCompleted:Completed\nFinalAnswer:2",
        ]
    )
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic."), context(tools, client)
    )
    assert result.answer == "2" and result.completion_status == "Completed"
    assert [row["stage"] for row in result.usage] == [
        "skill_tool_planning",
        "tool_argument_generation",
        "final_answer_generation",
    ]
    assert len(tools.calls) == 1 and tools.calls[0].arguments == {"x": [1, 2, 3]}
    assert all(row["usage"] is None for row in result.usage)


async def test_final_answer_failure_never_replays_successful_tools():
    tools = Tools()
    client = Client(
        [
            "Plan Chain:\n['RemoteSensingMean']",
            'Input:{"x":[1,2,3]}',
            RuntimeError("provider unavailable"),
        ]
    )
    with pytest.raises(AgentExecutionError) as caught:
        await AgentEngine().run(
            TaskInput(question="Compute the requested statistic."), context(tools, client, False)
        )
    assert len(tools.calls) == 1
    assert caught.value.result.tool_chain_completed is True
    assert caught.value.result.answer_generation_status == "failed"
    assert caught.value.result.tool_results[0].succeeded


async def test_exhausted_repairs_still_generate_an_explicit_incomplete_answer():
    tools = Tools(True)
    client = Client(
        [
            "Plan Chain:\n['RemoteSensingMean']",
            'Input:{"x":[1,2,3]}',
            'Input:{"x":[2,3,4]}',
            'Input:{"x":[3,4,5]}',
            "IsCompleted:Not Completed\nFinalAnswer:The result is uncertain.",
        ]
    )
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic."), context(tools, client)
    )
    assert len(tools.calls) == 3
    assert result.completion_status == "Not Completed"
    assert result.answer == "The result is uncertain."
    assert result.answer_generation_status == "success" and not result.tool_chain_completed
    assert len([event for event in result.trace if event["type"] == "tool_retry"]) == 3


async def test_unsupported_numeric_answer_is_preserved_and_marked_incomplete():
    tools = Tools()
    client = Client(
        [
            "Plan Chain:\n['RemoteSensingMean']",
            'Input:{"x":[1,2,3]}',
            "IsCompleted:Completed\nFinalAnswer:999",
        ]
    )
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic."), context(tools, client)
    )
    assert result.answer == "999" and result.completion_status == "Not Completed"
    assert result.tool_chain_completed
