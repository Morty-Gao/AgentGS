import json
from html import unescape

import pytest

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.state import AgentExecutionError
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.models import TaskInput, ToolError, ToolResult, ToolSpec


def function_call(name, arguments, call_id=None):
    if name == "submit_tool_plan" and isinstance(arguments, dict) and set(arguments) == {"tools"}:
        arguments = {
            "steps": [
                {"id": f"s{i}", "tool": tool, "purpose": f"Compute {tool}"}
                for i, tool in enumerate(arguments["tools"], 1)
            ]
        }
    from uuid import uuid4

    return {
        "role": "assistant",
        "content": None,
        "reasoning_content": "fixture reasoning",
        "tool_calls": [
            {
                "id": call_id or uuid4().hex,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def plan_call(names):
    return function_call("submit_tool_plan", {"tools": names})


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
        return ChatReply(
            message=value if isinstance(value, dict) else {"role": "assistant", "content": value}
        )


class StageClient:
    async def chat(self, messages, *, stage, **kwargs):
        if stage in {"skill_tool_planning", "tool_planning"}:
            value = plan_call(self.plan)
        elif stage == "tool_argument_generation":
            name = kwargs["tool_choice"]["function"]["name"]
            value = function_call(
                name, {"x": [1, 2, 3]} if name == "RemoteSensingMean" else {"a": 3, "b": 1}
            )
        else:
            value = {"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:2"}
        return ChatReply(message=value)


def context(tools, client, strict=True):
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
            allow_rule_planning=not strict,
            plan_attempts=1 if strict else 3,
        ),
    )


@pytest.mark.parametrize("explicit_contract", [True, False])
async def test_fixed_format_reaches_final_request_through_engine(explicit_contract):
    from pathlib import Path

    from agentgs.runtime.answer_contracts import AnswerContracts

    contracts = AnswerContracts(Path(__file__).resolve().parents[3])
    contract = next(c for c in contracts.contracts.values() if c["question_id"] == 199)
    tools = Tools()
    client = Client(
        [
            plan_call(["RemoteSensingMean"]),
            function_call("RemoteSensingMean", {"x": [1, 2, 3]}),
            "temperature_anomaly: 2 °C",
        ]
    )
    ctx = context(tools, client)
    result = await AgentEngine().run(
        TaskInput(
            question="Compute the requested statistic.",
            output_contract=contract,
            dataset_instructions=contracts.system_prompt,
        ),
        ctx,
    )
    final_request = client.requests[-1]
    assert contract["answer_format"]["template"] in unescape(final_request[-1]["content"])
    assert contract["answer_format"]["notes"] in unescape(final_request[-1]["content"])
    assert final_request[-1]["role"] == "tool"
    assert contracts.system_prompt in unescape(final_request[0]["content"])
    assert result.answer == "temperature_anomaly: 2 °C"
    validation = next(
        t for t in result.execution_events() if t["type"] == "final_answer_format_validation"
    )
    assert validation["valid"]


async def test_staged_plan_argument_answer_preserves_model_text():
    tools = Tools()
    client = Client(
        [
            plan_call(["RemoteSensingMean"]),
            function_call("RemoteSensingMean", {"x": [1, 2, 3]}),
            "IsCompleted:Completed\nFinalAnswer:2",
        ]
    )
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic."), context(tools, client)
    )
    assert result.answer == "2" and result.completion_status == "Completed"
    assert [row["stage"] for row in result.usage] == [
        "tool_planning",
        "tool_argument_generation",
        "final_answer_generation",
    ]
    assert len(tools.calls) == 1 and tools.calls[0].arguments == {"x": [1, 2, 3]}
    assert all(row["usage"] is None for row in result.usage)


async def test_final_answer_failure_never_replays_successful_tools():
    tools = Tools()
    client = Client(
        [
            plan_call(["RemoteSensingMean"]),
            function_call("RemoteSensingMean", {"x": [1, 2, 3]}),
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
    assert caught.value.result.usage[-1]["usage"] is None
    assert caught.value.result.usage[-1]["status"] == "failed"


async def test_final_authentication_error_preserves_status_for_benchmark_abort():
    from agentgs.client.errors import ProviderError

    tools = Tools()
    client = Client(
        [
            plan_call(["RemoteSensingMean"]),
            function_call("RemoteSensingMean", {"x": [1, 2, 3]}),
            ProviderError(401),
        ]
    )
    with pytest.raises(AgentExecutionError) as caught:
        await AgentEngine().run(
            TaskInput(question="Compute the requested statistic."), context(tools, client)
        )
    assert caught.value.status_code == 401 and "HTTP 401" in str(caught.value)
    assert len(tools.calls) == 1


async def test_exhausted_repairs_still_generate_an_explicit_incomplete_answer():
    tools = Tools(True)
    client = Client(
        [
            plan_call(["RemoteSensingMean"]),
            function_call("RemoteSensingMean", {"x": [1, 2, 3]}),
            function_call("RemoteSensingMean", {"x": [2, 3, 4]}),
            function_call("RemoteSensingMean", {"x": [3, 4, 5]}),
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
    assert len([event for event in result.execution_events() if event["type"] == "tool_retry"]) == 3


async def test_unsupported_numeric_answer_is_preserved_and_marked_incomplete():
    tools = Tools()
    client = Client(
        [
            plan_call(["RemoteSensingMean"]),
            function_call("RemoteSensingMean", {"x": [1, 2, 3]}),
            "IsCompleted:Completed\nFinalAnswer:999",
        ]
    )
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic."), context(tools, client)
    )
    assert result.answer == "999" and result.completion_status == "Not Completed"
    assert result.tool_chain_completed


async def test_replanning_does_not_reuse_unspecified_inputs_by_tool_name():
    class RepairTools(Tools):
        async def execute(self, call):
            self.fail = (
                call.tool == "RemoteSensingDifference"
                and sum(row.tool == call.tool for row in self.calls) < 3
            )
            return await super().execute(call)

    tools = RepairTools()
    plan = ["RemoteSensingMean", "RemoteSensingDifference"]
    client = StageClient()
    client.plan = plan
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic."), context(tools, client, False)
    )
    assert [row.tool for row in tools.calls] == ["RemoteSensingMean"] + [
        "RemoteSensingDifference"
    ] * 3 + ["RemoteSensingMean", "RemoteSensingDifference"]
    assert result.tool_chain_completed and result.answer == "2"
    resumed = [row for row in result.execution_events() if row["type"] == "plan_resume"]
    assert not resumed


async def test_strict_model_plan_preserves_duplicate_tool_instances():
    tools = Tools()
    client = StageClient()
    client.plan = ["RemoteSensingMean", "RemoteSensingMean"]
    result = await AgentEngine().run(
        TaskInput(question="Compute independent statistics."), context(tools, client)
    )
    assert len(tools.calls) == 2 and result.tool_chain_completed
    assert tools.calls[0].id != tools.calls[1].id


@pytest.mark.parametrize(
    "answer,completion",
    [
        ("**temperature_anomaly: 2 °C**", "Completed"),
        ("temperature_anomaly: unknown °C", "Not Completed"),
        ("temperature_anomaly: UNKNOWN °C", "Not Completed"),
        ("temperature_anomaly: <value> °C", "Not Completed"),
    ],
)
async def test_answer_format_is_diagnostic_but_unknown_is_incomplete(answer, completion):
    from pathlib import Path

    from agentgs.runtime.answer_contracts import AnswerContracts

    contracts = AnswerContracts(Path(__file__).resolve().parents[3])
    contract = next(c for c in contracts.contracts.values() if c["question_id"] == 199)
    client = Client(
        [
            plan_call(["RemoteSensingMean"]),
            function_call("RemoteSensingMean", {"x": [1, 2, 3]}),
            answer,
        ]
    )
    ctx = context(Tools(), client)
    result = await AgentEngine().run(
        TaskInput(question="Compute the requested statistic.", output_contract=contract),
        ctx,
    )
    assert result.answer == answer
    assert result.completion_status == completion
    event = next(
        e for e in result.execution_events() if e["type"] == "final_answer_format_validation"
    )
    assert event["diagnostic_only"]
    if "**" in answer:
        assert not event["valid"]
