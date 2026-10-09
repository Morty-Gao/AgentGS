"""Native wire protocol, binding, reasoning continuation and failure recovery."""

import json
from html import unescape
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.execution.service import ExecutionService
from agentgs.agent.planning.model_stages import ModelStages
from agentgs.agent.state import AgentExecutionError, RunState
from agentgs.client.models import LLMConfig
from agentgs.client.openai_compatible import OpenAICompatibleClient
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.catalog import load_tool_scope
from agentgs.contracts.models import AgentResult, TaskInput
from agentgs.datasets.registry import DatasetRegistry
from agentgs.messages import Conversation, Message
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.tools.executor import LocalToolExecutor
from agentgs.runtime.tools.registry import builtin_tools


class KG:
    def __init__(self, names):
        self.names = names

    async def retrieve(self, query):
        return [[name, "has the functionality that", "Scalar calculation"] for name in self.names]

    def get_tool_subgraph(self, name, **kwargs):
        return [
            [name, "has the functionality that", "Scalar calculation"],
            [name, "inputs", "JSON arguments"],
            [name, "outputs", "JSON result"],
        ]


class Events:
    def emit(self, kind, payload):
        pass


def native(name, arguments, call_id=None, **reasoning):
    if name == "submit_tool_plan" and isinstance(arguments, dict) and set(arguments) == {"tools"}:
        arguments = {
            "steps": [
                {"id": f"s{i}", "tool": tool, "purpose": f"Compute {tool}"}
                for i, tool in enumerate(arguments["tools"], 1)
            ]
        }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id or uuid4().hex,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                },
            }
        ],
        **reasoning,
    }


def setup(project, replies, names=("RemoteSensingMean",), *, skill_guided=False):
    payloads, checkpoints = [], []
    replies = iter(replies)

    def respond(request):
        payloads.append(json.loads(request.content))
        # Every request is a legal continuation: all original calls are answered.
        conversation = Conversation.from_openai(payloads[-1]["messages"])
        assert not conversation.pending_tool_calls()
        return httpx.Response(200, json={"choices": [{"message": next(replies)}]})

    config = LLMConfig.model_validate(
        {
            "providers": {"p": {"base_url": "https://fixture/v1", "api_key": "fixture"}},
            "models": {
                "m": {
                    "provider": "p",
                    "model": "fixture",
                    "context_window": 128000,
                    "max_output_tokens": 8192,
                }
            },
        }
    )
    client = OpenAICompatibleClient(config, httpx.MockTransport(respond))
    artifacts = LocalArtifactStore(project / "artifacts")
    tools = LocalToolExecutor(DatasetRegistry(project), artifacts, Events())
    context = AgentContext(
        tools,
        artifacts,
        Events(),
        {},
        None,
        client,
        tool_retriever=KG(names),
        options=AlgorithmSettings(
            planning_mode="skill_guided" if skill_guided else "tool_only",
            allow_rule_planning=False,
            plan_attempts=1,
            retry_base_seconds=0,
            retry_jitter_seconds=0,
        ),
        checkpoint=lambda result: checkpoints.append(result.model_dump_json()),
    )
    return context, client, payloads, checkpoints


async def test_stages_use_native_calls_and_preserve_every_current_turn_reasoning_field(project):
    plan = native(
        "submit_tool_plan",
        {"tools": ["RemoteSensingMean", "RemoteSensingKelvinToCelsius"]},
        "plan",
        reasoning_content="Choose independent cohorts.",
    )
    first = native(
        "RemoteSensingMean",
        {"x": [1, 3]},
        "first",
        reasoning_details=[{"type": "encrypted", "data": "opaque", "signature": "exact"}],
    )
    second = native(
        "RemoteSensingKelvinToCelsius", {"kelvin": 273.15}, "second", thinking="Second cohort."
    )
    final = {
        "role": "assistant",
        "content": "IsCompleted:Completed\nFinalAnswer:2; 0",
        "reasoning_content": "Compare observed means.",
    }
    context, client, payloads, checkpoints = setup(
        project,
        [plan, first, second, final],
        names=("RemoteSensingMean", "RemoteSensingKelvinToCelsius"),
    )
    try:
        result = await AgentEngine().run(
            TaskInput(question="Compute two independent statistics."), context
        )
    finally:
        await client.aclose()
    assert [row.data["value"] for row in result.tool_results] == [2, 0]
    assert result.tool_chain_completed
    for request in payloads:
        assert [m["role"] for m in request["messages"]].count("user") == 1
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][-1]["role"] in {"user", "tool"}
    assert [p["tool_choice"]["function"]["name"] for p in payloads[:3]] == [
        "submit_tool_plan",
        "RemoteSensingMean",
        "RemoteSensingKelvinToCelsius",
    ]
    assert all(p["parallel_tool_calls"] is False for p in payloads[:3])
    assert "tools" not in payloads[-1]
    assert [m for m in payloads[-1]["messages"] if m.get("tool_calls")] == [plan, first, second]
    restored = Conversation(messages=AgentResult.model_validate_json(checkpoints[-1]).messages)
    assert not restored.pending_tool_calls()
    requests = [e for e in restored.execution_events() if e["type"] == "model.started"]
    assert [restored.request_messages(e["request_id"]) for e in requests] == [
        p["messages"] for p in payloads
    ]
    assert [e["tool_name"] for e in restored.execution_events() if e["type"] == "tool_call"] == [
        "RemoteSensingMean",
        "RemoteSensingKelvinToCelsius",
    ]
    assert any(e["type"] == "planning_call" for e in restored.execution_events())


@pytest.mark.parametrize(
    "bad", ["{broken", 'Input:{"x":[1,3]}', "[1,3]", {"x": ["wrong"]}, {"x": [1, 3], "unknown": 5}]
)
async def test_invalid_arguments_are_answered_and_repaired_without_text_fallback(project, bad):
    rejected = native("RemoteSensingMean", bad, "bad", reasoning_content="Original attempt.")
    repaired = native(
        "RemoteSensingMean",
        {"x": [1, 3]},
        "repaired",
        reasoning_content="Repair the reported error.",
    )
    context, client, payloads, _ = setup(project, [rejected, repaired])
    state = RunState(context, TaskInput(question="Compute the requested statistic."))
    stages = ModelStages(state, lambda q: None)
    try:
        result = await ExecutionService(state, stages).run(
            "RemoteSensingMean", state.task.question, None, ""
        )
    finally:
        await client.aclose()
    assert result.succeeded
    assert len(result.attempts) == (2 if bad == {"x": ["wrong"]} else 1)
    assert next(m for m in payloads[1]["messages"] if m.get("tool_calls")) == rejected
    feedback = next(m for m in payloads[1]["messages"] if m.get("tool_call_id") == "bad")
    assert (
        json.loads(
            unescape(
                feedback["content"].split("<tool-result>\n", 1)[1].split("\n</tool-result>", 1)[0]
            )
        )["status"]
        == "error"
    )
    assert not state.conversation.pending_tool_calls()
    assert len([r for r in state.results if r.succeeded]) == 1
    assert {m.tool_call_id for m in state.conversation.messages if m.role == "tool"} == {
        "bad",
        "repaired",
    }


@pytest.mark.parametrize("unexpected", ["wrong_name", "multiple", "text_only"])
async def test_provider_protocol_errors_do_not_execute_unplanned_tools(project, unexpected):
    bad = native("RemoteSensingDifference", {"a": 3, "b": 1}, "bad")
    if unexpected == "multiple":
        bad = native("RemoteSensingMean", {"x": [999]}, "bad")
        bad["tool_calls"] += native("RemoteSensingMean", {"x": [999]}, "extra")["tool_calls"]
    elif unexpected == "text_only":
        bad = {
            "role": "assistant",
            "content": 'Input:{"x":[999]}',
            "reasoning_content": "Text is not a call.",
        }
    good = native("RemoteSensingMean", {"x": [1, 3]}, "good")
    context, client, payloads, _ = setup(project, [bad, good])
    state = RunState(context, TaskInput(question="Compute a statistic."))
    try:
        result = await ExecutionService(state, ModelStages(state, lambda q: None)).run(
            "RemoteSensingMean", state.task.question, None, ""
        )
    finally:
        await client.aclose()
    assert result.succeeded
    assert len(state.results) == 1 and state.results[0].call_id == "good"
    assert not state.conversation.pending_tool_calls()
    # No synthetic model calls are introduced to compensate for prose replies.
    assert [
        m.to_openai() for m in state.conversation.messages if m.role == "assistant" and m.tool_calls
    ] == ([bad, good] if unexpected != "text_only" else [good])


async def test_unknown_planned_argument_is_rejected_before_execution(project):
    step = {
        "id": "mean",
        "tool": "RemoteSensingMean",
        "purpose": "Mean",
        "arguments": {"typo": [1, 3]},
    }
    bad = native("submit_tool_plan", {"steps": [step]})
    good = native("submit_tool_plan", {"steps": [{**step, "arguments": {"x": [1, 3]}}]})
    context, client, payloads, _ = setup(
        project,
        [
            bad,
            good,
            native("RemoteSensingMean", {"x": [1, 3]}),
            {"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:2"},
        ],
    )
    try:
        result = await AgentEngine().run(TaskInput(question="Compute the mean."), context)
    finally:
        await client.aclose()
    assert result.tool_chain_completed and len(result.tool_results) == 1
    assert "Unknown planned input fields" in json.dumps(payloads[1])


async def test_invalid_native_plan_is_repaired_and_not_scored_as_domain_execution(project):
    bad = native(
        "submit_tool_plan",
        {"tools": ["ExecutePython"]},
        "badplan",
        reasoning_content="Invalid candidate.",
    )
    good = native(
        "submit_tool_plan",
        {"tools": ["RemoteSensingMean"]},
        "goodplan",
        reasoning_content="Use registered candidate.",
    )
    context, client, payloads, _ = setup(project, [bad, good])
    state = RunState(context, TaskInput(question="Compute a statistic."))
    try:
        plan = await ModelStages(state, lambda q: None).generate_tools_plan(
            state.task.question, await context.tool_retriever.retrieve(None)
        )
    finally:
        await client.aclose()
    assert plan == ["RemoteSensingMean"]
    assert next(m for m in payloads[1]["messages"] if m.get("tool_calls")) == bad
    assert not any(e["type"] in {"tool_call", "tool_retry"} for e in state.execution_events())
    assert not state.results and not state.conversation.pending_tool_calls()


async def test_final_generation_failure_preserves_native_calls_and_reasoning(project):
    context, client, _, snapshots = setup(
        project,
        [
            native(
                "submit_tool_plan",
                {"tools": ["RemoteSensingMean"]},
                "plan",
                reasoning_content="Plan.",
            ),
            native("RemoteSensingMean", {"x": [1, 3]}, "call", reasoning_content="Calculate."),
            {
                "role": "assistant",
                "content": None,
                "reasoning_details": [{"type": "encrypted", "data": "retained"}],
            },
        ],
    )
    try:
        with pytest.raises(AgentExecutionError) as caught:
            await AgentEngine().run(TaskInput(question="Compute the statistic."), context)
    finally:
        await client.aclose()
    assert caught.value.result.tool_chain_completed
    assert len(caught.value.result.tool_results) == 1
    assert caught.value.result.answer_generation_status == "failed"
    result = AgentResult.model_validate_json(snapshots[-1])
    assert [
        m.reasoning for m in result.messages if m.role == "assistant" and m.reasoning is not None
    ] == [
        "Plan.",
        "Calculate.",
    ]
    responses = [e for e in result.execution_events() if e["type"] == "model.response"]
    assert responses[-1]["response"]["reasoning"] == [{"type": "encrypted", "data": "retained"}]


def test_complete_catalog_has_typed_native_inputs_and_no_deprecated_entrypoints():
    registry = builtin_tools()
    assert set(registry) == set(load_tool_scope()["tool_ids"])
    for name, item in registry.items():
        schema = item.spec.input_schema
        assert schema["type"] == "object", name
        assert schema["additionalProperties"] is False, name
        assert schema["properties"], name
        assert "input_text" not in schema["properties"], name
        assert "parameters" not in schema["properties"], name
    with pytest.raises(ValidationError):
        TaskInput(question="q", algorithm="prototype")
    with pytest.raises(ValidationError):
        Message.from_openai(
            {"role": "assistant", "function_call": {"name": "T", "arguments": "{}"}}
        )


@pytest.mark.parametrize(
    "name,args",
    [
        ("RemoteSensingBatchNDSI", {"green_file_list": ["a", "b"], "swir_file_list": ["c"]}),
        (
            "RemoteSensingNDVI",
            {
                "nir_band_file": "a",
                "red_band_file": "b",
                "nir_band_files": ["c"],
                "red_band_files": ["d"],
            },
        ),
        ("RemoteSensingDivision", {"a_values": [1, 2], "b_values": [3]}),
        ("CCSRasterAlgebra", {"raster_path": "a.tif", "operation": "multiply"}),
        ("RemoteSensingMean", {"x": [float("nan")]}),
    ],
)
def test_input_contracts_reject_invalid_scientific_shapes(name, args):
    with pytest.raises(ValidationError):
        builtin_tools()[name].input_type.model_validate(args)
