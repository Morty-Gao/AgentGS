"""Independent cases from the original bridge argument acceptance tests."""

import json

import pytest

from agentgs.agent.base import AgentContext
from agentgs.agent.execution.service import ExecutionService
from agentgs.agent.state import RunState
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.models import TaskInput
from agentgs.datasets.registry import DatasetRegistry
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.tools.executor import LocalToolExecutor


def bridge_context(value):
    return {
        "shared_bridge": {
            "execution_index": 1,
            "state": {
                "step_bindings": {1: {"a": {"source": "prior_measurement", "path": ["value"]}}},
                "capability_indices": {"prior_measurement": 0},
                "execution_records": {
                    0: {"succeeded": True, "output": json.dumps({"value": value})}
                },
            },
        }
    }


async def execute(project, name, upstream, generated, context, question):
    class Events:
        def emit(self, *args):
            pass

    class Stages:
        calls = []

        async def generate_tool_call(self, name, question, registry, context, input_contract, output_contract, feedback, history):
            self.calls.append(feedback)
            value = generated[len(self.calls) - 1]
            if isinstance(value, Exception):
                raise value
            from agentgs.messages import Message
            from uuid import uuid4
            message = Message.from_openai({"role": "assistant", "reasoning_content": "preserve reference operand",
                "tool_calls": [{"id": uuid4().hex, "type": "function", "function": {"name": name, "arguments": value}}]})
            state.conversation.append(message)
            return message

    events = Events()
    artifacts = LocalArtifactStore(project / "artifacts")
    tools = LocalToolExecutor(DatasetRegistry(project), artifacts, events)
    state = RunState(
        AgentContext(tools, artifacts, events, {}, None, options=AlgorithmSettings()),
        TaskInput(question=question),
    )
    stages = Stages()
    result = await ExecutionService(state, stages).run(
        name,
        question,
        None,
        json.dumps({"value": upstream}) + "\n" + json.dumps({"value": upstream}),
        json.dumps({"value": upstream}),
        context,
    )
    return result, state, stages.calls


@pytest.mark.parametrize("measured,reference", [(5.0, 2.0), (9.5, -1.25)])
async def test_only_declared_measurement_is_projected_and_reference_survives(
    project, measured, reference
):
    result, state, calls = await execute(
        project,
        "RemoteSensingDifference",
        measured,
        [json.dumps({"a": 999, "b": reference})],
        bridge_context(measured),
        f"Subtract the fixed reference value {reference} from the upstream spatial mean.",
    )
    assert result.succeeded and len(calls) == 1
    assert json.loads(result.final_input)["a"] == measured
    assert json.loads(result.final_input)["b"] == reference
    assert state.results[0].data["value"] == measured - reference


async def test_divisor_survives_feedback_repair(project):
    result, state, calls = await execute(
        project,
        "RemoteSensingDivision",
        6.0,
        ['{"a":999,"b":0}', '{"a":123,"b":4}'],
        bridge_context(6.0),
        "Divide the upstream difference by a fixed divisor 4.",
    )
    assert result.succeeded and len(calls) == 2 and calls[1] is not None
    assert state.results[0].succeeded is False
    assert json.loads(result.final_input)["a"] == 6.0
    assert json.loads(result.final_input)["b"] == 4
    assert state.results[-1].data["value"] == 1.5


async def test_bridge_model_failure_cannot_invent_upstream_reference(project):
    with pytest.raises(RuntimeError, match="fixture generation unavailable"):
        await execute(
            project,
            "RemoteSensingDifference",
            5,
            [RuntimeError("fixture generation unavailable")],
            bridge_context(5),
            "Subtract a requested reference.",
        )


async def test_non_bridge_keeps_original_deterministic_upstream_pair(project):
    result, state, calls = await execute(
        project, "RemoteSensingDifference", 5, ['{"a":5,"b":2}'], None, "Compute a difference."
    )
    assert result.succeeded and len(calls) == 1
    assert json.loads(result.final_input)["a"] == json.loads(result.final_input)["b"] == 5
    assert state.results[-1].data["value"] == 0
