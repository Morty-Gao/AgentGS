"""Configuration receipts and observed-value projections must agree with execution."""

import json
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agentgs.agent.base import AgentContext
from agentgs.agent.execution.service import ExecutionService
from agentgs.agent.planning.bridge import _project
from agentgs.agent.planning.instances import (
    ExecutionPlan,
    InputBinding,
    PlanStep,
    bind_bridge_dependencies,
    resolve_inputs,
    reuse_key,
)
from agentgs.agent.planning.model_stages import ModelStages
from agentgs.agent.state import RunState
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import AlgorithmSettings, load_algorithm, resolve_algorithm
from agentgs.contracts.models import TaskInput, ToolSpec
from agentgs.contracts.projections import project_json


def test_identical_computations_with_distinct_identities_have_distinct_reuse_keys():
    first = PlanStep(id="first", tool="RemoteSensingMean", purpose="Mean", arguments={"x": [1, 3]})
    second = first.model_copy(update={"id": "second"})
    assert reuse_key(first, {}, "q") != reuse_key(second, {}, "q")
    assert reuse_key(first, {}, "q") == reuse_key(first.model_copy(deep=True), {}, "q")


@pytest.mark.parametrize(
    "value,path",
    [
        ("measurement", [0]),
        ([1, 2], [-1]),
        ([1, 2], [True]),
        ({"0": 3}, [0]),
        ([3], ["0"]),
        ({"x": "abc"}, ["x", 0]),
    ],
)
def test_json_projections_reject_python_specific_indexing(value, path):
    with pytest.raises(ValueError):
        project_json(value, path)
    with pytest.raises(ValueError):
        _project(json.dumps(value), {"path": path})


@pytest.mark.parametrize("path", [[True], [-1], [1.5]])
def test_binding_schema_rejects_non_json_array_indices(path):
    with pytest.raises(ValidationError):
        InputBinding(step_id="source", path=path)


def test_model_and_bridge_projections_use_identical_paths_and_transforms():
    output = {"samples": [{"value": [1, 2]}]}
    step = PlanStep(
        id="target",
        tool="T",
        purpose="consume",
        depends_on=["source"],
        input_bindings={
            "x": {"step_id": "source", "path": ["samples", 0, "value"], "transform": "list"}
        },
    )
    result = resolve_inputs(step, {"source": output})["x"]
    assert (
        result
        == _project(json.dumps(output), {"path": ["samples", 0, "value"], "transform": "list"})
        == [[1, 2]]
    )
    result[0].append(3)
    assert output == {"samples": [{"value": [1, 2]}]}


def test_bridge_projection_replaces_the_proposed_input_source_in_the_execution_plan():
    original = ExecutionPlan(
        [
            PlanStep(id="source", tool="A", purpose="source"),
            PlanStep(id="middle", tool="M", purpose="middle"),
            PlanStep(
                id="target",
                tool="B",
                purpose="target",
                arguments={"x": 99},
                depends_on=["source"],
                input_bindings={"x": {"step_id": "source", "path": ["value"]}},
            ),
        ]
    )
    state = {
        "capability_indices": {"a": 1, "m": 2, "b": 3},
        "routes": [],
        "step_bindings": {3: {"x": {"source": "m", "path": ["value"], "transform": "list"}}},
    }
    bound = bind_bridge_dependencies(original, state)
    assert bound.steps[2].input_bindings["x"].step_id == "middle"
    assert "middle" in bound.steps[2].depends_on
    assert "x" not in bound.steps[2].arguments
    assert resolve_inputs(bound.steps[2], {"source": {"value": 2}, "middle": {"value": 8}}) == {
        "x": [8]
    }
    assert original.steps[2].arguments == {"x": 99}
    assert original.steps[2].input_bindings["x"].step_id == "source"


def test_preset_order_and_duplicates_do_not_change_effective_experiment_identity():
    options = AlgorithmSettings(planning_mode="skill_guided")
    first = resolve_algorithm(options, TaskInput(ablations=["A6", "A3", "A3"]))
    second = resolve_algorithm(options, TaskInput(ablations=["A3", "A6"]))
    assert first == second
    assert first.module_resolution["ablations"] == ["A3", "A6"]


@pytest.mark.parametrize("change", ["policy", "alias", "requested", "active", "unknown", "presets"])
def test_resolution_metadata_cannot_mask_inconsistent_module_settings(change):
    resolved = resolve_algorithm(
        AlgorithmSettings(planning_mode="skill_guided"), TaskInput(ablations=["A3", "A6"])
    )
    payload = resolved.model_dump()
    if change == "policy":
        payload["modules"]["execution_recovery"] = True
    elif change == "alias":
        payload["execution_recovery"] = True
    elif change == "requested":
        payload["module_resolution"]["requested"]["capability_completion"] = True
    elif change == "active":
        payload["module_resolution"]["active"] = False
    elif change == "presets":
        payload["module_resolution"]["ablations"] = ["A7"]
    else:
        payload["module_resolution"] = {"ignored": True}
    with pytest.raises(ValidationError, match="resolution|ablation"):
        AlgorithmSettings.model_validate(deepcopy(payload))


def test_mutating_resolved_options_does_not_silently_reapply_stale_metadata():
    options = resolve_algorithm(
        AlgorithmSettings(planning_mode="skill_guided"), TaskInput(ablations=["A6"])
    )
    options.modules.execution_recovery = True
    with pytest.raises(ValidationError, match="resolution"):
        resolve_algorithm(options, TaskInput())


def test_project_settings_cannot_supply_runtime_resolution_records(tmp_path):
    (tmp_path / "config.toml").write_text("[agent.module_resolution]\nactive = true\n")
    with pytest.raises(ValueError, match="runtime-generated"):
        load_algorithm(tmp_path)


@pytest.mark.parametrize("extra", [True, None])
async def test_open_schema_fields_survive_protocol_validation_and_dispatch(extra):
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    if extra is not None:
        schema["additionalProperties"] = extra
    spec = ToolSpec(id="OpenTool", description="Open arguments", input_schema=schema)

    class Client:
        async def chat(self, messages, **kwargs):
            return ChatReply(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": uuid4().hex,
                            "type": "function",
                            "function": {"name": "OpenTool", "arguments": '{"x":1,"custom":2}'},
                        }
                    ],
                }
            )

    context = AgentContext(
        SimpleNamespace(specs=lambda: [spec]),
        None,
        SimpleNamespace(emit=lambda *args: None),
        client=Client(),
        options=AlgorithmSettings(),
    )
    state = RunState(context, TaskInput(question="Compute an observed value"))
    stages = ModelStages(state, lambda _: None)
    message = await stages.generate_tool_call(
        "OpenTool", state.task.question, {}, {}, "JSON", "JSON", None, []
    )
    arguments, internal = ExecutionService(state, stages).split_arguments(
        "OpenTool", message.tool_calls[0].function.arguments
    )
    assert arguments == {"x": 1, "custom": 2} and not internal


async def test_protocol_repair_retains_disabled_execution_recovery_instruction():
    spec = ToolSpec(id="T", description="Tool", input_schema={"type": "object"})

    class Client:
        def __init__(self):
            self.requests = []

        async def chat(self, messages, **kwargs):
            self.requests.append(deepcopy(messages))
            return ChatReply(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": uuid4().hex,
                            "type": "function",
                            "function": {
                                "name": "Wrong" if len(self.requests) == 1 else "T",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            )

    client = Client()
    context = AgentContext(
        SimpleNamespace(specs=lambda: [spec]),
        None,
        SimpleNamespace(emit=lambda *args: None),
        client=client,
        options=AlgorithmSettings(planning_mode="skill_guided", execution_recovery=False),
    )
    state = RunState(context, TaskInput(question="Compute a value"))
    stages = ModelStages(state, lambda _: None)
    await stages.generate_tool_call("T", state.task.question, {}, {}, "JSON", "JSON", None, [])
    instruction = client.requests[-1][-1]["content"].split("<system-reminder>")[-1]
    assert "Execution recovery is disabled" in instruction
    assert "No scientific tool executed" in instruction
