"""Behavioral checks for module combinations and their actual runtime effects."""

import json
from copy import deepcopy
from itertools import product
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.planning.capabilities import CapabilityGraph
from agentgs.agent.planning.instances import ExecutionPlan, PlanStep
from agentgs.agent.planning.module_context import module_guidance, planning_context
from agentgs.agent.planning.module_expansion import expand_requirements
from agentgs.agent.planning.module_selection import select_first_valid
from agentgs.agent.planning.skills import SkillPlanner
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import AlgorithmSettings, resolve_algorithm
from agentgs.contracts.models import TaskInput, ToolError, ToolResult, ToolSpec
from agentgs.contracts.models.modules import ModuleSettings
from agentgs.skills.definitions import load_skill_definitions


def declaration(sid="statistics", steps=None):
    value = deepcopy(load_skill_definitions()[0])
    value.update(id=sid, name=sid, description="Compute statistics", keywords=["statistics"])
    value["tool_candidates"] = ["RemoteSensingMean"]
    value["workflow"] = {
        "steps": steps
        or [
            {
                "id": "mean",
                "capability": "mean",
                "operation": "Compute the mean",
                "depends_on": [],
                "tool_selection": {"candidates": ["RemoteSensingMean"]},
            }
        ]
    }
    value["planning"] = {"compatible_skills": []}
    return value


@pytest.mark.parametrize(
    "presets", [[], ["A1"], ["A2", "A4"], ["A3", "A6"], ["A1", "A5", "A6"], ["BIPARTITE", "A2"]]
)
def test_policy_roundtrip_and_independent_budgets(presets):
    options = AlgorithmSettings(planning_mode="skill_guided", plan_attempts=4, tool_attempts=5)
    task = TaskInput(ablations=presets)
    resolved = resolve_algorithm(options, task)
    assert (
        resolve_algorithm(AlgorithmSettings.model_validate_json(resolved.model_dump_json()), task)
        == resolved
    )
    assert resolved.plan_attempts == 4 and resolved.tool_attempts == 5
    if "A3" in presets:
        assert not resolved.shared_bridge and not resolved.modules.directed_bridging
        assert resolved.module_resolution["suppressed"] == {
            "directed_bridging": "capability_completion_disabled"
        }
    if "A6" in presets:
        assert not resolved.execution_recovery
    assert options.modules == ModuleSettings()


@pytest.mark.parametrize("mode", ["tool_only", "data_only"])
def test_module_overrides_are_not_silently_ignored_in_baselines(mode):
    with pytest.raises(ValueError, match="require planning_mode"):
        resolve_algorithm(AlgorithmSettings(planning_mode=mode), TaskInput(ablations=["A4"]))


def test_aliases_and_conflicts():
    options = AlgorithmSettings(planning_mode="skill_guided", shared_bridge=False)
    assert not resolve_algorithm(options, TaskInput()).modules.directed_bridging
    assert resolve_algorithm(options, TaskInput(modules={"directed_bridging": True})).shared_bridge
    for task in [
        TaskInput(ablations=["A1", "BIPARTITE"]),
        TaskInput(ablations=["A6"], modules={"execution_recovery": True}),
        TaskInput(shared_bridge=False, modules={"directed_bridging": True}),
    ]:
        with pytest.raises(ValueError, match="[Cc]onflict|contradicts"):
            resolve_algorithm(options, task)


def test_unresolved_configuration_roundtrip():
    options = AlgorithmSettings(
        planning_mode="skill_guided", modules={"capability_completion": False}
    )
    restored = AlgorithmSettings.model_validate_json(options.model_dump_json())
    assert resolve_algorithm(restored, TaskInput()) == resolve_algorithm(options, TaskInput())


@pytest.mark.parametrize("bits", list(product([False, True], repeat=6)))
def test_every_a1_a6_combination_resolves(bits):
    presets = [f"A{i}" for i, enabled in enumerate(bits, 1) if enabled]
    resolved = resolve_algorithm(
        AlgorithmSettings(planning_mode="skill_guided"), TaskInput(ablations=presets)
    )
    assert resolved.modules.planning_structure == ("flat_tools" if bits[0] else "hierarchical")
    assert resolved.modules.predecessor_expansion is not bits[1]
    assert resolved.modules.capability_completion is not bits[2]
    assert resolved.modules.directed_bridging == (not bits[2] and not bits[3])
    assert resolved.modules.tool_selection == ("first_valid" if bits[4] else "optimized")
    assert resolved.modules.execution_recovery is not bits[5]


@pytest.mark.parametrize(
    "expand,complete", [(True, True), (False, True), (True, False), (False, False)]
)
def test_predecessor_insertion_and_completion_compensation(expand, complete):
    graph = CapabilityGraph(
        [
            declaration(
                steps=[
                    {
                        "id": "load",
                        "capability": "load",
                        "operation": "Load data",
                        "depends_on": [],
                    },
                    {
                        "id": "mean",
                        "capability": "mean",
                        "operation": "Mean",
                        "depends_on": ["load"],
                    },
                ]
            )
        ]
    )
    result = expand_requirements(
        graph, ["statistics"], "mean", predecessor_expansion=expand, capability_completion=complete
    )
    assert result["initial_instances"] == ["statistics::mean"]
    assert result["predecessor_added"] == (["statistics::load"] if expand else [])
    if expand or complete:
        assert result["capability_ids"] == ["load", "mean"]
        assert not result["unresolved_capabilities"]
    else:
        assert result["capability_ids"] == ["mean"]
        assert result["unresolved_capabilities"] == ["statistics::load"]
    assert result["compensated_predecessors"] == (
        ["statistics::load"] if complete and not expand else []
    )


def test_information_controls_retain_declarations_without_flat_order():
    skill = declaration()
    skill["workflow"]["execution_policy"] = "sequential"
    plan = SkillPlanner([skill]).plan("statistics")
    full = planning_context(plan, "hierarchical")
    flat = planning_context(plan, "flat_tools")
    bipartite = planning_context(plan, "bipartite")
    assert (
        full["declaration_sha256"] == flat["declaration_sha256"] == bipartite["declaration_sha256"]
    )
    assert flat["strategy_declarations"] == full["strategy_declarations"]
    assert "requirements" not in flat and "hypergraph" not in flat
    restored = []
    for node in bipartite["strategy_nodes"]:
        value = deepcopy(node)
        value["workflow"] = value.pop("workflow_attributes")
        value["workflow"]["steps"] = [
            edge["operation"]
            for edge in bipartite["incidence_edges"]
            if edge["strategy_id"] == value["id"]
        ]
        restored.append(value)
    assert restored == full["strategy_declarations"]
    policy = module_guidance(
        ModuleSettings(planning_structure="flat_tools", execution_recovery=False)
    )
    assert "Plan directly over tool execution instances" in policy
    assert "no argument repair or replanning" in policy


def test_disabled_completion_keeps_semantic_support_diagnostics():
    primary = declaration()
    support = declaration("calibration", [{"id": "calibrate", "capability": "calibrate", "operation": "calibration", "depends_on": []}])
    support["description"] = "calibration"
    primary["planning"]["compatible_skills"] = ["calibration"]
    graph = CapabilityGraph([primary, support])
    disabled = expand_requirements(graph, ["statistics"], "calibration", predecessor_expansion=True, capability_completion=False)
    enabled = expand_requirements(graph, ["statistics"], "calibration", predecessor_expansion=True, capability_completion=True)
    assert disabled["capability_ids"] == ["mean"]
    assert "calibration::calibrate" in disabled["unresolved_support_candidates"]
    assert "calibrate" in enabled["capability_ids"]
    assert not enabled["unresolved_support_candidates"]


def test_fixed_tool_order_preserves_instance_inputs_and_checks_scope():
    skill = declaration(
        steps=[
            {
                "id": "mean",
                "capability": "mean",
                "operation": "Mean",
                "depends_on": [],
                "tool_selection": {"candidates": ["ZTool", "ATool", "BTool"]},
            }
        ]
    )
    skill["tool_candidates"] = ["ZTool", "ATool", "BTool"]
    graph = CapabilityGraph([skill])
    plan = SkillPlanner([skill]).plan("statistics")

    def spec(name, fields):
        return ToolSpec(
            id=name,
            description="Mean",
            input_schema={
                "type": "object",
                "properties": {key: {} for key in fields},
                "additionalProperties": False,
            },
        )

    specs = {
        name: spec(name, fields)
        for name, fields in [("ATool", ["y"]), ("BTool", ["x"]), ("ZTool", ["x"])]
    }
    original = ExecutionPlan(
        [
            PlanStep(
                id="a",
                tool="ZTool",
                purpose="first cohort",
                capability_id="mean",
                skill_id="statistics",
                arguments={"x": [1, 3]},
            ),
            PlanStep(
                id="b",
                tool="ZTool",
                purpose="second cohort",
                capability_id="mean",
                skill_id="statistics",
                arguments={"x": [2, 4]},
            ),
        ]
    )
    chosen, decisions = select_first_valid(graph, plan, original, specs, specs)
    assert list(chosen) == ["BTool", "BTool"]
    assert [step.arguments for step in chosen.steps] == [step.arguments for step in original.steps]
    assert [step.id for step in chosen.steps] == ["a", "b"]
    assert decisions[0]["rejected"] == [
        {"tool": "ATool", "reason": "fixed_input_fields_outside_schema"}
    ]
    assert list(original) == ["ZTool", "ZTool"]


def native(name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": uuid4().hex,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class Client:
    def __init__(self, invalid_first_plan=False):
        self.requests = []
        self.plans = 0
        self.invalid_first_plan = invalid_first_plan

    async def chat(self, messages, **kwargs):
        self.requests.append((deepcopy(messages), kwargs))
        if kwargs["stage"] in {"tool_planning", "skill_tool_planning"}:
            self.plans += 1
            steps = [
                {
                    "id": "mean",
                    "tool": "RemoteSensingMean",
                    "purpose": "Compute statistic",
                    "capability_id": "mean",
                    "skill_id": "statistics",
                }
            ]
            if self.invalid_first_plan and self.plans == 1:
                raise ValueError("Initial planning temporarily unavailable")
            message = native("submit_tool_plan", {"steps": steps})
        elif kwargs["stage"] == "tool_argument_generation":
            message = native("RemoteSensingMean", {"x": [1, 3]})
        else:
            message = {"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:2"}
        return ChatReply(message=message)


class Tools:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def specs(self):
        return [
            ToolSpec(id="RemoteSensingMean", description="Mean", input_schema={"type": "object"})
        ]

    async def execute(self, call):
        self.calls.append(call)
        if self.fail_first and len(self.calls) == 1:
            return ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="domain", message="Bad input"),
            )
        return ToolResult(call_id=call.id, tool=call.tool, succeeded=True, data={"value": 2})


class KG:
    async def retrieve(self, query):
        return [["RemoteSensingMean", "has the functionality that", "Mean"]]

    def get_tool_subgraph(self, name, **kwargs):
        return [
            [name, "has the functionality that", "Mean"],
            [name, "inputs", "x: array"],
            [name, "outputs", "value: scalar"],
        ]

    def get_tool_info(self, relations):
        return {"input_format": "x: array", "output_format": "value: scalar"}


async def run(presets=(), fail_first=False, invalid_first_plan=False, tool_attempts=2):
    tools, client = Tools(fail_first), Client(invalid_first_plan)
    context = AgentContext(
        tools,
        None,
        SimpleNamespace(emit=lambda *args: None),
        client=client,
        tool_retriever=KG(),
        planner=SkillPlanner([declaration()]),
        options=AlgorithmSettings(
            planning_mode="skill_guided", plan_attempts=2, tool_attempts=tool_attempts
        ),
    )
    result = await AgentEngine().run(
        TaskInput(question="Compute statistics", ablations=list(presets)), context
    )
    return result, tools, client


@pytest.mark.parametrize(
    "presets",
    [
        [],
        ["A1"],
        ["A2"],
        ["A3"],
        ["A4"],
        ["A5"],
        ["A6"],
        ["BIPARTITE"],
        ["A1", "A2", "A3", "A5", "A6"],
    ],
)
async def test_presets_reach_real_engine_and_prompt(presets):
    result, tools, client = await run(presets)
    assert result.tool_chain_completed and len(tools.calls) == 1
    effective = result.provenance["effective_algorithm"]
    assert effective["module_resolution"]["ablations"] == presets
    events = result.execution_events()
    assert any(row["type"] == "module_policy" for row in events)
    serialized = json.dumps(client.requests[0][0])
    if "A1" in presets:
        assert "Plan directly over tool execution instances" in serialized
        assert "flat_tools" in serialized
    if "BIPARTITE" in presets:
        assert "incidence_edges" in serialized
    if "A5" in presets:
        assert any(row["type"] == "tool_selection" for row in events)
    if "A3" in presets or "A4" in presets:
        assert not any(row["type"] == "shared_capability_bridge_search" for row in events)


@pytest.mark.parametrize("tool_attempts", [1, 2])
async def test_recovery_switch_blocks_both_tool_repair_and_replanning(tool_attempts):
    enabled, tools, _ = await run(fail_first=True, tool_attempts=tool_attempts)
    assert enabled.tool_chain_completed and len(tools.calls) == 2
    disabled, tools, client = await run(["A6"], fail_first=True, tool_attempts=tool_attempts)
    assert not disabled.tool_chain_completed and len(tools.calls) == 1
    assert client.plans == 1
    assert any(
        row["type"] == "execution_feedback" and row["status"] == "recovery_disabled"
        for row in disabled.execution_events()
    )


async def test_no_recovery_preserves_initial_planning_retry():
    result, tools, client = await run(["A6"], invalid_first_plan=True)
    assert result.tool_chain_completed and len(tools.calls) == 1
    assert client.plans == 2
