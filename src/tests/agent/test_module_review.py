"""Scientific scope and dependency checks across composable module policies."""

from copy import deepcopy

import pytest

from agentgs.agent.engine import MutablePlan
from agentgs.agent.planning.bridge import (
    complete_shared_capability_bridges,
    ground_shared_bridge_chain,
)
from agentgs.agent.planning.capabilities import CapabilityGraph
from agentgs.agent.planning.grounding import ground_plan
from agentgs.agent.planning.instances import ExecutionPlan, bind_bridge_dependencies
from agentgs.agent.planning.module_context import planning_context
from agentgs.agent.planning.module_expansion import expand_requirements
from agentgs.agent.planning.module_selection import select_first_valid
from agentgs.contracts.models import ToolSpec


def operation(cid, tool, dependencies=(), **extra):
    return {
        "id": cid,
        "capability": cid,
        "operation": cid,
        "depends_on": list(dependencies),
        "tool_selection": {"candidates": [tool]},
        **extra,
    }


def strategy(sid, operations, compatible=()):
    return {
        "id": sid,
        "inputs": [],
        "outputs": [],
        "tool_candidates": [],
        "planning": {"compatible_skills": list(compatible)},
        "workflow": {"steps": operations},
    }


def plan_for(graph, selected, **options):
    expansion = graph.expand(selected, **options)
    return {
        "capabilities": expansion["capabilities"],
        "capability_expansion": expansion,
        "missing_capabilities": expansion["unresolved_capabilities"],
        "constraints": [],
        "steps": [{"skill_id": sid} for sid in selected],
        "declarations": [graph.skills[sid] for sid in selected],
    }


def instance(identity, tool, cid, sid, dependencies=()):
    return {
        "id": identity,
        "tool": tool,
        "purpose": identity,
        "capability_id": cid,
        "skill_id": sid,
        "depends_on": list(dependencies),
    }


def test_first_valid_allows_explicit_predecessors_when_both_insertion_phases_are_off():
    graph = CapabilityGraph(
        [strategy("s", [operation("load", "Load"), operation("mean", "Mean", ["load"])])]
    )
    plan = plan_for(graph, ["s"], expand_predecessors=False, complete_missing=False)
    proposed = ExecutionPlan(
        [instance("load", "Load", "load", "s"), instance("mean", "Mean", "mean", "s", ["load"])]
    )
    specs = {
        name: ToolSpec(id=name, description=name, input_schema={"type": "object"})
        for name in proposed
    }
    selected, _ = select_first_valid(graph, plan, proposed, specs, specs)
    grounded = ground_plan(graph, plan, specs, proposed_chain=selected, execution_plan=selected)
    assert grounded["validation"]["valid"]
    assert plan["capability_expansion"]["capability_ids"] == ["mean"]


def test_explicit_optional_operation_has_the_same_scope_under_both_selection_policies():
    graph = CapabilityGraph(
        [strategy("s", [operation("main", "Main"), operation("extra", "Extra", required=False)])]
    )
    plan = plan_for(graph, ["s"])
    proposed = ExecutionPlan(
        [instance("main", "Main", "main", "s"), instance("extra", "Extra", "extra", "s")]
    )
    specs = {
        name: ToolSpec(id=name, description=name, input_schema={"type": "object"})
        for name in proposed
    }
    fixed, _ = select_first_valid(graph, plan, proposed, specs, specs)
    for execution in (proposed, fixed):
        grounded = ground_plan(
            graph, plan, specs, proposed_chain=execution, execution_plan=execution
        )
        assert grounded["validation"]["valid"], grounded["validation"]
        assert "extra" in grounded["capability_candidates"]
        assert {row["id"] for row in grounded["required_instances"]} == {"s::main", "s::extra"}
    assert plan["capability_expansion"]["capability_ids"] == ["main"]


def test_tool_overlap_cannot_satisfy_another_scoped_capability():
    graph = CapabilityGraph(
        [strategy("s", [operation("a", "Shared")]), strategy("t", [operation("b", "Shared")])]
    )
    plan = plan_for(graph, ["s", "t"])
    proposed = ExecutionPlan([instance("a", "Shared", "a", "s")])
    grounded = ground_plan(
        graph, plan, ["Shared"], proposed_chain=proposed, execution_plan=proposed
    )
    assert not grounded["validation"]["valid"]
    assert "Missing required capability: t::b" in grounded["validation"]["errors"]


def test_identical_tool_candidates_do_not_resolve_ambiguous_strategy_identity():
    graph = CapabilityGraph(
        [strategy("s", [operation("mean", "Mean")]), strategy("t", [operation("mean", "Mean")])]
    )
    plan = plan_for(graph, ["s", "t"])
    proposed = ExecutionPlan([instance("x", "Mean", "mean", None)])
    specs = {"Mean": ToolSpec(id="Mean", description="Mean", input_schema={"type": "object"})}
    with pytest.raises(ValueError, match="unambiguous skill_id"):
        select_first_valid(graph, plan, proposed, specs, specs)
    grounded = ground_plan(graph, plan, specs, proposed_chain=proposed, execution_plan=proposed)
    assert not grounded["validation"]["valid"]
    assert "Ambiguous capability or strategy assignment: x" in grounded["validation"]["errors"]


def test_each_consumer_instance_requires_a_declared_producer_dependency():
    graph = CapabilityGraph([strategy("s", [operation("a", "A"), operation("b", "B", ["a"])])])
    plan = plan_for(graph, ["s"])
    steps = [
        instance("a", "A", "a", "s"),
        instance("b1", "B", "b", "s", ["a"]),
        instance("b2", "B", "b", "s"),
    ]
    invalid = ExecutionPlan(steps)
    assert not ground_plan(graph, plan, ["A", "B"], proposed_chain=invalid, execution_plan=invalid)[
        "validation"
    ]["valid"]
    steps[-1]["depends_on"] = ["a"]
    valid = ExecutionPlan(steps)
    assert ground_plan(graph, plan, ["A", "B"], proposed_chain=valid, execution_plan=valid)[
        "validation"
    ]["valid"]


def test_bipartite_has_nodes_for_inactive_declared_operations():
    graph = CapabilityGraph(
        [
            strategy(
                "s",
                [
                    operation("a", "A"),
                    operation("b", "B", ["a"]),
                    operation("optional", "O", required=False),
                ],
            )
        ]
    )
    plan = plan_for(graph, ["s"], expand_predecessors=False, complete_missing=False)
    context = planning_context(plan, "bipartite")
    nodes = {row["capability_id"] for row in context["capability_nodes"]}
    assert nodes == {"a", "b", "optional"}
    assert all(edge["capability_id"] in nodes for edge in context["incidence_edges"])
    assert context["requirements"]["capability_ids"] == ["b"]
    with pytest.raises(ValueError, match="Unknown planning structure"):
        planning_context(plan, "typo")


def test_completion_does_not_reinsert_a_requirement_already_covered_by_a_predecessor():
    graph = CapabilityGraph(
        [
            strategy(
                "s",
                [operation("calibration", "Calibrate"), operation("mean", "Mean", ["calibration"])],
                ["support"],
            ),
            strategy(
                "support",
                [
                    operation(
                        "alternative",
                        "Alternative",
                        description="calibration",
                        operation="calibration",
                    )
                ],
            ),
        ]
    )
    expanded = expand_requirements(
        graph, ["s"], "calibration", predecessor_expansion=True, capability_completion=True
    )
    assert expanded["capability_ids"] == ["calibration", "mean"]
    assert not expanded["support_added"] and not expanded["support_candidates"]


def test_support_can_be_added_without_restoring_disabled_predecessors():
    graph = CapabilityGraph(
        [
            strategy(
                "s", [operation("load", "Load"), operation("mean", "Mean", ["load"])], ["support"]
            ),
            strategy("support", [operation("calibration", "Calibrate", operation="calibration")]),
        ]
    )
    result = expand_requirements(
        graph, ["s"], "calibration", predecessor_expansion=False, capability_completion=True
    )
    assert result["support_added"] == ["support::calibration"]
    assert result["unresolved_capabilities"] == ["s::load"]
    assert "load" not in result["capability_ids"]


def test_a_declaration_without_workflow_uses_its_goal_and_tools():
    graph = CapabilityGraph(
        [{"id": "s", "goal": {"capability": "lookup"}, "tool_candidates": ["Lookup"]}]
    )
    plan = plan_for(graph, ["s"])
    proposed = ExecutionPlan([instance("lookup", "Lookup", "lookup", "s")])
    specs = {"Lookup": ToolSpec(id="Lookup", description="Lookup", input_schema={"type": "object"})}
    selected, _ = select_first_valid(graph, plan, proposed, specs, specs)
    assert graph.scopes["s"] == {"lookup": []}
    assert ground_plan(graph, plan, specs, proposed_chain=selected, execution_plan=selected)[
        "validation"
    ]["valid"]


def test_grounding_first_valid_uses_the_same_scoped_selector():
    graph = CapabilityGraph(
        [strategy("s", [operation("mean", "Z", tool_selection={"candidates": ["Z", "A"]})])]
    )
    plan = plan_for(graph, ["s"])
    proposed = ExecutionPlan([instance("x", "Z", "mean", "s")])
    specs = {
        name: ToolSpec(id=name, description=name, input_schema={"type": "object"})
        for name in ["A", "Z"]
    }
    with pytest.raises(ValueError, match="registered tool schemas"):
        ground_plan(
            graph,
            plan,
            specs,
            proposed_chain=proposed,
            execution_plan=proposed,
            tool_selection="first_valid",
        )
    grounded = ground_plan(
        graph,
        plan,
        specs,
        proposed_chain=proposed,
        execution_plan=proposed,
        tool_selection="first_valid",
        tool_specs=specs,
    )
    assert grounded["tool_chain"] == ["A"] and grounded["validation"]["valid"]
    assert list(proposed) == ["Z"]


def test_bridge_dependencies_preserve_endpoint_scopes_and_middle_identity():
    graph = CapabilityGraph(
        [
            strategy("source", [operation("a", "A")]),
            strategy("target", [operation("b", "B")]),
            strategy(
                "connector",
                [operation("a", "A"), operation("m", "M", ["a"]), operation("b", "B", ["m"])],
            ),
        ]
    )
    plan = plan_for(graph, ["source", "target"])
    route = {
        "skill_id": "connector",
        "source_skill_id": "source",
        "target_skill_id": "target",
        "capability_path": ["a", "m", "b"],
        "execution_contract": {"a": {}, "m": {}, "b": {}},
        "data_edges": [{"source": "a", "target": "m"}, {"source": "m", "target": "b"}],
    }
    plan["capability_expansion"]["shared_bridge"] = {"accepted": [route]}
    plan["steps"].append({"skill_id": "connector", "active_bridge_capabilities": ["a", "m", "b"]})
    plan["capabilities"].append(deepcopy(graph.capabilities["m"]))
    proposed = ExecutionPlan(
        [instance("source", "A", "a", "source"), instance("target", "B", "b", "target")]
    )
    aligned = proposed.align(["A", "M", "B"])
    bound = bind_bridge_dependencies(
        aligned, {"routes": [route], "capability_indices": {"a": 1, "m": 2, "b": 3}}
    )
    assert bound.steps[1].depends_on == ["source"]
    assert bound.steps[2].depends_on == [bound.steps[1].id]
    grounded = ground_plan(graph, plan, ["A", "M", "B"], proposed_chain=bound, execution_plan=bound)
    assert grounded["validation"]["valid"], grounded["validation"]


def test_declared_bridge_search_and_disabled_control_reach_grounding():
    def typed(cid, tool, dependencies=()):
        return operation(
            cid,
            tool,
            dependencies,
            input_types=["Scalar"],
            output_types=["Scalar"],
            bridge_input_bindings={"x": {"source": dependencies[0], "path": ["value"]}}
            if dependencies
            else {},
        )

    definitions = [
        strategy("source", [typed("a", "A")]),
        strategy("target", [typed("b", "B")]),
        strategy("connector", [typed("a", "A"), typed("m", "M", ["a"]), typed("b", "B", ["m"])]),
    ]
    graph = CapabilityGraph(definitions)
    initial = plan_for(graph, ["source", "target"])
    initial.update(hypergraph={}, validation={})
    disabled = MutablePlan(**deepcopy(initial))
    report = complete_shared_capability_bridges(
        disabled, definitions, ["A", "M", "B"], enabled=False
    )
    assert not report["enabled"] and disabled.as_dict() == initial
    active = MutablePlan(**deepcopy(initial))
    report = complete_shared_capability_bridges(active, definitions, ["A", "M", "B"], enabled=True)
    assert report["accepted"], report
    original = ExecutionPlan(
        [instance("source", "A", "a", "source"), instance("target", "B", "b", "target")]
    )
    chain, state = ground_shared_bridge_chain(original, report)
    assert chain == ["A", "M", "B"]
    aligned = bind_bridge_dependencies(original.align(chain), state)
    specs = {
        name: ToolSpec(id=name, description=name, input_schema={"type": "object"}) for name in chain
    }
    fixed = {index: chain[index - 1] for index in state["capability_indices"].values()}
    selected, _ = select_first_valid(
        graph, active.as_dict(), aligned, specs, specs, fixed_tools=fixed
    )
    grounded = ground_plan(
        graph, active.as_dict(), specs, proposed_chain=selected, execution_plan=selected
    )
    assert grounded["validation"]["valid"], grounded["validation"]
