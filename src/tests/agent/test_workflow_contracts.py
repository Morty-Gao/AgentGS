"""Scientific plan invariants and controlled execution under different budgets."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.planning.capabilities import CapabilityGraph
from agentgs.agent.planning.grounding import ground_plan
from agentgs.agent.planning.instances import ExecutionPlan, PlanStep, resolve_inputs, reuse_key
from agentgs.agent.planning.selection import select_skill_ids
from agentgs.agent.prompts import StagePrompt, tool_feedback
from agentgs.benchmark.scope import select_scope
from agentgs.client.context import fit_context, token_bound
from agentgs.client.errors import BudgetExceeded
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.models import TaskInput, ToolError, ToolResult, ToolSpec


def skill(id, operations):
    return {
        "id": id,
        "inputs": [],
        "outputs": [],
        "tool_candidates": [],
        "workflow": {"steps": operations},
    }


def node(id, deps=(), required=True):
    return {
        "id": id,
        "capability": id,
        "depends_on": list(deps),
        "required": required,
        "tool_selection": {"candidates": ["T" + id.upper()]},
    }


def test_scoped_predecessors_order_and_missing_diagnostics():
    graph = CapabilityGraph([skill("s", [node("c", ["b"]), node("b", ["a"]), node("a")])])
    full = graph.expand(["s"], complete_missing=False)
    assert full["dependency_order"] == ["s::a", "s::b", "s::c"]
    partial = graph.expand(["s"], expand_predecessors=False, complete_missing=False)
    assert partial["capability_ids"] == ["c"]
    assert partial["unresolved_capabilities"] == ["s::b"]
    other = CapabilityGraph(
        [skill("ab", [node("a"), node("b", ["a"])]), skill("cb", [node("c"), node("b", ["c"])])]
    )
    assert other.expand(["cb"])["dependency_order"] == ["cb::c", "cb::b"]


def test_disconnected_cycles_and_unknown_predecessors_are_detected():
    graph = CapabilityGraph([skill("s", [node("a", ["b"]), node("b", ["a"]), node("c")])])
    with pytest.raises(ValueError, match="Cyclic"):
        graph.expand(["s"])
    graph = CapabilityGraph([skill("s", [node("a", ["absent"])])])
    assert graph.expand(["s"])["unresolved_capabilities"] == ["s::absent"]


def test_grounding_cannot_admit_missing_capability_or_bad_order():
    graph = CapabilityGraph([skill("s", [node("a"), node("b", ["a"])])])
    expansion = graph.expand(["s"])
    plan = {
        "capabilities": expansion["capabilities"],
        "steps": [],
        "constraints": [],
        "capability_expansion": expansion,
    }
    for tools in (["TB"], ["TB", "TA"]):
        result = ground_plan(graph, plan, ["TA", "TB"], proposed_chain=tools)
        assert not result["validation"]["valid"]
    result = ground_plan(graph, plan, ["TA", "TB"], proposed_chain=["TA", "TB"])
    assert result["validation"]["valid"]
    assert not result["validation"]["schema_checked"]
    partial = graph.expand(["s"], expand_predecessors=False, complete_missing=False)
    plan.update(
        capabilities=partial["capabilities"],
        capability_expansion=partial,
        missing_capabilities=partial["unresolved_capabilities"],
    )
    assert ground_plan(graph, plan, ["TA", "TB"], proposed_chain=["TA", "TB"])["validation"][
        "valid"
    ]


def test_binding_and_reuse_respect_input_identity_and_artifact_state():
    steps = ExecutionPlan(
        [
            PlanStep(
                id="a", tool="RemoteSensingMean", purpose="First cohort", arguments={"x": [1, 3]}
            ),
            PlanStep(
                id="b",
                tool="RemoteSensingDifference",
                purpose="Compare",
                depends_on=["a"],
                input_bindings={"a": {"step_id": "a", "path": ["value"]}},
                arguments={"b": 1},
            ),
        ]
    )
    assert resolve_inputs(steps.steps[1], {"a": {"value": 2}}) == {"a": 2, "b": 1}
    with pytest.raises(ValueError, match="Missing upstream"):
        resolve_inputs(steps.steps[1], {"a": {}})
    original = reuse_key(steps.steps[0], {}, "q")
    changed = steps.steps[0].model_copy(update={"arguments": {"x": [3, 5]}})
    assert original != reuse_key(changed, {}, "q")
    assert (
        reuse_key(changed.model_copy(update={"arguments": {"path": "/work/data"}}), {}, "q") is None
    )
    assert (
        steps.align(["Load", "RemoteSensingMean", "RemoteSensingDifference"]).steps[-1]
        == steps.steps[-1]
    )
    with pytest.raises(ValueError, match="reorder"):
        steps.align(["RemoteSensingDifference", "RemoteSensingMean"])


def native(name, args):
    return {
        "tool_calls": [
            {
                "id": uuid4().hex,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
    }


class Events:
    def emit(self, *args):
        pass


class ScalarTools:
    def __init__(self):
        self.calls = []
        self.failed = False

    def specs(self):
        return [
            ToolSpec(id=name, description="Scalar", input_schema={"type": "object"})
            for name in ["RemoteSensingMean", "RemoteSensingDifference"]
        ]

    async def execute(self, call):
        self.calls.append(call)
        if call.tool == "RemoteSensingDifference" and not self.failed:
            self.failed = True
            return ToolResult(
                call_id=call.id,
                tool=call.tool,
                succeeded=False,
                error=ToolError(code="temporary", message="Retry"),
            )
        return ToolResult(call_id=call.id, tool=call.tool, succeeded=True, data={"value": 2})


@pytest.mark.parametrize("copies", [1, 2])
async def test_replanning_reuses_only_fully_bound_scalar_instances(monkeypatch, copies):
    def forbidden_binding(*args):
        raise AssertionError(
            "Disabled binding assistance must not run the implicit binding pipeline"
        )

    monkeypatch.setattr("agentgs.agent.execution.service.prepare_tool_input", forbidden_binding)
    steps = [
        PlanStep(id="mean", tool="RemoteSensingMean", purpose="Mean", arguments={"x": [1, 3]}),
        PlanStep(
            id="diff",
            tool="RemoteSensingDifference",
            purpose="Difference",
            arguments={"a": 3, "b": 1},
        ),
    ]
    if copies == 2:
        steps.insert(1, steps[0].model_copy(update={"id": "independent-mean"}, deep=True))

    class Client:
        async def chat(self, messages, *, stage, **kwargs):
            if stage.endswith("planning"):
                value = native(
                    "submit_tool_plan", {"steps": [s.model_dump(mode="json") for s in steps]}
                )
            elif stage == "tool_argument_generation":
                step = next(s for s in steps if s.tool == kwargs["tool_choice"]["function"]["name"])
                value = native(step.tool, step.arguments)
            else:
                value = {"content": "IsCompleted:Completed\nFinalAnswer:2"}
            return ChatReply(message=value)

    tools = ScalarTools()
    context = AgentContext(
        tools,
        None,
        Events(),
        client=Client(),
        options=AlgorithmSettings(
            planning_mode="tool_only",
            candidate_pool="all_registered",
            routing_hints=False,
            binding_assistance=False,
            tool_attempts=1,
        ),
    )
    result = await AgentEngine().run(TaskInput(question="Compute the requested value"), context)
    assert result.tool_chain_completed
    assert [c.tool for c in tools.calls] == ["RemoteSensingMean"] * copies + [
        "RemoteSensingDifference",
        "RemoteSensingDifference",
    ]
    assert any(e["type"] == "plan_resume" for e in result.execution_events())


async def test_disabled_execution_recovery_still_repairs_native_call_format():
    class Client:
        def __init__(self):
            self.arguments = 0

        async def chat(self, messages, *, stage, **kwargs):
            if stage.endswith("planning"):
                value = native(
                    "submit_tool_plan",
                    {"steps": [{"id": "mean", "tool": "RemoteSensingMean", "purpose": "Mean"}]},
                )
            elif stage == "tool_argument_generation":
                self.arguments += 1
                value = native(
                    "RemoteSensingMean", "not an object" if self.arguments == 1 else {"x": [1, 3]}
                )
            else:
                value = {"content": "IsCompleted:Completed\nFinalAnswer:2"}
            return ChatReply(message=value)

    tools, client = ScalarTools(), Client()
    context = AgentContext(
        tools,
        None,
        Events(),
        client=client,
        options=AlgorithmSettings(
            planning_mode="tool_only",
            candidate_pool="all_registered",
            execution_recovery=False,
            routing_hints=False,
        ),
    )
    result = await AgentEngine().run(TaskInput(question="Compute the requested value"), context)
    assert result.tool_chain_completed and len(tools.calls) == 1
    assert client.arguments == 2
    assert any(e["type"] == "protocol_call" for e in result.execution_events())


def test_context_budget_preserves_pairs_and_rejects_oversized_required_inputs():
    request = {
        "max_tokens": 20,
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "task"},
            {"role": "assistant", **native("T", {})},
            {"role": "tool", "tool_call_id": "id", "content": "x" * 2000},
            {"role": "assistant", **native("T", {})},
            {"role": "tool", "tool_call_id": "next", "content": "current evidence"},
        ],
    }
    result, compacted = fit_context(request, 1100)
    assert compacted and token_bound(result) + 20 <= 1100
    assert result["messages"][-1] == request["messages"][-1]
    assert [m["role"] for m in result["messages"]] == [m["role"] for m in request["messages"]]
    with pytest.raises(BudgetExceeded):
        fit_context({"messages": [{"role": "user", "content": "x" * 2000}], "max_tokens": 20}, 100)


def test_reminder_does_not_promote_observations_to_instructions():
    prompt = StagePrompt("Call the next tool", {"evidence": "INJECTED_OBSERVATION"})
    text = tool_feedback("raw", prompt, "execute")
    assert "INJECTED_OBSERVATION" in text.split("<system-reminder>")[0]
    assert "INJECTED_OBSERVATION" not in text.split("<system-reminder>")[1]


def test_common_data_selection_is_identical_across_modes():
    rows = [
        SimpleNamespace(question_id=1, category="GeoBigDataTools"),
        SimpleNamespace(question_id=2, category="Other"),
    ]
    for mode in ["skill_guided", "tool_only", "data_only"]:
        selected, policy = select_scope(rows, "common_data", planning_mode=mode)
        assert [e.question_id for e in selected] == [2]
        assert policy["excluded"] == [
            {
                "question_id": 1,
                "reason": "materialized_query_lookup_unavailable_to_bash",
            }
        ]
    selected, policy = select_scope(rows, "full", planning_mode="data_only")
    assert [e.question_id for e in selected] == [1, 2]
    assert policy["excluded"] == []


def test_common_data_offline_completeness_uses_selected_population(tmp_path):
    from agentgs.benchmark.score import score_predictions
    from agentgs.evaluation.datasets.base import Dataset, Example

    class Sample(Dataset):
        domain_count = 2

        def load(self):
            return [
                Example(
                    question_id=i,
                    category_id=i,
                    category=category,
                    question="Compute",
                    expected_answer="2",
                    answer_type="number",
                )
                for i, category in [(1, "GeoBigDataTools"), (2, "Other")]
            ]

    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"question_id": 2, "answer": "2"}) + "\n")
    report = score_predictions(
        Sample(tmp_path / "source.jsonl"),
        predictions,
        tmp_path / "scored.json",
        benchmark_scope="common_data",
    )
    assert report["complete"] and report["expected"] == 1


def test_generic_instruction_does_not_select_an_unrelated_scientific_skill():
    declarations = [
        dict(skill("s", [node("a")]), description="Compute spatial distances", keywords=[])
    ]
    assert select_skill_ids(declarations, "Compute the requested statistic.") == []
