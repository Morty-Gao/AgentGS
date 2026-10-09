"""Task adoption and executable bridges over the packaged scientific declarations."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agentgs.agent.base import AgentContext
from agentgs.agent.engine import AgentEngine
from agentgs.agent.planning.grounding import ground_plan
from agentgs.agent.planning.instances import ExecutionPlan
from agentgs.agent.planning.skills import builtin_planner
from agentgs.client.models import ChatReply
from agentgs.config.algorithm import AlgorithmSettings
from agentgs.contracts.catalog import load_tool_scope
from agentgs.contracts.models import TaskInput, ToolResult
from agentgs.datasets.registry import DatasetRegistry
from agentgs.evaluation.analysis.conversation import tool_executions
from agentgs.runtime.artifacts.store import LocalArtifactStore
from agentgs.runtime.tools.executor import LocalToolExecutor
from agentgs.runtime.tools.registry import builtin_tools


def native(name, arguments):
    return ChatReply(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }
    )


def test_adopted_model_query_does_not_require_an_unrelated_retrieved_strategy():
    planner = builtin_planner()
    question = "According to LITHO1.0, what is the Moho depth at the model node nearest Beijing?"
    candidates = {row["id"] for row in planner.retrieve(question)["declarations"]}
    assert {"deep-earth-model-query", "earthquake-event-model-sampling"} <= candidates
    plan = planner.plan(question, requirements={"deep-earth-model-query": ["model-query"]})
    execution = ExecutionPlan(
        [
            {
                "id": "beijing",
                "tool": "DeepEarthQueryLITHO1",
                "purpose": "Read Moho depth",
                "skill_id": "deep-earth-model-query",
                "capability_id": "model-query",
            }
        ]
    )
    grounded = ground_plan(
        planner.graph,
        plan,
        load_tool_scope()["tool_ids"],
        proposed_chain=execution,
        execution_plan=execution,
    )
    assert grounded["validation"]["valid"], grounded["validation"]
    assert [r["id"] for r in grounded["required_instances"]] == [
        "deep-earth-model-query::model-query"
    ]
    assert plan["support_candidates"]


def test_intermediate_goal_does_not_import_downstream_operations():
    planner = builtin_planner()
    plan = planner.plan(
        "Compute a remote sensing raster mean.",
        requirements={"remote-sensing-raster-statistics": ["raster-aggregate"]},
    )
    assert plan["selected_requirements"] == {
        "remote-sensing-raster-statistics": ["raster-aggregate"]
    }
    assert "remote-sensing-raster-statistics::raster-algebra" not in {
        row["id"] for row in plan["capability_expansion"]["instances"]
    }


@pytest.mark.parametrize("ablations", [[], ["A1"], ["A3"], ["A5"], ["BIPARTITE"]])
@pytest.mark.parametrize("decline_all", [False, True])
async def test_single_query_execution_rejects_irrelevant_support(ablations, decline_all):
    class Tools:
        def specs(self):
            return [row.spec for row in builtin_tools().values()]

        async def execute(self, call):
            assert call.tool == "DeepEarthQueryLITHO1"
            return ToolResult(
                call_id=call.id, tool=call.tool, succeeded=True, data={"value_km": 46.54}
            )

    stages = []

    class Client:
        async def chat(self, messages, *, stage, **kwargs):
            stages.append(stage)
            if stage == "capability_selection":
                if decline_all:
                    return native("select_capabilities", {"requirements": []})
                return native(
                    "select_capabilities",
                    {
                        "requirements": [
                            {"skill_id": "deep-earth-model-query", "capability_id": "model-query"}
                        ]
                    },
                )
            if stage == "capability_completion":
                return native("select_capabilities", {"requirements": []})
            if stage == "skill_tool_planning":
                if decline_all:
                    assert "No strategy was adopted" in messages[-1]["content"]
                    return native(
                        "submit_tool_plan",
                        {
                            "steps": [
                                {
                                    "id": "beijing",
                                    "tool": "DeepEarthQueryLITHO1",
                                    "purpose": "Read Moho depth",
                                }
                            ]
                        },
                    )
                return native(
                    "submit_tool_plan",
                    {
                        "steps": [
                            {
                                "id": "beijing",
                                "tool": "DeepEarthQueryLITHO1",
                                "purpose": "Read Moho depth",
                                "skill_id": "deep-earth-model-query",
                                "capability_id": "model-query",
                            }
                        ]
                    },
                )
            if stage == "tool_argument_generation":
                return native("DeepEarthQueryLITHO1", {"location_key": "beijing", "layer": "moho"})
            return ChatReply(
                message={"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:46.54"}
            )

    context = AgentContext(
        Tools(),
        None,
        SimpleNamespace(emit=lambda *args: None),
        client=Client(),
        options=AlgorithmSettings(
            planning_mode="skill_guided",
            candidate_pool="all_registered",
            binding_assistance=False,
            plan_attempts=1,
        ),
    )
    result = await AgentEngine().run(
        TaskInput(
            question="According to LITHO1.0, what is the Moho depth at the model node nearest Beijing?",
            ablations=ablations,
        ),
        context,
    )
    assert result.tool_chain_completed
    assert [call["tool_name"] for call in tool_executions(result.messages)] == [
        "DeepEarthQueryLITHO1"
    ]
    assert ("capability_completion" in stages) == ("A3" not in ablations and not decline_all)


class HazardClient:
    def __init__(self, reject_first=False):
        self.calls = []
        self.reject_first = reject_first

    async def chat(self, messages, *, stage, **kwargs):
        self.calls.append((stage, messages))
        if stage == "capability_selection":
            if self.reject_first:
                self.reject_first = False
                return native(
                    "select_capabilities",
                    {
                        "requirements": [
                            {
                                "skill_id": "hazard-table-loading",
                                "capability_id": "model-query",
                            }
                        ]
                    },
                )
            return native(
                "select_capabilities",
                {
                    "requirements": [
                        {
                            "skill_id": "hazard-table-loading",
                            "capability_id": "hazard-table-source",
                        },
                        {
                            "skill_id": "hazard-subset-counting",
                            "capability_id": "hazard-subset-count",
                        },
                    ]
                },
            )
        if stage == "skill_tool_planning":
            return native(
                "submit_tool_plan",
                {
                    "steps": [
                        {
                            "id": "load",
                            "tool": "NaturalHazardLoadEventTable",
                            "purpose": "Inspect columns",
                            "skill_id": "hazard-table-loading",
                            "capability_id": "hazard-table-source",
                            "arguments": {"source": "data/hazard.jsonl", "sheet": "Sheet1"},
                        },
                        {
                            "id": "count",
                            "tool": "NaturalHazardCountRecords",
                            "purpose": "Count China records",
                            "skill_id": "hazard-subset-counting",
                            "capability_id": "hazard-subset-count",
                            "depends_on": ["load"],
                            "arguments": {"filters": {"Country": "China"}},
                        },
                    ]
                },
            )
        if stage == "tool_argument_generation":
            name = kwargs["tool_choice"]["function"]["name"]
            return native(
                name,
                {"source": "data/hazard.jsonl", "sheet": "Sheet1"}
                if name == "NaturalHazardLoadEventTable"
                else {"filters": {"Country": "China"}},
            )
        return ChatReply(
            message={"role": "assistant", "content": "IsCompleted:Completed\nFinalAnswer:1"}
        )


@pytest.mark.parametrize(
    "ablations,bridge",
    [
        ([], True),
        (["A4"], False),
        (["A3"], False),
        (["A1", "A5"], True),
        (["BIPARTITE", "A2"], True),
    ],
)
async def test_packaged_bridge_executes_real_tools_with_observed_artifact_bindings(
    project, ablations, bridge
):
    (project / "data/hazard.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [{"Country": "China"}, {"Country": "Japan"}]) + "\n"
    )
    artifacts = LocalArtifactStore(project / "artifacts")
    events = SimpleNamespace(emit=lambda *args: None)
    tools = LocalToolExecutor(DatasetRegistry(project), artifacts, events)
    client = HazardClient(reject_first=True)
    context = AgentContext(
        tools,
        artifacts,
        events,
        client=client,
        options=AlgorithmSettings(
            planning_mode="skill_guided",
            candidate_pool="all_registered",
            binding_assistance=False,
            plan_attempts=1,
        ),
    )
    try:
        result = await AgentEngine().run(
            TaskInput(
                question="Count disaster records with Country equal to China in the natural hazard event table.",
                input_resource={"source": "data/hazard.jsonl", "sheet": "Sheet1"},
                ablations=ablations,
            ),
            context,
        )
    finally:
        await tools.aclose()
    assert result.tool_chain_completed, [
        e for e in result.execution_events() if e["type"] == "plan_retry"
    ]
    assert result.answer == "1"
    executed = [e for e in result.execution_events() if e["type"] == "tool_call"]
    assert len(tool_executions(result.messages)) == len(executed)
    assert [e["tool_name"] for e in executed] == ["NaturalHazardLoadEventTable"] + (
        ["NaturalHazardFilterRecords"] if bridge else []
    ) + ["NaturalHazardCountRecords"]
    if bridge:
        loaded = json.loads(executed[0]["tool_output"])
        filtered = json.loads(executed[1]["tool_output"])
        filter_input = json.loads(executed[1]["tool_input"])
        count_input = json.loads(executed[2]["tool_input"])
        assert filter_input["source"] == loaded["source"]
        assert count_input["filtered_records_path"] == filtered["filtered_records_path"]
    assert sum(stage == "capability_selection" for stage, _ in client.calls) == 2
    selection = next(e for e in result.execution_events() if e["type"] == "capability_selection")
    assert set(selection["requirements"]) == {"hazard-table-loading", "hazard-subset-counting"}
