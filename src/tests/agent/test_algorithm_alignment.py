"""Library coverage, scoped dependency validity and historical audit reporting."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from agentgs.agent.planning.candidate_context import build_candidate_context
from agentgs.agent.planning.capabilities import CapabilityGraph
from agentgs.agent.planning.grounding import ground_plan
from agentgs.agent.planning.selection import select_skill_ids
from agentgs.agent.planning.skills import SkillPlanner, builtin_planner
from agentgs.agent.retrieval.scored_skills import WeightedSkillRetriever
from agentgs.contracts.tool_feedback import assess_tool_execution
from agentgs.evaluation.analysis.alignment import (
    alignment_report,
    decision_digest,
    verify_components,
)
from agentgs.skills.definitions import (
    load_capability_extensions,
    load_library_scope,
    load_skill_definitions,
)

ROOT = Path(__file__).parents[3]
SNAPSHOT = json.loads((ROOT / "src/tests/fixtures/alignment/baseline.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def definitions():
    return load_skill_definitions()


@pytest.fixture(scope="module")
def graph(definitions):
    return CapabilityGraph(definitions, extension_capabilities=load_capability_extensions())


def test_original_declarations_and_both_graphs(definitions, graph):
    assert definitions == SNAPSHOT["library"]
    assert set(graph.edges) == {skill["id"] for skill in definitions}
    assert graph.tool_mapping() == SNAPSHOT["tool_graph"]
    assert len(definitions) == 24 and len(graph.capabilities) == 31
    assert len(graph.tool_capabilities) == 101
    scope = load_library_scope()
    assert set(graph.tool_capabilities) == set(scope["tool_ids"])
    assert not set(graph.tool_capabilities) & set(scope["excluded_tool_ids"])
    for node in graph.capabilities.values():
        assert set(node["dependencies"]) <= set(graph.capabilities)
        assert set(node["source_skills"]) <= set(scope["skill_ids"])
    for edge in graph.edges.values():
        # Historical semantic hints such as spatial-clip remain unresolved in
        # the original library; only newly excluded Skills must disappear.
        assert not set(edge["compatible_skills"]) & set(scope["excluded_skill_ids"])


def test_runtime_registries_respect_the_selected_scope():
    from agentgs.runtime.tools.registry import builtin_tools
    from agentgs.skills.registry import builtin_skills

    scope = load_library_scope()
    assert set(builtin_tools()) == set(scope["tool_ids"])
    assert all(callable(tool.handler) for tool in builtin_tools().values())
    assert set(builtin_skills()) == set(scope["skill_ids"])
    assert all(set(s.spec.tools) <= set(scope["tool_ids"]) for s in builtin_skills().values())


@pytest.mark.parametrize("case", SNAPSHOT["library_profiles"], ids=lambda case: case["id"])
def test_startup_profile_preserves_library_and_graph_when_extensions_are_disabled(case):
    planner = builtin_planner(mapping_extensions=case["mapping_extensions"])
    assert list(planner.skills) == case["skill_ids"]
    assert set(planner.graph.edges) == set(case["skill_ids"])
    assert decision_digest(planner.graph.tool_mapping()) == case["tool_graph_sha256"]
    assert planner.plan(case["query"])["mode"] == "skill_guided"


@pytest.mark.parametrize("case", SNAPSHOT["cases"], ids=lambda case: case["id"])
def test_ranking_selection_complete_plan_and_expansion(case, definitions, graph):
    matches = WeightedSkillRetriever(
        definitions, mapping_extensions=case["mapping_extensions"]
    ).retrieve(case["query"], tool_names=case["inferred_tools"])
    actual = [
        {
            "skill_id": match.skill_id,
            "score": match.score,
            "matched_terms": list(match.matched_terms),
            "score_components": match.score_components,
        }
        for match in matches
    ]
    assert actual == case["matches"]
    selected = select_skill_ids(
        definitions,
        case["query"],
        case["inferred_tools"],
        mapping_extensions=case["mapping_extensions"],
    )
    assert set(selected) <= {skill["id"] for skill in definitions}
    expansion = graph.expand(selected, case["query"])
    positions = {key: i for i, key in enumerate(expansion["dependency_order"])}
    for step in expansion["instances"]:
        assert all(dep in positions and positions[dep] < positions[step["id"]] for dep in step["depends_on"])
    plan = SkillPlanner(
        definitions,
        extension_capabilities=load_capability_extensions(),
        mapping_extensions=case["mapping_extensions"],
    ).plan(case["query"], case["inferred_tools"])
    assert {row["skill_id"] for row in plan["constraints"]} == set(selected)
    assert not plan["capability_expansion"]["unresolved_capabilities"]


@pytest.mark.parametrize("case", SNAPSHOT["candidate_cases"], ids=lambda case: case["id"])
def test_candidate_weights_availability_and_order(case, graph):
    assert (
        graph.candidates(
            graph.capabilities,
            case["available_tools"],
            case["inferred_tools"],
            case["schemas"],
            case["artifact_types"],
        )
        == case["expected"]
    )


@pytest.mark.parametrize("case", SNAPSHOT["grounding_cases"], ids=lambda case: case["id"])
def test_grounding_repeated_instances_cross_skill_tools_and_missing_backends(
    case, definitions, graph
):
    plan = SkillPlanner(definitions, extension_capabilities=load_capability_extensions()).plan(
        case["query"], case["inferred_tools"]
    )
    actual = ground_plan(
        graph,
        plan,
        case["available_tools"],
        case["inferred_tools"],
        proposed_chain=case["proposed_chain"],
        tool_schemas=case["schemas"],
        tool_next=case["tool_next"],
        artifact_types=case["artifact_types"],
    )
    assert actual.pop("tool_graph") == SNAPSHOT["tool_graph"]
    assert actual["tool_chain"] == case["expected"]["tool_chain"]
    assert actual["validation"]["valid"] == (not actual["validation"]["errors"])
    assert len(actual["execution_steps"]) == len(actual["tool_chain"])


@pytest.mark.parametrize("case", SNAPSHOT["feedback_cases"], ids=lambda case: case["id"])
def test_domain_negative_results_are_distinct_from_execution_failure(case):
    actual = assess_tool_execution(
        case["output"],
        input_format=case["input_format"],
        attempt=case["attempt"],
        max_attempts=case["max_attempts"],
    )
    assert actual.as_dict() == case["expected"]


@pytest.mark.parametrize("case", SNAPSHOT["context_cases"], ids=lambda case: case["id"])
def test_small_prompt_budgets_preserve_every_required_contract(case):
    assert (
        json.loads(build_candidate_context(case["references"], case["max_chars"]))
        == case["expected"]
    )


def test_audit_rejects_drift_and_does_not_certify_unverified_runtime():
    assert verify_components(SNAPSHOT)
    changed = deepcopy(SNAPSHOT)
    changed["cases"][0]["matches"][0]["score"] += 0.1
    assert any(not check["passed"] for check in verify_components(changed))
    # The retired migration document is not a runtime dependency. This contract
    # explicitly describes unverified runtime evidence against the frozen oracle.
    contract = {
        "baseline": {
            "id": SNAPSHOT["baseline_id"],
            "tree_sha256": SNAPSHOT["source_tree_sha256"],
            "selection": {"sha256": SNAPSHOT["selection_sha256"]},
        },
        "acceptance": {"production_algorithm": "geotoolagent-native-function-calling-v1"},
        "algorithms": [{"id": "entry-routing", "implementation_status": "implemented"}],
    }
    report = alignment_report(contract, SNAPSHOT)
    assert report["component_parity"] is False
    assert report["runtime_equivalent"] is False
    assert "entry-routing" in report["runtime_pending_algorithms"]
