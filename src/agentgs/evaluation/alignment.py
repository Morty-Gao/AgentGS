"""Offline differential checks against decisions captured from the source oracle.

This verifies pure components. It deliberately does not certify the production
harness, backend execution, model prompts or benchmark scores.
"""

import hashlib
import json

from agentgs.agent.execution.feedback import assess_tool_execution
from agentgs.agent.planning.candidate_context import build_candidate_context
from agentgs.agent.planning.capabilities import CapabilityGraph
from agentgs.agent.planning.grounding import ground_plan
from agentgs.agent.planning.selection import select_skill_ids
from agentgs.agent.planning.skills import SkillPlanner, builtin_planner
from agentgs.agent.retrieval.scored_skills import WeightedSkillRetriever
from agentgs.skills.definitions import (
    load_capability_extensions,
    load_skill_definitions,
    load_tool_scope,
)


def decision_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def verify_components(snapshot):
    definitions = load_skill_definitions()
    extensions = load_capability_extensions()
    graph = CapabilityGraph(definitions, extension_capabilities=extensions)
    checks = []

    def compare(component, rows):
        # Each row is (case id, actual decision, independently frozen decision).
        failures = [case_id for case_id, actual, expected in rows if actual != expected]
        checks.append(
            {
                "component": component,
                "passed": not failures,
                "cases": len(rows),
                "failed_case_ids": failures,
            }
        )

    compare("skill-declarations", [("library", definitions, snapshot["library"])])
    compare("capability-graph", [("graph", graph.as_dict(), snapshot["graph"])])
    compare("tool-graph", [("tool-graph", graph.tool_mapping(), snapshot["tool_graph"])])
    profiles = []
    for case in snapshot["library_profiles"]:
        planner = builtin_planner(mapping_extensions=case["mapping_extensions"])
        actual = {
            "skill_ids": list(planner.skills),
            "graph_sha256": decision_digest(planner.graph.as_dict()),
            "tool_graph_sha256": decision_digest(planner.graph.tool_mapping()),
            "plan_sha256": decision_digest(planner.plan(case["query"])),
        }
        expected = {key: case[key] for key in actual}
        profiles.append((case["id"], actual, expected))
    compare("library-startup-profiles", profiles)
    retrieval, selection, plans, expansions = [], [], [], []
    for case in snapshot["cases"]:
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
        selected = select_skill_ids(
            definitions,
            case["query"],
            case["inferred_tools"],
            mapping_extensions=case["mapping_extensions"],
        )
        plan = SkillPlanner(
            definitions,
            extension_capabilities=extensions,
            mapping_extensions=case["mapping_extensions"],
        ).plan(case["query"], case["inferred_tools"])
        retrieval.append((case["id"], actual, case["matches"]))
        selection.append((case["id"], selected, case["selected_skill_ids"]))
        plans.append((case["id"], decision_digest(plan), case["plan_sha256"]))
        expansions.append(
            (case["id"], graph.expand(selected, case["query"]), case["capability_expansion"])
        )
    compare("skill-retrieval", retrieval)
    compare("skill-selection", selection)
    compare("complete-skill-plan", plans)
    compare("capability-expansion", expansions)
    compare(
        "tool-candidates",
        [
            (
                case["id"],
                graph.candidates(
                    graph.capabilities,
                    case["available_tools"],
                    case["inferred_tools"],
                    case["schemas"],
                    case["artifact_types"],
                ),
                case["expected"],
            )
            for case in snapshot["candidate_cases"]
        ],
    )
    grounding = []
    planner = SkillPlanner(definitions, extension_capabilities=extensions)
    for case in snapshot["grounding_cases"]:
        plan = planner.plan(case["query"], case["inferred_tools"])
        result = ground_plan(
            graph,
            plan,
            case["available_tools"],
            case["inferred_tools"],
            proposed_chain=case["proposed_chain"],
            tool_schemas=case["schemas"],
            tool_next=case["tool_next"],
            artifact_types=case["artifact_types"],
        )
        result.pop("tool_graph")
        grounding.append((case["id"], result, case["expected"]))
    compare("grounding", grounding)
    compare(
        "candidate-context",
        [
            (
                case["id"],
                json.loads(build_candidate_context(case["references"], case["max_chars"])),
                case["expected"],
            )
            for case in snapshot["context_cases"]
        ],
    )
    compare(
        "tool-feedback-classification",
        [
            (
                case["id"],
                assess_tool_execution(
                    case["output"],
                    input_format=case["input_format"],
                    attempt=case["attempt"],
                    max_attempts=case["max_attempts"],
                ).as_dict(),
                case["expected"],
            )
            for case in snapshot["feedback_cases"]
        ],
    )
    return checks


def alignment_report(contract, snapshot):
    if (
        snapshot["baseline_id"] != contract["baseline"]["id"]
        or snapshot["source_tree_sha256"] != contract["baseline"]["tree_sha256"]
        or snapshot["selection_sha256"] != contract["baseline"]["selection"]["sha256"]
    ):
        raise ValueError("Snapshot belongs to a different algorithm baseline")
    scope = load_tool_scope()
    if (
        set(snapshot["tool_graph"]["tool_to_capabilities"]) != set(scope["tool_ids"])
        or [s["id"] for s in snapshot["library"]] != scope["skill_ids"]
        or [c["capability_id"] for c in snapshot["graph"]["capabilities"]]
        != scope["capability_ids"]
    ):
        raise ValueError("Snapshot declarations differ from the selected GeoToolEval scope")
    checks = verify_components(snapshot)
    missing = [
        unit["id"]
        for unit in contract["algorithms"]
        if unit["implementation_status"] != "runtime_verified"
    ]
    connected = contract["acceptance"]["production_algorithm"] == contract["baseline"]["id"]
    equivalent = connected and not missing and all(check["passed"] for check in checks)
    return {
        "baseline_id": contract["baseline"]["id"],
        "source_tree_sha256": contract["baseline"]["tree_sha256"],
        "declarations": {
            "skills": len(snapshot["library"]),
            "capabilities": len(snapshot["graph"]["capabilities"]),
            "tools": len(snapshot["tool_graph"]["tool_to_capabilities"]),
        },
        "component_parity": all(check["passed"] for check in checks),
        "component_checks": checks,
        "component_decisions_compared": sum(check["cases"] for check in checks),
        "production_algorithm": contract["acceptance"]["production_algorithm"],
        "runtime_equivalent": equivalent,
        "runtime_pending_algorithms": missing,
        "scope": "Pure deterministic component parity; no model experiment or end-to-end equivalence claim.",
    }
