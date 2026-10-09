"""Lossless information controls and instructions for composable interventions."""

import hashlib
import json
from copy import deepcopy


def declaration_digest(declarations):
    return hashlib.sha256(
        json.dumps(declarations, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def planning_context(plan, structure):
    declarations = deepcopy(plan.get("declarations", plan.get("steps", [])))
    expansion = deepcopy(plan.get("capability_expansion", {}))
    common = {
        "declaration_sha256": declaration_digest(declarations),
        "constraints": deepcopy(plan.get("constraints", [])),
        "missing_capabilities": deepcopy(plan.get("missing_capabilities", [])),
        "query_profile": deepcopy(plan.get("query_profile", {})),
    }
    if structure == "flat_tools":
        # The same declarations remain visible, but no ordered capability plan is supplied.
        return {
            **common,
            "representation": "flat_tools",
            "strategy_declarations": declarations,
            "admitted_capabilities": sorted(
                row["capability_id"] for row in plan.get("capabilities", [])
            ),
            "capability_declarations": sorted(
                deepcopy(plan.get("capabilities", [])), key=lambda row: row["capability_id"]
            ),
            "support_insertions": expansion.get("completed_capabilities", []),
            "bridge_declarations": deepcopy(expansion.get("shared_bridge", {})),
        }
    if structure == "bipartite":
        nodes, edges = [], []
        for declaration in declarations:
            node = dict(declaration)
            workflow = node.pop("workflow", {})
            node["workflow_attributes"] = {
                key: value for key, value in workflow.items() if key != "steps"
            }
            sid = node.get("id", node.get("skill_id"))
            nodes.append(node)
            for index, operation in enumerate(workflow.get("steps", [])):
                edges.append({"strategy_id": sid, "position": index, "operation": operation})
        return {
            **common,
            "representation": "bipartite",
            "strategy_nodes": nodes,
            "capability_nodes": deepcopy(plan.get("capabilities", [])),
            "incidence_edges": edges,
            "requirements": expansion,
        }
    return {
        **common,
        "representation": "hierarchical",
        "strategy_declarations": declarations,
        "capabilities": deepcopy(plan.get("capabilities", [])),
        "requirements": expansion,
        "hypergraph": deepcopy(plan.get("hypergraph", {})),
    }


def module_guidance(modules):
    structure = {
        "hierarchical": "Organize the execution instances by the supplied capability requirements and their scoped dependencies, then select tools for those operations.",
        "flat_tools": "Plan directly over tool execution instances. Strategy and dependency declarations are reference information; no ordered capability execution plan is supplied. Derive the tool sequence yourself.",
        "bipartite": "Plan using the supplied strategy and capability nodes and incidence edges. Edge positions and operation attributes preserve declaration order and dependency scope; membership alone is not a data dependency.",
    }[modules.planning_structure]
    selection = (
        "For each execution instance provide capability_id and skill_id to identify its declared candidate scope. The controller selects the first eligible tool in ascending tool-ID order, without model preference or candidate scores. Preserve the operation, dependencies and separate instances."
        if modules.tool_selection == "first_valid"
        else "Select a suitable tool for each execution instance within its declared capability and strategy scope."
    )
    recovery = (
        "Use execution feedback to repair failed operations; successful inputs may be reused by the runtime."
        if modules.execution_recovery
        else "Execution recovery is disabled. Prepare and validate the initial plan and inputs normally; after an execution failure there will be no argument repair or replanning."
    )
    return " ".join(
        [
            structure,
            selection,
            recovery,
            "Automatic predecessor expansion is "
            + (
                "enabled."
                if modules.predecessor_expansion
                else "disabled; dependency declarations still apply and explicit model planning may include prerequisites."
            ),
            "Automatic support insertion is "
            + (
                "enabled."
                if modules.capability_completion
                else "disabled; reported gaps are diagnostics, not evidence of completed operations."
            ),
            "Directed bridge insertion is "
            + (
                "enabled subject to declaration validation."
                if modules.directed_bridging
                else "disabled; do not assume the controller will insert connecting operations."
            ),
        ]
    )
