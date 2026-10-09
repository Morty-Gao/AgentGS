"""Lossless information controls and instructions for composable interventions."""

import hashlib
import json
from copy import deepcopy


def declaration_digest(declarations):
    return hashlib.sha256(
        json.dumps(declarations, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def planning_context(plan, structure):
    from .capabilities import capability_id

    declarations = deepcopy(plan.get("declarations", plan.get("steps", [])))
    expansion = deepcopy(plan.get("capability_expansion", {}))
    common = {
        "declaration_sha256": declaration_digest(declarations),
        "constraints": deepcopy(plan.get("constraints", [])),
        "missing_capabilities": deepcopy(plan.get("missing_capabilities", [])),
        "query_profile": deepcopy(plan.get("query_profile", {})),
        "selected_requirements": deepcopy(plan.get("selected_requirements")),
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
        capabilities = {row["capability_id"]: deepcopy(row) for row in plan.get("capabilities", [])}
        for declaration in declarations:
            node = dict(declaration)
            workflow = node.pop("workflow", {})
            node["workflow_attributes"] = {
                key: value for key, value in workflow.items() if key != "steps"
            }
            sid = node.get("id", node.get("skill_id"))
            nodes.append(node)
            for index, operation in enumerate(workflow.get("steps", [])):
                cid = capability_id(
                    operation.get("capability_id")
                    or operation.get("capability")
                    or operation.get("operation")
                )
                capabilities.setdefault(cid, {"capability_id": cid})
                edges.append(
                    {
                        "strategy_id": sid,
                        "capability_id": cid,
                        "position": index,
                        "operation": operation,
                    }
                )
        return {
            **common,
            "representation": "bipartite",
            "strategy_nodes": nodes,
            "capability_nodes": list(capabilities.values()),
            "incidence_edges": edges,
            "requirements": expansion,
        }
    if structure != "hierarchical":
        raise ValueError(f"Unknown planning structure: {structure}")
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
        "For each execution instance provide capability_id and skill_id to identify its declared candidate scope, including necessary predecessors you explicitly restore. The controller selects the first eligible tool in ascending tool-ID order, without model preference or candidate scores. Your tool field is a candidate proposal, not the final selection. Preserve the operation, dependencies and separate instances."
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
            "For a declared operation, use its strategy's skill_id and capability_id; do not substitute an execution step ID. Declare depends_on for every execution instance that needs a producer, and input_bindings for values it consumes. Listing a producer earlier or calling the same tool in another strategy does not satisfy that dependency. Independent cohorts require separate producer/consumer instances.",
            recovery,
            "Automatic predecessor expansion is "
            + (
                "enabled."
                if modules.predecessor_expansion
                else "disabled in both predecessor and support phases; dependency declarations still apply and you must explicitly plan required prerequisites."
            ),
            "Automatic support insertion is "
            + (
                "enabled."
                if modules.capability_completion
                else "disabled; reported gaps are diagnostics, not evidence of completed operations. You may explicitly plan a necessary declared operation; the controller will not automatically insert it."
            ),
            "Directed bridge insertion is "
            + (
                "enabled subject to declaration validation."
                if modules.directed_bridging
                else "disabled; do not assume the controller will insert connecting operations."
            ),
        ]
    )
