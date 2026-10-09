"""Declared operation scopes shared by grounding and tool-selection policies."""

from copy import deepcopy

from .capabilities import capability_id, workflow_steps


def scope_tools(graph, skill_id, cid):
    if skill_id is None:
        return list(graph.capability_tools.get(cid, []))
    skill = graph.skills.get(skill_id, {})
    operations = workflow_steps(skill)
    for operation in operations:
        identity = capability_id(
            operation.get("capability_id")
            or operation.get("capability")
            or operation.get("operation")
        )
        if identity == cid:
            return list(
                (operation.get("tool_selection") or {}).get("candidates")
                or skill.get("tool_candidates", [])
            )
    if not operations and cid in graph.scopes.get(skill_id, {}):
        return list(skill.get("tool_candidates", []))
    return []


def required_instances(graph, plan, execution_plan=None):
    """Include declared prerequisites for validation, without inserting executions."""
    nodes = deepcopy(plan.get("capability_expansion", {}).get("instances", []))
    known = {row["id"] for row in nodes}

    def add(sid, cid):
        identity = f"{sid}::{cid}"
        if identity not in known and cid in graph.scopes.get(sid, {}):
            nodes.append(
                {
                    "id": identity,
                    "skill_id": sid,
                    "capability_id": cid,
                    "depends_on": [f"{sid}::{dep}" for dep in graph.scopes[sid][cid]],
                }
            )
            known.add(identity)

    for route in plan.get("capability_expansion", {}).get("shared_bridge", {}).get("accepted", []):
        owners = bridge_owners(route)
        for cid, sid in owners.items():
            identity = f"{sid}::{cid}"
            if identity not in known:
                nodes.append(
                    {"id": identity, "skill_id": sid, "capability_id": cid, "depends_on": []}
                )
                known.add(identity)
        by_id = {row["id"]: row for row in nodes}
        for edge in route["data_edges"]:
            source, target = edge["source"], edge["target"]
            ref = f"{owners[source]}::{source}"
            deps = by_id[f"{owners[target]}::{target}"]["depends_on"]
            if ref not in deps:
                deps.append(ref)
    if execution_plan is not None:
        for step in execution_plan.steps:
            if step.skill_id is not None and step.capability_id is not None:
                add(step.skill_id, step.capability_id)
    for node in nodes:
        for ref in node["depends_on"]:
            sid, separator, cid = ref.partition("::")
            if separator:
                add(sid, cid)
    return nodes


def bridge_owners(route):
    path = route["capability_path"]
    return {
        cid: contract.get("skill_id")
        or (
            route["source_skill_id"]
            if cid == path[0]
            else route["target_skill_id"]
            if cid == path[-1]
            else route["skill_id"]
        )
        for cid, contract in route["execution_contract"].items()
    }
