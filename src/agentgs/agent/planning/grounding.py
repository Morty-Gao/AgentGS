"""Bind a proposed capability plan to executable tools, preserving instances."""

from agentgs.agent.planning.capabilities import capability_id, workflow_steps


def ground_plan(
    graph,
    plan,
    available_tools,
    inferred_tools=None,
    *,
    proposed_chain=None,
    tool_schemas=None,
    tool_next=None,
    artifact_types=None,
):
    available = set(available_tools)
    inferred = list(dict.fromkeys(inferred_tools or plan.get("inferred_tools", [])))
    proposed = list(proposed_chain or [])
    schemas, next_edges = tool_schemas or {}, tool_next or {}
    artifacts = list(artifact_types or [])
    tool_skills = {}
    for sid, skill in graph.skills.items():
        for name in skill.get("tool_candidates", []):
            tool_skills.setdefault(name, sid)
    ids = [str(row["capability_id"]) for row in plan["capabilities"] if row.get("capability_id")]
    if not ids:
        for step in plan["steps"]:
            if step.get("capability"):
                ids.append(capability_id(step["capability"]))
            for row in workflow_steps(step):
                label = row.get("capability") or row.get("operation")
                if label:
                    ids.append(capability_id(label))
        ids = list(dict.fromkeys(ids))
    candidates = graph.candidates(ids, available, inferred, schemas, artifacts)
    unavailable, chain = [], []
    if proposed:
        for name in proposed:
            (chain if name in available else unavailable).append(name)
    elif inferred:
        chain = [name for name in inferred if name in available]
    else:
        for cid in ids:
            selected = next(
                (row["tool_name"] for row in candidates.get(cid, []) if row["available"]), None
            )
            if selected:
                chain.append(selected)
    bindings, recovery = [], []
    for step in plan["steps"]:
        sid = step["skill_id"]
        cid = capability_id(step.get("capability"))
        names = [row["tool_name"] for row in candidates.get(cid, []) if row["available"]]
        if not names:
            names = [
                name for name in graph.skills[sid].get("tool_candidates", []) if name in available
            ]
        selected = [name for name in chain if name in names]
        if not selected and inferred:
            selected = [
                name for name in inferred if tool_skills.get(name) == sid and name in available
            ]
        unavailable.extend(
            name for name in inferred if tool_skills.get(name) == sid and name not in available
        )
        for constraint in plan["constraints"]:
            if constraint["skill_id"] == sid:
                recovery.extend(constraint.get("recovery_skills", []))
        bindings.append(
            {
                "skill_id": sid,
                "capability_id": cid,
                "candidate_tools": names,
                "selected_tools": selected,
                "selection_mode": "validated_model_plan"
                if proposed
                else "inferred_hint"
                if inferred
                else "semantic_candidate",
                "schema_compatibility": [
                    {
                        "tool": name,
                        "input_format": schemas.get(name, {}).get("input_format")
                        or schemas.get(name, {}).get("Input", ""),
                        "output_format": schemas.get(name, {}).get("output_format")
                        or schemas.get(name, {}).get("Output", ""),
                        "artifact_types": artifacts,
                    }
                    for name in selected
                ],
            }
        )
    executions, previous = [], None
    for index, name in enumerate(chain, 1):
        schema = schemas.get(name, {})
        executions.append(
            {
                "execution_index": index,
                "tool_name": name,
                "instance_id": f"{name}#{index}",
                "skill_id": tool_skills.get(name, ""),
                "capability_id": next(
                    (
                        cid
                        for cid, rows in candidates.items()
                        if name in [row["tool_name"] for row in rows]
                    ),
                    "",
                ),
                "depends_on_tools": [previous] if previous else [],
                "input_contract": schema.get("input_format") or schema.get("Input", ""),
                "output_contract": schema.get("output_format") or schema.get("Output", ""),
                "dependency_edge_valid": previous is None
                or name in next_edges.get(previous, [])
                or previous == name,
                "artifact_types_before": artifacts,
            }
        )
        previous = name
    errors = list(unavailable)
    if proposed and not chain:
        errors.append("No proposed tool is available in the registry")
    return {
        "tool_chain": chain,
        "bindings": bindings,
        "unavailable_tools": list(dict.fromkeys(unavailable)),
        "recovery_skills": list(dict.fromkeys(recovery)),
        "execution_steps": executions,
        "validation": {
            "valid": not errors,
            "errors": list(dict.fromkeys(errors)),
            "schema_checked": bool(schemas),
            "dependency_edges_checked": bool(next_edges),
            "capability_graph_checked": bool(candidates),
            "missing_capabilities": list(plan.get("missing_capabilities", [])),
        },
        "capability_candidates": candidates,
        "tool_graph": graph.tool_mapping(),
    }
