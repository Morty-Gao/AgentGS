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
    execution_plan=None,
    tool_selection="model",
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
    if tool_selection == "first_valid":
        for i, name in enumerate(chain):
            cid = execution_plan.steps[i].capability_id if execution_plan else None
            matched = [key for key, rows in candidates.items() if any(r["tool_name"] == name for r in rows)]
            cid = cid or (matched[0] if len(matched) == 1 else None)
            if cid is None:
                unavailable.append(f"Ambiguous capability for tool-selection control: {name}")
                continue
            eligible = sorted(row["tool_name"] for row in candidates.get(cid, []) if row["available"])
            if eligible:
                chain[i] = eligible[0]
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
    scope_constraints = plan.get("capability_expansion", {}).get("instances", [])
    positions = {}
    for node in scope_constraints:
        cid = node["capability_id"]
        choices = graph.capability_tools.get(cid, [])
        matches = [i for i, name in enumerate(chain) if name in choices]
        if matches:
            positions[node["id"]] = matches
    for node in scope_constraints:
        if node["id"] not in positions:
            errors.append(f"Missing required capability: {node['id']}")
        for dep in node["depends_on"]:
            if dep not in positions or not any(
                before < after for before in positions[dep] for after in positions.get(node["id"], [])
            ):
                errors.append(f"Unsatisfied capability dependency: {dep} -> {node['id']}")
    for missing in plan.get("missing_capabilities", []):
        cid = missing.split("::")[-1]
        if not any(name in graph.capability_tools.get(cid, []) for name in chain):
            errors.append(f"Unresolved capability: {missing}")
    if execution_plan:
        by_id = {step.id: i for i, step in enumerate(execution_plan.steps)}
        for i, step in enumerate(execution_plan.steps):
            executions[i]["instance_id"] = step.id
            executions[i]["depends_on_tools"] = list(step.depends_on)
            for dep in step.depends_on:
                if dep not in by_id or by_id[dep] >= i:
                    errors.append(f"Invalid execution dependency: {dep} -> {step.id}")
            executions[i]["dependency_edge_valid"] = all(
                dep in by_id and by_id[dep] < i for dep in step.depends_on
            )
    else:
        for row in executions:
            if next_edges and not row["dependency_edge_valid"]:
                errors.append(f"Undeclared serial tool dependency before {row['tool_name']}")
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
            "schema_checked": False,
            "schemas_available": bool(schemas),
            "schema_status": "deferred_to_argument_validation",
            "dependency_edges_checked": bool(execution_plan or next_edges or scope_constraints),
            "capability_graph_checked": bool(scope_constraints),
            "missing_capabilities": list(plan.get("missing_capabilities", [])),
        },
        "capability_candidates": candidates,
        "tool_graph": graph.tool_mapping(),
    }
