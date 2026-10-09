"""Bind a proposed capability plan to executable tools, preserving instances."""

from agentgs.agent.planning.capabilities import capability_id, workflow_steps
from agentgs.agent.planning.scopes import required_instances, scope_tools


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
    tool_specs=None,
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
    ids = list(
        dict.fromkeys(
            [
                *ids,
                *(row["capability_id"] for row in required_instances(graph, plan, execution_plan)),
            ]
        )
    )
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
        from agentgs.agent.planning.module_selection import select_first_valid

        if execution_plan is None or tool_specs is None:
            raise ValueError("first_valid requires execution instances and registered tool schemas")
        execution_plan, _ = select_first_valid(graph, plan, execution_plan, available, tool_specs)
        chain = list(execution_plan)
    elif tool_selection != "model":
        raise ValueError(f"Unknown tool selection policy: {tool_selection}")
    if execution_plan is not None and list(execution_plan) != chain:
        raise ValueError("Grounding requires execution instances aligned with the proposed chain")
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
    scope_constraints = required_instances(graph, plan, execution_plan)
    ancestors = {}
    if execution_plan is not None:
        for step in execution_plan.steps:
            ancestors[step.id] = set(step.depends_on)
            for dep in step.depends_on:
                ancestors[step.id].update(ancestors.get(dep, ()))

    def matches_scope(index, sid, cid):
        if chain[index] not in scope_tools(graph, sid, cid):
            return False
        if execution_plan is None:
            return True
        step = execution_plan.steps[index]
        return (step.capability_id in (None, cid)) and (sid is None or step.skill_id in (None, sid))

    ambiguous = set()
    if execution_plan is not None:
        for i, step in enumerate(execution_plan.steps):
            matching = {
                node["id"]
                for node in scope_constraints
                if matches_scope(i, node["skill_id"], node["capability_id"])
            }
            if len(matching) > 1:
                errors.append(f"Ambiguous capability or strategy assignment: {step.id}")
                ambiguous.add(i)
    positions = {}
    for node in scope_constraints:
        cid = node["capability_id"]
        matches = [
            i
            for i in range(len(chain))
            if i not in ambiguous and matches_scope(i, node["skill_id"], cid)
        ]
        if matches:
            positions[node["id"]] = matches
    for node in scope_constraints:
        if node["id"] not in positions:
            errors.append(f"Missing required capability: {node['id']}")
        for dep in node["depends_on"]:
            targets = positions.get(node["id"], [])

            def linked(before, after):
                if before >= after:
                    return False
                if execution_plan is None:
                    return True
                producer, consumer = execution_plan.steps[before], execution_plan.steps[after]
                return producer.id in ancestors[consumer.id]

            if dep not in positions or any(
                not any(linked(before, after) for before in positions.get(dep, []))
                for after in targets
            ):
                errors.append(f"Unsatisfied capability dependency: {dep} -> {node['id']}")
    for missing in plan.get("missing_capabilities", []):
        sid, separator, cid = missing.partition("::")
        if not separator:
            sid, cid = None, missing
        if not any(matches_scope(i, sid, cid) for i in range(len(chain))):
            errors.append(f"Unresolved capability: {missing}")
    if execution_plan:
        by_id = {step.id: i for i, step in enumerate(execution_plan.steps)}
        for i, step in enumerate(execution_plan.steps):
            if step.capability_id is not None:
                if step.capability_id not in graph.capabilities:
                    errors.append(f"Unknown capability on {step.id}: {step.capability_id}")
                elif step.tool not in scope_tools(graph, step.skill_id, step.capability_id):
                    errors.append(f"Tool outside declared capability scope: {step.id}")
            if step.skill_id is not None and step.skill_id not in graph.skills:
                errors.append(f"Unknown Skill on {step.id}: {step.skill_id}")
            executions[i]["instance_id"] = step.id
            executions[i]["skill_id"] = step.skill_id or executions[i]["skill_id"]
            executions[i]["capability_id"] = step.capability_id or executions[i]["capability_id"]
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
        "required_instances": scope_constraints,
        "tool_graph": graph.tool_mapping(),
    }
