"""Fixed tool selection within each execution instance's declared scope."""

from .instances import ExecutionPlan
from .scopes import required_instances, scope_tools


def select_first_valid(graph, plan, execution_plan, available_tools, specs, *, fixed_tools=None):
    available = set(available_tools)
    fixed_tools = fixed_tools or {}
    active = {row["capability_id"] for row in plan["capabilities"]}
    requirements = required_instances(graph, plan, execution_plan)
    allowed_scopes = {(row["skill_id"], row["capability_id"]) for row in requirements}
    active.update(row["capability_id"] for row in requirements)
    selected_skills = {row["skill_id"] for row in plan["steps"]}
    selected_skills.update(plan.get("capability_expansion", {}).get("skill_ids", []))
    selected_skills.update(row["skill_id"] for row in requirements)
    steps, decisions = [], []
    for index, step in enumerate(execution_plan.steps, 1):
        cid = step.capability_id
        if cid is None:
            matched = [key for key in active if step.tool in graph.capability_tools.get(key, [])]
            if len(matched) != 1:
                raise ValueError(f"A5 requires an unambiguous capability_id for {step.id}")
            cid = matched[0]
        if cid not in active:
            raise ValueError(f"Capability is outside the active requirements: {cid}")
        scopes = {}
        for sid in selected_skills:
            if allowed_scopes and (sid, cid) not in allowed_scopes:
                continue
            if step.skill_id is not None and sid != step.skill_id:
                continue
            choices = scope_tools(graph, sid, cid)
            if choices:
                scopes[sid] = tuple(sorted(set(choices)))
        if len(scopes) != 1:
            raise ValueError(f"A5 requires an unambiguous skill_id candidate scope for {step.id}")
        candidates = next(iter(scopes.values()))
        if step.tool not in candidates:
            raise ValueError(f"Proposed tool is outside the instance scope: {step.id}")
        rejected, chosen = [], None
        fields = set(step.arguments) | set(step.input_bindings)
        for name in candidates:
            reason = None
            spec = specs.get(name)
            if name not in available or spec is None:
                reason = "not_available"
            elif index in fixed_tools and name != fixed_tools[index]:
                reason = "declared_bridge_binding_requires_original_tool"
            else:
                schema = spec.input_schema
                properties = schema.get("properties", {})
                if schema.get("additionalProperties") is False and fields - set(properties):
                    reason = "fixed_input_fields_outside_schema"
            if reason:
                rejected.append({"tool": name, "reason": reason})
            else:
                chosen = name
                break
        if chosen is None:
            raise ValueError(f"No legal tool candidate for {step.id}: {rejected}")
        steps.append(
            step.model_copy(
                update={"tool": chosen, "capability_id": cid, "skill_id": next(iter(scopes))},
                deep=True,
            )
        )
        decisions.append(
            {
                "instance_id": step.id,
                "capability_id": cid,
                "skill_scopes": sorted(scopes),
                "candidate_order": list(candidates),
                "selected_tool": chosen,
                "rejected": rejected,
                "input_validation": "fixed_fields_checked; generated_arguments_validated_at_execution",
            }
        )
    return ExecutionPlan(steps), decisions
