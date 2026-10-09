"""Fixed tool selection within each execution instance's declared scope."""

from .capabilities import capability_id, workflow_steps
from .instances import ExecutionPlan


def select_first_valid(graph, plan, execution_plan, available_tools, specs, *, fixed_tools=None):
    available = set(available_tools)
    fixed_tools = fixed_tools or {}
    active = {row["capability_id"] for row in plan["capabilities"]}
    selected_skills = {row["skill_id"] for row in plan["steps"]}
    selected_skills.update(plan.get("capability_expansion", {}).get("skill_ids", []))
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
            if step.skill_id is not None and sid != step.skill_id:
                continue
            skill = graph.skills[sid]
            for operation in workflow_steps(skill):
                identity = capability_id(
                    operation.get("capability_id")
                    or operation.get("capability")
                    or operation.get("operation")
                )
                if identity == cid:
                    choices = (operation.get("tool_selection") or {}).get(
                        "candidates"
                    ) or skill.get("tool_candidates", [])
                    scopes[sid] = tuple(sorted(set(choices)))
        if not scopes or len(set(scopes.values())) != 1:
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
        steps.append(step.model_copy(update={"tool": chosen, "capability_id": cid}, deep=True))
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
