"""Native calls integrated into the original planning/execution/answer stages."""

import json

from agentgs.agent.planning.candidate_context import build_candidate_context
from agentgs.agent.planning.instances import ExecutionPlan, PlanStep
from agentgs.agent.planning.module_context import module_guidance, planning_context
from agentgs.agent.prompts import StagePrompt, budget_exhausted_instruction
from agentgs.agent.state import ModelProtocolError


def _compact_context(value, limit=12000):
    """Keep structured values parseable; never splice through JSON or bindings."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            if len(value) <= limit:
                return value
            return {"excerpt": value[:limit], "truncated": True, "original_chars": len(value)}
    if isinstance(value, (dict, list, tuple)):
        return value
    return str(value or "")


class ModelStages:
    def __init__(self, state, find_contract):
        self.state = state
        self.call = state.model
        self.find_contract = find_contract
        self.specs = {spec.id: spec for spec in state.context.tools.specs()}

    async def select_capabilities(self, retrieval, *, completion=False):
        """Adopt task requirements before expanding any candidate strategy."""
        from agentgs.agent.planning.capabilities import capability_id, workflow_steps

        scopes = {}
        for skill in retrieval["declarations"]:
            operations = workflow_steps(skill)
            scopes[skill["id"]] = (
                {
                    capability_id(
                        row.get("capability_id") or row.get("capability") or row.get("operation")
                    )
                    for row in operations
                }
                if operations
                else {capability_id((skill.get("goal") or {}).get("capability") or skill["id"])}
            )
        if completion:
            allowed = retrieval["support_candidates"]
            scopes = {
                sid: ids & {row["capability_id"] for row in allowed if row["skill_id"] == sid}
                for sid, ids in scopes.items()
            }
            scopes = {sid: ids for sid, ids in scopes.items() if ids}
        if not scopes:
            return {}
        definition = {
            "type": "function",
            "function": {
                "name": "select_capabilities",
                "description": "Select the task's required operations from candidate strategies.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "requirements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "skill_id": {"type": "string", "enum": list(scopes)},
                                    "capability_id": {
                                        "type": "string",
                                        "enum": sorted(set().union(*scopes.values())),
                                    },
                                },
                                "required": ["skill_id", "capability_id"],
                            },
                        }
                    },
                    "required": ["requirements"],
                },
            },
        }
        instruction = (
            "Call select_capabilities exactly once. The retrieved strategies are candidates, "
            "not obligations. Choose only the operations needed by the original task and public "
            "resources, using exact skill_id and capability_id values from the declarations. "
            "Do not adopt unrelated operations merely because they share a domain or input type. "
            "A requested intermediate operation can be a goal; do not import its downstream steps. "
            "Select goal operations before prerequisite expansion; the next stage handles declared "
            "dependencies. The selection is an unordered set, not an execution plan. "
            "Return an empty requirements array when none of the candidates applies; direct tool "
            "planning remains available. Do not generate measurements or tool arguments here."
        )
        if completion:
            instruction = (
                "Call select_capabilities exactly once to review possible missing support operations. "
                "These are lexical suggestions, not mandatory requirements. Select only operations "
                "actually needed by the task that are absent from the adopted requirements. "
                "Unmatched query words or a shared domain do not prove a missing operation. "
                "Return an empty requirements array when the adopted operations are sufficient. "
                "Use only the skill_id/capability_id pairs in support_candidates; do not reselect "
                "adopted goals or restore prerequisites disabled by the predecessor intervention."
            )
        stage = "capability_completion" if completion else "capability_selection"
        prompt = StagePrompt(instruction, retrieval)
        self.state.begin_protocol_stage()
        for attempt in range(self.state.context.options.argument_attempts):
            message = await self.call(
                prompt,
                stage=stage,
                tools=[definition],
                tool_choice={"type": "function", "function": {"name": "select_capabilities"}},
            )
            selected, error = {}, None
            try:
                if (
                    len(message.tool_calls) != 1
                    or message.tool_calls[0].function.name != "select_capabilities"
                ):
                    raise ValueError("Return exactly one select_capabilities call")
                data = json.loads(message.tool_calls[0].function.arguments)
                if (
                    not isinstance(data, dict)
                    or set(data) != {"requirements"}
                    or not isinstance(data["requirements"], list)
                ):
                    raise ValueError("Selection requires a requirements array")
                for row in data["requirements"]:
                    if not isinstance(row, dict) or set(row) != {"skill_id", "capability_id"}:
                        raise ValueError("Each requirement needs skill_id and capability_id")
                    sid, cid = row["skill_id"], row["capability_id"]
                    if (
                        not isinstance(sid, str)
                        or not isinstance(cid, str)
                        or cid not in scopes.get(sid, set())
                    ):
                        raise ValueError(
                            "Selected operation is outside the retrieved strategy scope"
                        )
                    if cid not in selected.setdefault(sid, []):
                        selected[sid].append(cid)
            except (ValueError, TypeError) as exc:
                error = str(exc)
            for call in message.tool_calls:
                self.state.conversation.finish_tool(
                    call.id,
                    json.dumps(
                        {"status": "error", "error": error}
                        if error
                        else {
                            "status": "selected",
                            "requirements": selected,
                            "execution": "pending_planning",
                        }
                    ),
                    operation="planning",
                    status="failed" if error else "success",
                )
            await self.state.save()
            if error is None:
                await self.state.emit(stage, {"requirements": selected})
                return selected
            self.state.protocol_failure()
            await self.state.emit(
                "planning_protocol_error", {"attempt": attempt + 1, "error": error}
            )
            prompt = StagePrompt(instruction, {**retrieval, "validation_error": error})
        raise ModelProtocolError("Capability selection protocol attempts exhausted")

    async def generate_tools_plan(
        self,
        question,
        relations,
        previous_answer="",
        *,
        planning_mode=None,
        skill_plan=None,
        tool_hints=(),
    ):
        mode = planning_mode or self.state.context.options.planning_mode
        if mode not in {"tool_only", "skill_guided"}:
            raise ValueError(f"Unknown planning mode: {mode}")
        if mode != self.state.context.options.planning_mode:
            raise ValueError("Planning mode must match the resolved task policy")
        if mode == "tool_only" and skill_plan is not None:
            raise ValueError("Tool-only planning cannot consume a Skill plan")
        names = list(
            dict.fromkeys(
                row[0]
                for row in relations
                if len(row) >= 2 and row[1] == "has the functionality that" and row[0] in self.specs
            )
        )
        if not names:
            raise ValueError("No executable planning candidates")
        schema = PlanStep.model_json_schema()
        schema["properties"]["tool"]["enum"] = names
        if mode == "tool_only":
            for key in ("capability_id", "skill_id"):
                schema["properties"].pop(key)
        definitions = schema.pop("$defs", {})
        parameters = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"steps": {"type": "array", "minItems": 1, "items": schema}},
            "required": ["steps"],
            "$defs": definitions,
        }
        definition = {
            "type": "function",
            "function": {
                "name": "submit_tool_plan",
                "description": "Submit execution instances with explicit dependencies and inputs.",
                "parameters": parameters,
            },
        }
        context = {
            "candidates": json.loads(build_candidate_context(relations)),
            "routing_hints": list(tool_hints),
            "execution_feedback": _compact_context(previous_answer),
        }
        if mode == "skill_guided":
            context["skill_constraints"] = planning_context(
                skill_plan or {}, self.state.context.options.modules.planning_structure
            )
            context["module_policy"] = self.state.context.options.modules.model_dump()
            context["selected_requirements"] = (skill_plan or {}).get("selected_requirements")
        policy_instruction = (
            module_guidance(self.state.context.options.modules) if mode == "skill_guided" else ""
        )
        if (
            mode == "skill_guided"
            and (skill_plan or {}).get("selected_requirements") == {}
            and not (skill_plan or {}).get("steps")
        ):
            policy_instruction = (
                "No strategy was adopted. Plan directly over the supplied tools; omit skill_id "
                "and capability_id. There is no declared scope for fixed-order tool selection."
            )
        prompt = StagePrompt(
            "Submit exactly one submit_tool_plan call with a nonempty steps array. "
            "Each step needs a unique id, tool and purpose. Declare depends_on using earlier step IDs; "
            "use input_bindings with step_id and path for upstream values. Put fixed public parameters "
            "in arguments. Preserve separate instances for distinct entities and periods. "
            "Even identical tools and arguments require separate IDs for independent executions. "
            "Keep an ID stable across replanning only for the same execution instance. "
            "Binding paths use string object keys or nonnegative integer array indices; "
            "use transform=list only to wrap the projected value in a one-element list. "
            "Use only the listed tools, include required prerequisites, and respect declared constraints. "
            "Do not add tools for narration or formatting. "
            "The same tool name does not imply the same computation. " + policy_instruction,
            context,
        )
        # Native protocol repairs have their own bounded loop, never a text parser.
        self.state.begin_protocol_stage()
        for attempt in range(self.state.context.options.argument_attempts):
            message = await self.call(
                prompt,
                stage="skill_tool_planning" if mode == "skill_guided" else "tool_planning",
                tools=[definition],
                tool_choice={"type": "function", "function": {"name": "submit_tool_plan"}},
            )
            error = None
            plan = None
            try:
                if (
                    len(message.tool_calls) != 1
                    or message.tool_calls[0].function.name != "submit_tool_plan"
                ):
                    raise ValueError("Return exactly one submit_tool_plan function call")
                data = json.loads(message.tool_calls[0].function.arguments)
                if not isinstance(data, dict) or set(data) != {"steps"} or not data["steps"]:
                    raise ValueError("Planning arguments require a nonempty steps array")
                plan = ExecutionPlan(data["steps"])
                if any(name not in names for name in plan):
                    raise ValueError("Plan contains an unavailable tool")
                if (
                    mode == "tool_only"
                    or self.state.context.options.modules.tool_selection != "first_valid"
                ):
                    for step in plan.steps:
                        schema = self.specs[step.tool].input_schema
                        fields = set(step.arguments) | set(step.input_bindings)
                        if schema.get("additionalProperties") is False and fields - set(
                            schema.get("properties", {})
                        ):
                            raise ValueError(
                                f"Unknown planned input fields for {step.id}: {sorted(fields - set(schema.get('properties', {})))}"
                            )
                if mode == "tool_only" and any(
                    step.capability_id is not None or step.skill_id is not None
                    for step in plan.steps
                ):
                    raise ValueError(
                        "Tool-only plans cannot specify Skill or Capability assignments"
                    )
            except (ValueError, TypeError) as exc:
                error = str(exc)
            for raw in message.tool_calls:
                output = (
                    {"status": "error", "error": error}
                    if error
                    else {
                        "status": "submitted",
                        "steps": [
                            step.model_dump(mode="json", exclude_none=True) for step in plan.steps
                        ],
                        "execution": "pending_validation",
                    }
                )
                self.state.conversation.finish_tool(
                    raw.id,
                    json.dumps(output, ensure_ascii=False),
                    operation="planning",
                    status="failed" if error else "success",
                )
            await self.state.save()
            if error is None:
                return plan
            self.state.protocol_failure()
            await self.state.emit(
                "planning_protocol_error", {"attempt": attempt + 1, "error": error}
            )
            prompt = StagePrompt(
                "Repair the rejected planning call. Submit exactly one submit_tool_plan call. "
                + policy_instruction,
                {**context, "validation_error": error},
            )
        raise ModelProtocolError("Native tool-plan protocol attempts exhausted")

    async def generate_tool_call(
        self,
        name,
        question,
        registry_context,
        execution_context,
        input_contract,
        output_contract,
        feedback,
        history,
    ):
        spec = self.specs.get(name)
        self.state.begin_protocol_stage()
        if spec is None:
            raise ValueError(f"Unavailable tool: {name}")
        definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.description + "\nOutput: " + output_contract,
                "parameters": spec.input_schema,
            },
        }
        recovery_instruction = (
            "Fix rejected arguments using the feedback. On recovery you may correct literal arguments from the plan; upstream input_bindings remain fixed to observed producer values. Do not repeat a failed input unchanged."
            if self.state.context.options.execution_recovery
            else "Execution recovery is disabled. Prepare the initial arguments carefully; no repair call follows execution failure."
        )
        prompt = StagePrompt(
            f"Call {name} exactly once for the current execution instance. Use observed upstream "
            "values and public inputs. Preserve entities, units and time scope. "
            + recovery_instruction,
            {
                "input_contract": input_contract,
                "output_contract": output_contract,
                "artifacts": registry_context,
                "execution_instance": execution_context,
                "feedback": feedback,
                "recent_attempts": list(history[-3:]),
            },
        )
        context = prompt.context
        for _ in range(self.state.context.options.argument_attempts):
            message = await self.call(
                prompt,
                stage="tool_argument_generation",
                tools=[definition],
                tool_choice={"type": "function", "function": {"name": name}},
            )
            try:
                if len(message.tool_calls) != 1 or message.tool_calls[0].function.name != name:
                    raise ValueError(f"Return exactly one call to {name}")
                arguments = json.loads(message.tool_calls[0].function.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Function arguments must be a JSON object")
                properties = spec.input_schema.get("properties", {})
                if spec.input_schema.get("additionalProperties") is False and set(arguments) - set(
                    properties
                ):
                    raise ValueError("Unknown function argument fields")
                return message
            except (ValueError, TypeError) as exc:
                for call in message.tool_calls:
                    self.state.conversation.finish_tool(
                        call.id,
                        json.dumps({"status": "error", "error": str(exc)}),
                        operation="protocol",
                        status="protocol_error",
                    )
                await self.state.save()
                self.state.protocol_failure()
                prompt = StagePrompt(
                    f"Repair the function-call format. No scientific tool executed for the rejected call. Return exactly one call to {name}. "
                    + recovery_instruction,
                    {**context, "protocol_error": str(exc)},
                )
        raise ModelProtocolError("Function-call format repair budget exhausted")

    async def generate_output(
        self, question, output, *, execution_incomplete=False, budget_exhausted=None
    ):
        self.state.begin_protocol_stage()
        completion_instruction = (
            "Execution is incomplete. Distinguish supported partial results from missing results; do not claim the task is complete. When using the framework labels, set IsCompleted:Not Completed. "
            if execution_incomplete
            else "Claim completion only when all requested results are supported by successful evidence. "
        )
        budget_instruction = (
            budget_exhausted_instruction(budget_exhausted) + " " if budget_exhausted else ""
        )
        prompt = StagePrompt(
            budget_instruction
            + "Return the final answer directly in the task's required format. No more tools are available. "
            + completion_instruction
            + "Report supported results and identify missing values; do not fabricate measurements or compute unobserved scientific results in the answer.",
            {
                "execution_incomplete": execution_incomplete,
                "evidence": _compact_context(output, 14000),
                "output_contract": self.find_contract(question),
            },
        )
        return await self.call(prompt, stage="final_answer_generation")
