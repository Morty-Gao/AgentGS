"""Native calls integrated into the original planning/execution/answer stages."""

import json

from agentgs.agent.planning.candidate_context import build_candidate_context
from agentgs.agent.planning.instances import ExecutionPlan, PlanStep
from agentgs.agent.planning.module_context import module_guidance, planning_context
from agentgs.agent.prompts import StagePrompt
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


def _skill_context(plan):
    return {key: plan.get(key) for key in (
        "capabilities", "constraints", "missing_capabilities", "query_profile"
    ) if plan.get(key)}


class ModelStages:
    def __init__(self, state, find_contract):
        self.state = state
        self.call = state.model
        self.find_contract = find_contract
        self.specs = {spec.id: spec for spec in state.context.tools.specs()}

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
        definitions = schema.pop("$defs", {})
        parameters = {"type": "object", "additionalProperties": False,
                      "properties": {"steps": {"type": "array", "minItems": 1, "items": schema}},
                      "required": ["steps"], "$defs": definitions}
        definition = {"type": "function", "function": {
            "name": "submit_tool_plan", "description": "Submit execution instances with explicit dependencies and inputs.",
            "parameters": parameters,
        }}
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
        policy_instruction = module_guidance(self.state.context.options.modules) if mode == "skill_guided" else ""
        prompt = StagePrompt(
            "Submit exactly one submit_tool_plan call with a nonempty steps array. "
            "Each step needs a unique id, tool and purpose. Declare depends_on using earlier step IDs; "
            "use input_bindings with step_id and path for upstream values. Put fixed public parameters "
            "in arguments. Preserve separate instances for distinct entities and periods. "
            "Use only the listed tools, include required prerequisites, and respect declared constraints. "
            "Do not add tools for narration or formatting. "
            "The same tool name does not imply the same computation. " + policy_instruction, context)
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
            except (ValueError, TypeError) as exc:
                error = str(exc)
            for raw in message.tool_calls:
                output = (
                    {"status": "error", "error": error}
                    if error
                    else {"status": "submitted", "steps": [step.model_dump(mode="json") for step in plan.steps], "execution": "pending_validation"}
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
            prompt = StagePrompt("Repair the rejected planning call. Submit exactly one submit_tool_plan call. " + policy_instruction, {**context, "validation_error": error})
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
            "Fix rejected arguments using the feedback. Do not repeat a failed input unchanged."
            if self.state.context.options.execution_recovery
            else "Execution recovery is disabled. Prepare the initial arguments carefully; no repair call follows execution failure."
        )
        prompt = StagePrompt(
            f"Call {name} exactly once for the current execution instance. Use observed upstream "
            "values and public inputs. Preserve entities, units and time scope. " + recovery_instruction,
            {"input_contract": input_contract, "output_contract": output_contract,
             "artifacts": registry_context, "execution_instance": execution_context,
             "feedback": feedback, "recent_attempts": list(history[-3:])})
        return await self.call(
            prompt,
            stage="tool_argument_generation",
            tools=[definition],
            tool_choice={"type": "function", "function": {"name": name}},
        )

    async def generate_output(self, question, output, *, execution_incomplete=False):
        self.state.begin_protocol_stage()
        prompt = StagePrompt(
            "Return the final answer directly in the task's required format. No more tools are available. "
            "Report supported results and identify missing values; do not fabricate measurements.",
            {"execution_incomplete": execution_incomplete,
             "evidence": _compact_context(output, 14000),
             "output_contract": self.find_contract(question)})
        return await self.call(prompt, stage="final_answer_generation")
