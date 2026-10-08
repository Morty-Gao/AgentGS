"""Parameter planning, declared bindings and the original atomic repair loop."""

import json

from agentgs.agent.execution import references, scope
from agentgs.agent.execution.bindings import (
    _parse_generated,
    prepare_tool_input,
    tool_input_is_ready,
)
from agentgs.agent.execution.feedback import run_tool_with_feedback
from agentgs.agent.execution.scientific import (
    overlay_scientific_tool_schema,
    preserve_scientific_arguments,
    requires_scientific_argument_generation,
    scientific_input_issues,
)
from agentgs.agent.planning.bridge import apply_shared_bridge_inputs
from agentgs.contracts.catalog import get_tool_spec
from agentgs.contracts.models import ToolCall


def legacy_arguments(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return {"input_text": str(text)}
    if isinstance(value, dict):
        return value
    return {"x": value} if isinstance(value, list) else {"value": value}


class ExecutionService:
    def __init__(self, state, stages):
        self.state, self.stages = state, stages

    async def physical(self, name, text, files):
        call = ToolCall(tool=name, arguments=legacy_arguments(text))
        result = await self.state.context.tools.execute(call)
        self.state.results.append(result)
        self.state.save()
        if result.raw_output is not None:
            return result.raw_output
        if result.succeeded:
            return json.dumps(result.data, ensure_ascii=False)
        return "Tool function execution error, " + result.error.code + ": " + result.error.message

    async def comparison(self, name, text, files):
        """Keep the optional counterfactual outside the main result/trace chain."""
        result = await self.state.context.tools.execute(
            ToolCall(tool=name, arguments=legacy_arguments(text))
        )
        if result.raw_output is not None:
            return result.raw_output
        if result.succeeded:
            return json.dumps(result.data, ensure_ascii=False)
        return "Tool function execution error, " + result.error.code + ": " + result.error.message

    async def run(
        self, name, question, registry, total_output, current_output="", execution_context=None
    ):
        row = get_tool_spec(name)
        if row is None:
            raise ValueError(f"Tool is outside the frozen scope: {name}")
        info = overlay_scientific_tool_schema(
            name, {"input_format": row["Input"], "output_format": row["Output"]}
        )
        scientific = requires_scientific_argument_generation(name)
        bridge = bool((execution_context or {}).get("shared_bridge"))

        async def input_provider(attempt, history):
            if history and history[-1].feedback.error_type == "transport_error":
                return history[-1].tool_input
            explicit = references.requires_explicit_source(name, registry)
            if attempt == 1 and not scientific and not bridge and not explicit:
                prepared, event = prepare_tool_input(
                    name, "{}", question, current_output, total_output, registry, execution_context
                )
                if tool_input_is_ready(name, prepared):
                    event.update(
                        attempt=attempt, source="schema_artifact_binding", model_call_skipped=True
                    )
                    self.state.trace.append(event)
                    return prepared
            feedback = history[-1].to_prompt_context() if history else None
            try:
                generated = await self.stages.generate_plan_input(
                    question,
                    current_output,
                    name,
                    references.argument_contract(info["input_format"], registry, tool_name=name),
                    info["output_format"],
                    model=None,
                    previous_answer=total_output,
                    execution_feedback=feedback,
                    attempt_history=[row.to_prompt_context() for row in history],
                )
            except Exception as exc:
                if scientific or bridge:
                    raise
                generated = "{}"
                self.state.trace.append(
                    {
                        "type": "tool_argument_fallback",
                        "tool_name": name,
                        "attempt": attempt,
                        "reason": type(exc).__name__,
                        "fallback": "authoritative_resource_and_upstream_output",
                    }
                )
            reference_binding = (
                references.bind_parameters(name, _parse_generated(generated), registry)
                if scope.active()
                else None
            )
            if reference_binding is not None:
                bound, event = reference_binding
                if bridge:
                    diagnostic = {
                        "code": "UNSUPPORTED_REFERENCE_CONTEXT",
                        "detail": "Explicit references with shared bridge projections are outside this migration",
                    }
                    bound = {references.DIAGNOSTICS: [diagnostic]}
                    event["diagnostics"] = [diagnostic]
                event["attempt"] = attempt
                self.state.trace.append(event)
                return json.dumps(bound, ensure_ascii=False)
            prepared, event = prepare_tool_input(
                name, generated, question, current_output, total_output, registry, execution_context
            )
            prepared = preserve_scientific_arguments(name, generated, prepared)
            if bridge:
                prepared = json.dumps(_parse_generated(generated), ensure_ascii=False)
                prepared = apply_shared_bridge_inputs(prepared, execution_context)
                event["shared_bridge_binding"] = True
            if scientific:
                event["scientific_argument_contract"] = {
                    "source": "model_generated_explicit_bindings",
                    "issues": scientific_input_issues(name, prepared),
                }
            event["attempt"] = attempt
            self.state.trace.append(event)
            return prepared

        return await run_tool_with_feedback(
            tool_name=name,
            file_path_list=self.state.task.file_path_list,
            input_provider=input_provider,
            tool_runner=self.physical,
            input_format=info["input_format"],
            max_attempts=3,
            trace_list=self.state.trace,
            producer_step=f"tool_step_{execution_context['execution_index']:02d}"
            if execution_context and isinstance(execution_context.get("execution_index"), int)
            else "",
        )
